#!/usr/bin/env python
"""Two locked four-stage ablations, always restricted to train/validation.

Each arm has one independent CPU worker. Historical condition-repair auxiliary
checkpoints are immutable parents for the adapter arm. There is no test-scoring
entry point and no restart that can overwrite or silently reconfigure a study.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stageworld_tcwm.data import file_sha256
from stageworld_tcwm.modality_schema import SCHEMA
from stageworld_tcwm.timeline_data import split_indices
from stageworld_tcwm.timeline_training import source_manifest, verify_source
from stageworld_tcwm.research_neural import INPUT_FIELDS
from run_four_stage import (SEEDS, SPLIT_SHA256, METRICS, normalized_metrics,
                            require_train_validation_only, training_schedule,
                            safe_evaluation_batch)
from run_modality import DEFAULT_DATA_ROOT, FIXED_COUNTS, FIXED_PROTOCOL

ARMS = ("pcr_only_frozen", "ct_pcr_adapter")
PARENT = Path(os.environ.get("GASTRIC_V6_PARENT", "runs/condition_repair"))
DESIGN = ROOT / "docs" / "FOUR_STAGE_AUXILIARY_ADAPTER_EXPERIMENTS_20261001.md"
RECIPE = {
    "arms": list(ARMS), "workers": 2, "threads_per_worker": 2,
    "adapter_initialization": "same_seed_condition_repair_aux_best",
    "pcr_only_initialization": "train_from_scratch",
    "checkpoint_selection": "minimum_validation_terminal_NLL_including_epoch0",
    "best_trained_reporting": "minimum_validation_NLL_among_positive_step_checkpoints_diagnostic_only",
    "test_policy": "disabled_no_test_entry_point",
    "incomplete_study_policy": "fail_closed_use_new_output_directory_no_resume",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


class TrainValidationTensor:
    """Store only allowed rows, preserving original global row indices."""

    def __init__(self, tensor, allowed_rows, cohort_size):
        allowed_rows = torch.as_tensor(allowed_rows, dtype=torch.long)
        if tensor.shape[0] != cohort_size:
            raise ValueError("Tensor patient axis differs from cohort")
        self._lookup = torch.full((cohort_size,), -1, dtype=torch.long)
        self._lookup[allowed_rows] = torch.arange(len(allowed_rows))
        self._values = tensor[allowed_rows].clone()

    def __getitem__(self, rows):
        rows = torch.as_tensor(rows, dtype=torch.long).cpu()
        if (rows < 0).any() or (rows >= len(self._lookup)).any():
            raise ValueError("Patient row outside cohort")
        local = self._lookup[rows]
        if (local < 0).any():
            raise ValueError("Test rows are inaccessible to this train/validation experiment")
        return self._values[local]


def load_train_validation(data_root):
    """Map the cache, then materialize only permitted train/validation values."""
    from stageworld_tcwm.four_stage_training import TARGET_FIELDS

    payload = torch.load(Path(data_root) / "cohort.pt", map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Require the audited modality-event cohort schema")
    cohort = SimpleNamespace(ids=payload["ids"], encoders=payload["encoders"],
                             metadata=payload["metadata"])
    roles = split_indices(cohort, read(Path(data_root) / "split.json"))
    if len(cohort.ids) != 651 or {key: len(value) for key, value in roles.items()} != FIXED_COUNTS:
        raise ValueError("Require fixed651 train456/validation65/test130")
    metadata = cohort.metadata
    if (metadata.get("protocol") != FIXED_PROTOCOL or metadata.get("split_seed") != 17
            or metadata.get("split_ratio") != [7, 1, 2]):
        raise ValueError("Require fixed seed17 and ratio7:1:2 data preparation")
    roles = {key: roles[key] for key in ("train", "validation")}
    allowed = torch.cat(list(roles.values()))
    cohort.tensors = {key: TrainValidationTensor(payload["tensors"][key], allowed, len(cohort.ids))
                      for key in dict.fromkeys(INPUT_FIELDS + TARGET_FIELDS)}
    return cohort, roles


def verify_receipt(path, directory):
    receipt = read(path)
    if receipt.get("test_evaluated", False):
        raise ValueError("Parent must be a train/validation-only comparison")
    for name, digest in receipt["artifacts_sha256"].items():
        artifact = Path(directory) / name
        if not artifact.resolve().is_relative_to(Path(directory).resolve()):
            raise ValueError("Receipt artifact escapes its experiment")
        if file_sha256(artifact) != digest:
            raise ValueError(f"Frozen parent artifact changed: {artifact.name}")
    return receipt


def parent_manifest(parent, seeds, data_hashes):
    """Verify historical receipts without applying today's source to old runs."""
    parent = Path(parent).resolve()
    protocol = read(parent / "study_protocol.json")
    receipt = verify_receipt(parent / "completion.json", parent)
    if (receipt["protocol_sha256"] != file_sha256(parent / "study_protocol.json")
            or protocol.get("test_policy") != "disabled" or protocol.get("diagnostic")
            or protocol["data_sha256"] != data_hashes or protocol["seeds"] != list(SEEDS)):
        raise ValueError("Parent protocol is not the completed fixed-split repair comparison")
    filenames = ["study_protocol.json", "completion.json", "summary.json",
                 "per_seed_train_validation.csv", "validation_reproduction.json"]
    for seed in seeds:
        base = f"runs/four_stage_seed{seed}"
        run = parent / base
        run_receipt = verify_receipt(run / "completion.json", run)
        metrics = read(run / "metrics.json")
        require_train_validation_only(metrics)
        if (run_receipt["protocol_sha256"] != receipt["protocol_sha256"]
                or metrics.get("seed") != seed or metrics.get("diagnostic")
                or metrics.get("config", {}).get("medical_condition_encoding") != "masked_inapplicable"):
            raise ValueError("Parent checkpoint identity mismatch")
        filenames.extend(f"{base}/{name}" for name in
                         ("aux_best.pt", "aux_last.pt", "aux_history.json", "metrics.json", "completion.json"))
    baseline = parent / "runs" / "D01"
    baseline_receipt = verify_receipt(baseline / "completion.json", baseline)
    if baseline_receipt["protocol_sha256"] != receipt["protocol_sha256"]:
        raise ValueError("Parent baseline protocol mismatch")
    require_train_validation_only(read(baseline / "metrics.json"))
    filenames += ["runs/D01/metrics.json", "runs/D01/completion.json"]
    return {name: file_sha256(parent / name) for name in filenames}


def make_protocol(args):
    from stageworld_tcwm import four_stage_training as training

    data = args.data_root.resolve()
    data_hashes = {name: file_sha256(data / name) for name in ("cohort.pt", "split.json")}
    if data_hashes["split.json"] != SPLIT_SHA256:
        raise ValueError("Require the original frozen patient split")
    seeds = [17] if args.diagnostic else list(SEEDS)
    return {
        "schema": "four-stage-auxiliary-adapter-ablation-v1", "source_root": str(ROOT),
        "source_sha256": source_manifest(ROOT), "python": sys.executable,
        "design_path": str(DESIGN), "design_sha256": file_sha256(DESIGN),
        "data_root": str(data), "data_sha256": data_hashes, "split_counts": FIXED_COUNTS,
        "seeds": seeds, "arms": list(ARMS), "diagnostic": bool(args.diagnostic),
        "recipe": RECIPE, "training_defaults": training.DEFAULTS,
        "training_experiments": getattr(training, "EXPERIMENTS", {}),
        "training_schedule": training_schedule(training.DEFAULTS, args.diagnostic),
        "parent_root": str(args.parent.resolve()),
        "parent_sha256": parent_manifest(args.parent, seeds, data_hashes),
        "test_evaluated": False, "test_policy": "disabled_no_test_entry_point",
        "concept_validated": False, "causal_strategy_validated": False,
        "historical_test_exposure": True, "source_snapshot_usage": "provenance_only",
    }


def verify_identity(protocol):
    from stageworld_tcwm import four_stage_training as training

    expected_seeds = [17] if protocol["diagnostic"] else list(SEEDS)
    if (protocol["source_root"] != str(ROOT) or protocol["seeds"] != expected_seeds
            or protocol["arms"] != list(ARMS) or protocol["recipe"] != RECIPE
            or protocol["test_policy"] != "disabled_no_test_entry_point"
            or protocol["test_evaluated"] is not False
            or protocol["training_defaults"] != training.DEFAULTS
            or protocol["training_experiments"] != getattr(training, "EXPERIMENTS", {})
            or protocol["split_counts"] != FIXED_COUNTS
            or protocol["data_sha256"]["split.json"] != SPLIT_SHA256):
        raise ValueError("Frozen source, recipe, seed or split identity mismatch")
    verify_source(ROOT, protocol["source_sha256"])
    if file_sha256(protocol["design_path"]) != protocol["design_sha256"]:
        raise ValueError("Experiment design changed")
    for root_key, manifest_key in (("data_root", "data_sha256"), ("parent_root", "parent_sha256")):
        for name, digest in protocol[manifest_key].items():
            if file_sha256(Path(protocol[root_key]) / name) != digest:
                raise ValueError(f"Frozen {root_key} artifact changed: {name}")


def bind_protocol(args):
    if any(args.out.iterdir()):
        raise FileExistsError("Refuse existing output or unsafe resume; use a fresh study directory")
    protocol = make_protocol(args)
    write(protocol, args.out / "study_protocol.json")
    for name, digest in protocol["source_sha256"].items():
        target = args.out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        target.chmod(0o600)
        if file_sha256(target) != digest:
            raise ValueError("Source changed while taking provenance snapshot")
    shutil.copyfile(DESIGN, args.out / "source_snapshot" / DESIGN.name)
    (args.out / "source_snapshot" / DESIGN.name).chmod(0o600)
    return protocol


@contextmanager
def worker_lock(path):
    with Path(path).open("a+") as handle:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Worker already running") from error
        yield


def job_key(arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Undeclared experiment job")
    return f"{arm}_seed{seed}"


def complete_job(out, arm, seed, protocol):
    run = out / "runs" / job_key(arm, seed)
    for name in ("metrics.json", "inference.pt", "history.json", "best.pt", "last.pt", "aux_best.pt"):
        if not (run / name).is_file():
            raise ValueError(f"Incomplete training output: {name}")
    metrics = read(run / "metrics.json")
    require_train_validation_only(metrics)
    if (metrics["seed"] != seed or metrics.get("experiment") != arm
            or metrics["diagnostic"] != protocol["diagnostic"]
            or metrics["partition_counts"] != {"train": 456, "validation": 65}
            or metrics["steps_per_epoch"] != protocol["training_schedule"]["steps_per_epoch"]):
        raise ValueError("Training result differs from its declared job")
    for phase in (metrics, metrics["auxiliary"]):
        for epoch, step in (("selected_epoch", "selected_step"), ("completed_epochs", "completed_steps")):
            if not math.isclose(phase[epoch], phase[step] / metrics["steps_per_epoch"], abs_tol=1e-9):
                raise ValueError("Training epoch count differs from step count")
    for artifact in run.rglob("*"):
        if artifact.is_file():
            artifact.chmod(0o600)
    write({"job": job_key(arm, seed), "protocol_sha256": file_sha256(out / "study_protocol.json"),
           "test_evaluated": False, "diagnostic": protocol["diagnostic"],
           "artifacts_sha256": {str(path.relative_to(run)): file_sha256(path)
                                for path in sorted(run.rglob("*")) if path.is_file()},
           "completed_at_unix": time.time()}, run / "completion.json")


def completed_metrics(out, arm, seed, protocol):
    run = out / "runs" / job_key(arm, seed)
    receipt = verify_receipt(run / "completion.json", run)
    if (receipt["protocol_sha256"] != file_sha256(out / "study_protocol.json")
            or receipt["job"] != job_key(arm, seed) or receipt["diagnostic"] != protocol["diagnostic"]):
        raise ValueError("Training completion identity mismatch")
    metrics = read(run / "metrics.json")
    require_train_validation_only(metrics)
    return metrics


def run_worker(args):
    from stageworld_tcwm import four_stage_training as training

    out, arm = args.out.resolve(), args.worker_arm
    protocol = read(out / "study_protocol.json")
    verify_identity(protocol)
    if not Path(training.__file__).resolve().is_relative_to(ROOT):
        raise ValueError("Training must execute original repository code")
    torch.set_num_threads(2)
    status_path = out / f"worker_{arm}.json"
    with worker_lock(out / f"worker_{arm}.lock"):
        status = {"status": "running", "pid": os.getpid(), "arm": arm,
                  "completed_seeds": [], "test_evaluated": False}
        write(status, status_path)
        try:
            cohort, rows = load_train_validation(Path(protocol["data_root"]))
            for seed in protocol["seeds"]:
                verify_identity(protocol)
                status["active_seed"] = seed
                write(status, status_path)
                auxiliary = (Path(protocol["parent_root"]) / "runs" / f"four_stage_seed{seed}" / "aux_best.pt"
                             if arm == "ct_pcr_adapter" else None)
                training.train_four_stage(cohort, rows["train"], rows["validation"],
                                          out / "runs" / job_key(arm, seed), seed=seed,
                                          device="cpu", diagnostic=protocol["diagnostic"],
                                          experiment=arm, auxiliary_checkpoint=auxiliary)
                verify_identity(protocol)
                complete_job(out, arm, seed, protocol)
                status["completed_seeds"].append(seed)
                write(status, status_path)
            status.update(status="pass", completed_at_unix=time.time())
            write(status, status_path)
        except BaseException as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            write(status, status_path)
            raise
    return 0


def write_csv(rows, path):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(path).chmod(0o600)


def freeze_and_reproduce(out, protocol):
    from stageworld_tcwm.four_stage_training import load_four_stage_export, evaluate_four_stage

    plan = {}
    for arm in ARMS:
        for seed in protocol["seeds"]:
            metrics = completed_metrics(out, arm, seed, protocol)
            key = job_key(arm, seed)
            path = out / "runs" / key / "inference.pt"
            plan[key] = {"artifact_sha256": file_sha256(path), "selected_step": metrics["selected_step"]}
    write({"jobs": plan, "test_evaluated": False}, out / "validation_export_plan.json")
    cohort, roles = load_train_validation(Path(protocol["data_root"]))
    batch = safe_evaluation_batch(cohort, roles["validation"], "cpu")
    checks = {}
    for arm in ARMS:
        for seed in protocol["seeds"]:
            key = job_key(arm, seed)
            path = out / "runs" / key / "inference.pt"
            if file_sha256(path) != plan[key]["artifact_sha256"]:
                raise ValueError("Frozen inference export changed")
            model, payload = load_four_stage_export(path)
            if payload["seed"] != seed or payload["selected_step"] != plan[key]["selected_step"]:
                raise ValueError("Frozen export selection identity mismatch")
            actual, _ = evaluate_four_stage(model, batch)
            expected = completed_metrics(out, arm, seed, protocol)["validation"]
            differences = {name: abs(actual[name] - expected[name]) for name in METRICS}
            if max(differences.values()) > 2e-6:
                raise ValueError(f"Validation export reproduction failed: {key}")
            checks[key] = {"pass": True, "absolute_differences": differences}
    write({"jobs": checks, "test_evaluated": False,
           "export_plan_sha256": file_sha256(out / "validation_export_plan.json")},
          out / "validation_reproduction.json")


def summarize(out, protocol):
    rows = []
    for arm in ARMS:
        for seed in protocol["seeds"]:
            metrics = completed_metrics(out, arm, seed, protocol)
            history = read(out / "runs" / job_key(arm, seed) / "history.json")
            trained = min((entry for entry in history if entry["step"] > 0),
                          key=lambda entry: entry["validation"]["nll"])
            auxiliary = metrics["auxiliary"]
            row = {"arm": arm, "seed": seed, "diagnostic": protocol["diagnostic"],
                   "selected_epoch": metrics["selected_epoch"], "selected_kind": metrics["selected_kind"],
                   "completed_epochs": metrics["completed_epochs"], "best_trained_epoch": trained["epoch"],
                   "auxiliary_selected_epoch": auxiliary["selected_epoch"],
                   "auxiliary_completed_epochs": auxiliary["completed_epochs"],
                   "auxiliary_reused": arm == "ct_pcr_adapter",
                   "auxiliary_actual_new_steps": 0 if arm == "ct_pcr_adapter" else auxiliary["completed_steps"],
                   "parameter_count": metrics["parameter_count"],
                   "terminal_trainable_parameter_count": metrics["terminal_trainable_parameter_count"]}
            for role in ("train", "validation"):
                for metric in METRICS:
                    row[f"{role}_{metric}"] = normalized_metrics(metrics[role])[metric]
                    row[f"best_trained_{role}_{metric}"] = normalized_metrics(trained[role])[metric]
                for stage, auxiliary_metrics in (("auxiliary", auxiliary[role]),
                        ("terminal_auxiliary", metrics.get("terminal_auxiliary", {}).get(role, {}))):
                    for name in ("pcr_nll", "pcr_auc", "pcr_ap", "pcr_brier", "pcr_patients",
                                 "ct1_smooth_l1", "ct1_patients"):
                        row[f"{stage}_{role}_{name}"] = auxiliary_metrics.get(name)
            rows.append(row)
    write_csv(rows, out / "per_seed_train_validation.csv")
    baseline = read(Path(protocol["parent_root"]) / "runs" / "D01" / "metrics.json")
    reference = {"source": "completed_condition_repair_D01_train_validation_only",
                 **{role: normalized_metrics(baseline[role]) for role in ("train", "validation")}}
    write(reference, out / "baseline_train_validation.json")
    summary = {"diagnostic": protocol["diagnostic"], "test_evaluated": False,
               "concept_validated": False, "seed_selection": False, "ensemble": False,
               "best_trained_is_diagnostic_only": True, "baseline": reference, "arms": {}}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        measures = [key for key in selected[0] if key.startswith(
            ("train_", "validation_", "best_trained_train_", "best_trained_validation_"))]
        summary["arms"][arm] = {"n_seeds": len(selected),
            "trained_dynamics_selected": sum(row["selected_epoch"] > 0 for row in selected),
            "metrics": {key: {"mean": statistics.mean(row[key] for row in selected),
                              "sd": statistics.stdev(row[key] for row in selected) if len(selected) > 1 else 0.}
                        for key in measures}}
    write(summary, out / "summary.json")
    return summary


def main(argv=None):
    args = parse_args(argv)
    os.umask(0o077)
    args.out = args.out.resolve()
    torch.set_num_threads(2)
    if args.worker_arm:
        return run_worker(args)
    # Atomic creation prevents two coordinators from racing into the same run.
    args.out.mkdir(parents=True, exist_ok=False)
    args.out.chmod(0o700)
    status = {"status": "initializing", "pid": os.getpid(), "test_evaluated": False,
              "started_at_unix": time.time(), "diagnostic": args.diagnostic}
    workers, logs = [], []
    try:
        protocol = bind_protocol(args)
        verify_identity(protocol)
        status.update(status="running", stage="two_parallel_train_validation_workers")
        write(status, args.out / "status.json")
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONPATH=str(ROOT / "src"))
        for arm in ARMS:
            log = (args.out / f"worker_{arm}.log").open("w")
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), "--out", str(args.out), "--worker-arm", arm]
            workers.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        status["workers"] = {arm: process.pid for arm, process in zip(ARMS, workers)}
        write(status, args.out / "status.json")
        while any(process.poll() is None for process in workers):
            if any(process.poll() not in (None, 0) for process in workers):
                raise RuntimeError("An ablation worker failed; inspect its private log and status")
            time.sleep(1)
        if any(process.returncode != 0 for process in workers):
            raise RuntimeError("An ablation worker failed; inspect its private log and status")
        verify_identity(protocol)
        status["stage"] = "validation_reproduction"
        write(status, args.out / "status.json")
        freeze_and_reproduce(args.out, protocol)
        summarize(args.out, protocol)
        verify_identity(protocol)
        status.update(status="pass", stage="diagnostic_complete" if args.diagnostic else "train_validation_complete",
                      completed_at_unix=time.time(), completed_jobs=len(ARMS) * len(protocol["seeds"]))
        write(status, args.out / "status.json")
        artifacts = ("status.json", "validation_export_plan.json", "validation_reproduction.json",
                     "per_seed_train_validation.csv", "baseline_train_validation.json", "summary.json")
        write({"protocol_sha256": file_sha256(args.out / "study_protocol.json"), "test_evaluated": False,
               "diagnostic": args.diagnostic,
               "artifacts_sha256": {name: file_sha256(args.out / name) for name in artifacts},
               "completed_at_unix": time.time()}, args.out / "completion.json")
        print(json.dumps({"status": "pass", "out": str(args.out), "test_evaluated": False}), flush=True)
        return 0
    except BaseException as error:
        for process in workers:
            if process.poll() is None:
                process.terminate()
        for process in workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        status.update(status="failed", error=f"{type(error).__name__}: {error}", failed_at_unix=time.time())
        write(status, args.out / "status.json")
        raise
    finally:
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
