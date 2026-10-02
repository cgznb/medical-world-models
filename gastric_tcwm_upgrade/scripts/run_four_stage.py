#!/usr/bin/env python
"""Locked ten-seed, four-stage experiment with validation-only checkpoint choice.

The currently executable mode is explicitly weakly supervised. It does not claim
measured longitudinal concept validation. Every seed is frozen and its validation
export reproduced before the requested complete test audit can begin.
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
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stageworld_tcwm.data import file_sha256
from stageworld_tcwm.timeline_data import TimelineCohort, split_indices
from stageworld_tcwm.timeline_training import binary_metrics, source_manifest, verify_source
from stageworld_tcwm.research_neural import INPUT_FIELDS
from run_modality import DEFAULT_DATA_ROOT, FIXED_COUNTS, validate_fixed_data

SEEDS = (17, 29, 43, 71, 101, 137, 173, 211, 257, 307)
SPLIT_SHA256 = "2b6b17fe70291c94673e0fbf31ac482828a55b318acc7407e13f216151edad48"
DESIGN = ROOT / "docs" / "MULTISTAGE_STATE_REVISION_20261001.md"
METRICS = ("auc", "nll", "ap", "brier")
TRAIN_FILES = ("metrics.json", "inference.pt", "history.json", "best.pt", "last.pt",
               "aux_best.pt", "aux_last.pt", "aux_history.json")


def read(path):
    return json.loads(Path(path).read_text())


def write(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("weak_four_stage",), required=True,
                   help="Explicit existing-data scope; not validated measured-concept dynamics")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--diagnostic", action="store_true",
                   help="One seed, short train/validation only; never test; use separate output directory")
    p.add_argument("--train-validation-only", action="store_true",
                   help="Full ten-seed budget and validation reproduction, with test evaluation disabled")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error("threads must be positive")
    return args


def require_seed_set(seeds, diagnostic=False):
    expected = (17,) if diagnostic else SEEDS
    if tuple(seeds) != expected:
        raise ValueError("Require the complete frozen ten-seed list in its declared order")


def training_schedule(defaults, diagnostic=False):
    """Persist the epoch budget and its exact update-count interpretation."""
    batch_size = defaults["batch_size"]
    steps_per_epoch = math.ceil(FIXED_COUNTS["train"] / batch_size)
    return {
        "unit": "epoch", "batch_size": batch_size, "drop_last": False,
        "training_patients": FIXED_COUNTS["train"], "steps_per_epoch": steps_per_epoch,
        "maximum_epochs": {phase: defaults[f"{phase}_max_epochs"]
                           for phase in ("auxiliary", "terminal")},
        "maximum_steps": {phase: 2 if diagnostic else defaults[f"{phase}_max_epochs"] * steps_per_epoch
                          for phase in ("auxiliary", "terminal")},
        "warmup_epochs": defaults["warmup_epochs"],
        "patience_epochs": defaults["patience_epochs"],
        "validation_interval_epochs": defaults["validation_interval_epochs"],
        "early_stopping_count_starts_at_epoch": defaults["warmup_epochs"] + 1,
        "earliest_early_stopping_epoch": defaults["warmup_epochs"] + defaults["patience_epochs"],
        "warmup_policy": "validate for checkpoint selection; do not accumulate early-stopping patience",
        "early_stopping_policy": "after warmup, stop after consecutive patience_epochs without validation improvement",
        "diagnostic": diagnostic,
    }


def make_protocol(args):
    from stageworld_tcwm.four_stage_training import DEFAULTS

    validate_fixed_data(args.data_root)
    if file_sha256(args.data_root / "split.json") != SPLIT_SHA256:
        raise ValueError("Must reuse the exact frozen 456/65/130 patient split")
    return {
        "schema": "gastric-four-stage-ten-seed-v3", "mode": args.mode,
        "source_root": str(ROOT), "source_sha256": source_manifest(ROOT),
        "python": sys.executable, "design_path": str(DESIGN),
        "design_sha256": file_sha256(DESIGN),
        "data_root": str(args.data_root.resolve()),
        "data_sha256": {name: file_sha256(args.data_root / name) for name in ("cohort.pt", "split.json")},
        "split_counts": FIXED_COUNTS, "split_seed": 17, "split_ratio": [7, 1, 2],
        "seeds": [17] if args.diagnostic else list(SEEDS), "diagnostic": args.diagnostic,
        "train_validation_only": args.train_validation_only,
        "device": args.device, "threads": args.threads, "parallel_jobs": 1,
        "training_defaults": DEFAULTS,
        "training_schedule": training_schedule(DEFAULTS, args.diagnostic),
        "diagnostic_step_override": {"auxiliary": 2, "terminal": 2} if args.diagnostic else None,
        "baseline": "D01_fixed_train_only_clinical_logistic_regression_seed17",
        "endpoint": "retrospectively_recorded_binary_recurrence_no_fixed_followup_horizon",
        "checkpoint_selection": "validation_only_per_seed; no test-based seed selection",
        "test_policy": "disabled" if args.diagnostic or args.train_validation_only else "all_ten_frozen_exports_after_all_validation_reproductions",
        "seed_aggregation": "mean_and_sample_SD_of_metrics; no_probability_ensemble",
        "concept_validated": False, "causal_strategy_claim": False,
        "historical_test_exposure": True, "independent_external_validation": False,
        "source_snapshot_usage": "provenance_only; execute original repository source",
        "requested_test_scope": ("none for this train/validation comparison" if args.diagnostic or args.train_validation_only
                                 else "all ten seeds; user explicitly requested each seed test result"),
    }


def verify_identity(protocol):
    require_seed_set(protocol["seeds"], protocol["diagnostic"])
    if protocol["source_root"] != str(ROOT):
        raise ValueError("This run must execute the original training repository")
    verify_source(ROOT, protocol["source_sha256"])
    if file_sha256(Path(protocol["design_path"])) != protocol["design_sha256"]:
        raise ValueError("Four-stage design changed during the source-locked experiment")
    data = Path(protocol["data_root"])
    for name, digest in protocol["data_sha256"].items():
        if file_sha256(data / name) != digest:
            raise ValueError("Frozen data or split changed")
    if protocol["data_sha256"]["split.json"] != SPLIT_SHA256 or protocol["split_counts"] != FIXED_COUNTS:
        raise ValueError("Incorrect fixed patient split")


def bind_protocol(args):
    candidate = make_protocol(args)
    path = args.out / "study_protocol.json"
    if path.exists():
        if not args.resume:
            raise ValueError("Existing experiment requires --resume; historical outputs are never overwritten")
        if read(path) != candidate:
            raise ValueError("Protocol changed; use a fresh experiment directory")
    elif any(p.name != "study.lock" for p in args.out.iterdir()):
        raise ValueError("Refuse nonempty directory without its frozen protocol")
    else:
        write(candidate, path)
    snapshot = args.out / "source_snapshot"
    for name, digest in candidate["source_sha256"].items():
        dest = snapshot / name
        if dest.exists():
            if file_sha256(dest) != digest:
                raise ValueError("Provenance source snapshot changed")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    return candidate


@contextmanager
def study_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "study.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This experiment already has a live coordinator") from exc
        yield


def require_train_validation_only(metrics):
    if any(metrics.get(key) is not None for key in ("test", "test_metrics")) or any(
        metrics.get(key, False) for key in ("test_evaluated", "test_scored")
    ):
        raise ValueError("Training jobs must not evaluate the test set")
    for role in ("train", "validation"):
        normalized_metrics(metrics[role])


def normalized_metrics(value):
    result = {"auc": value["auc"], "nll": value["nll"],
              "ap": value.get("ap", value.get("average_precision")), "brier": value["brier"],
              "n": value.get("n", value.get("patients")), "positive": value.get("positive")}
    if any(result[name] is None or not math.isfinite(float(result[name])) for name in METRICS):
        raise ValueError("Incomplete or nonfinite binary metrics")
    return result


def run_key(seed):
    return f"four_stage_seed{seed}"


def run_receipt(out, key, files, protocol):
    run = out / "runs" / key
    for name in files:
        if not (run / name).is_file():
            raise ValueError(f"Incomplete training output {key}: missing {name}")
    metrics = read(run / "metrics.json")
    require_train_validation_only(metrics)
    if key != "D01":
        if metrics.get("seed") != int(key.rsplit("seed", 1)[1]):
            raise ValueError("Training result seed differs from its frozen job")
        for name in ("selected_step", "completed_steps"):
            if not isinstance(metrics.get(name), int) or metrics[name] < 0:
                raise ValueError(f"Invalid training {name}")
        if metrics["selected_step"] > metrics["completed_steps"] or not metrics.get("selected_kind"):
            raise ValueError("Invalid selected checkpoint metadata")
        steps_per_epoch = protocol["training_schedule"]["steps_per_epoch"]
        if metrics.get("steps_per_epoch") != steps_per_epoch:
            raise ValueError("Training steps_per_epoch differs from the frozen schedule")
        for phase in (metrics, metrics.get("auxiliary", {})):
            for name, step_name in (("selected_epoch", "selected_step"), ("completed_epochs", "completed_steps")):
                value, step = phase.get(name), phase.get(step_name)
                if (not isinstance(value, (int, float)) or not math.isfinite(value)
                        or not isinstance(step, int) or step < 0
                        or not math.isclose(value, step / steps_per_epoch, abs_tol=1e-9)):
                    raise ValueError(f"Invalid training {name}: must equal steps / steps_per_epoch")
            if phase["selected_step"] > phase["completed_steps"] or not phase.get("stop_reason"):
                raise ValueError("Invalid phase checkpoint or stop metadata")
    verify_identity(protocol)
    receipt = {"job": key, "protocol_sha256": file_sha256(out / "study_protocol.json"),
               "diagnostic": protocol["diagnostic"], "test_evaluated": False,
               "artifacts_sha256": {str(p.relative_to(run)): file_sha256(p)
                                    for p in sorted(run.rglob("*")) if p.is_file() and p.name != "completion.json"},
               "completed_at_unix": time.time()}
    write(receipt, run / "completion.json")
    return metrics


def completed_metrics(out, key, protocol):
    run = out / "runs" / key
    path = run / "completion.json"
    if not path.exists():
        return None
    receipt = read(path)
    if (receipt["job"] != key or receipt["protocol_sha256"] != file_sha256(out / "study_protocol.json")
            or receipt["diagnostic"] != protocol["diagnostic"] or receipt["test_evaluated"]):
        raise ValueError("Training completion receipt belongs to a different experiment")
    for name, digest in receipt["artifacts_sha256"].items():
        if file_sha256(run / name) != digest:
            raise ValueError(f"Completed training artifact changed: {key}/{name}")
    metrics = read(run / "metrics.json")
    require_train_validation_only(metrics)
    return metrics


def preserve_incomplete_run(out, key):
    run = out / "runs" / key
    if run.exists():
        archive = out / "interrupted_attempts" / f"{key}_{time.time_ns()}"
        archive.parent.mkdir(exist_ok=True)
        run.rename(archive)


def freeze_exports(out, protocol):
    """All ten completed jobs are required before test export selection is frozen."""
    if protocol["diagnostic"]:
        raise ValueError("Diagnostics cannot freeze exports for test")
    require_seed_set(protocol["seeds"])
    jobs = {}
    for key in [run_key(seed) for seed in SEEDS] + ["D01"]:
        metrics = completed_metrics(out, key, protocol)
        if metrics is None:
            raise ValueError("All ten seeds and the fixed reference must finish before test")
        filename = "inference.joblib" if key == "D01" else "inference.pt"
        export = out / "runs" / key / filename
        jobs[key] = {"artifact": str(export.resolve()), "artifact_sha256": file_sha256(export),
                     "metrics_sha256": file_sha256(out / "runs" / key / "metrics.json"),
                     "seed": 17 if key == "D01" else int(key.rsplit("seed", 1)[1]),
                     "selected_step": metrics.get("selected_step"),
                     "selected_epoch": metrics.get("selected_epoch"),
                     "selected_kind": metrics.get("selected_kind", "fixed_clinical_logistic_regression")}
    plan = {"schema": "four-stage-all-seed-test-plan-v1", "jobs": jobs,
            "protocol_sha256": file_sha256(out / "study_protocol.json"),
            "split_sha256": SPLIT_SHA256, "test_patients": 130,
            "checkpoint_selection": "validation_only", "test_based_selection": False,
            "concept_validated": False, "historical_test_exposure": True,
            "test_policy": protocol.get("test_policy")}
    target = out / "evaluation_plan.json"
    if target.exists():
        if read(target) != plan:
            raise ValueError("Frozen export list changed after test plan registration")
    else:
        write(plan, target)
    return plan


def safe_evaluation_batch(cohort, indices, device):
    """Recurrence labels are scoring targets; CT1/pCR/pathology never enter this batch."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    fields = tuple(INPUT_FIELDS) + ("binary", "binary_valid")
    batch = {key: cohort.tensors[key][indices].to(device) for key in fields}
    batch["row_index"] = indices.to(device)
    if not batch["binary_valid"].all():
        raise ValueError("Every patient in the fixed denominator requires an observed endpoint")
    return batch


def validate_predictions(predictions, cohort, indices):
    indices = torch.as_tensor(indices, dtype=torch.long).cpu()
    probability = torch.as_tensor(predictions["probability"]).detach().cpu().double()
    labels = torch.as_tensor(predictions["labels"]).detach().cpu()
    rows = torch.as_tensor(predictions["row_index"]).detach().cpu()
    if (probability.shape != (len(indices),) or labels.shape != (len(indices),)
            or not torch.equal(rows, indices)
            or not torch.equal(labels, cohort.tensors["binary"][indices].cpu())):
        raise ValueError("Prediction rows or labels do not match the fixed patient order")
    if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid recurrence probabilities")
    metrics = normalized_metrics(binary_metrics(labels.numpy(), probability.numpy()))
    metrics["positive"] = int(labels.sum())
    return metrics, probability.numpy()


@torch.inference_mode()
def predict_export(job, cohort, indices, device):
    path = Path(job["artifact"])
    if file_sha256(path) != job["artifact_sha256"]:
        raise ValueError("Selected inference export changed")
    if path.suffix == ".joblib":
        import joblib
        from stageworld_tcwm.research_baselines import predict_baseline
        artifact = joblib.load(path)
        probabilities = predict_baseline(artifact, cohort, indices)
        predictions = {"probability": probabilities, "labels": cohort.tensors["binary"][indices],
                       "row_index": indices}
    else:
        from stageworld_tcwm.four_stage_training import load_four_stage_export, evaluate_four_stage
        model, payload = load_four_stage_export(path, device=device)
        if payload.get("seed") != job["seed"] or payload.get("selected_step", payload.get("step")) != job["selected_step"]:
            raise ValueError("Export does not contain the declared seed and validation-selected step")
        model.eval()
        _, predictions = evaluate_four_stage(model, safe_evaluation_batch(cohort, indices, device))
    return validate_predictions(predictions, cohort, indices)


def require_test_ready(out, protocol, plan):
    if protocol["diagnostic"]:
        raise ValueError("Diagnostic runs must never score test")
    if protocol.get("test_policy") == "disabled" or protocol.get("train_validation_only", False):
        raise ValueError("Train/validation-only runs must never score test")
    require_seed_set(protocol["seeds"])
    if read(out / "evaluation_plan.json") != plan:
        raise ValueError("Test requires the frozen complete export plan")
    expected = {run_key(seed) for seed in SEEDS} | {"D01"}
    if set(plan["jobs"]) != expected:
        raise ValueError("Test requires all ten seeds without selection")
    reproduction = read(out / "validation_reproduction.json")
    if (reproduction.get("evaluation_plan_sha256") != file_sha256(out / "evaluation_plan.json")
            or set(reproduction.get("jobs", {})) != expected
            or not all(check.get("pass") is True for check in reproduction["jobs"].values())):
        raise ValueError("Every validation export must reproduce before any test prediction")


def reproduce_validation(out, protocol, plan, cohort, rows):
    checks = {}
    for key, job in plan["jobs"].items():
        actual, _ = predict_export(job, cohort, rows, protocol["device"])
        expected = normalized_metrics(read(out / "runs" / key / "metrics.json")["validation"])
        differences = {name: abs(actual[name] - expected[name]) for name in METRICS}
        if max(differences.values()) > 2e-6:
            raise ValueError(f"Validation-selected export did not reproduce: {key}")
        checks[key] = {"pass": True, "absolute_differences": differences}
    write({"evaluation_plan_sha256": file_sha256(out / "evaluation_plan.json"), "jobs": checks},
          out / "validation_reproduction.json")
    return checks


def write_csv(rows, path):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_metrics(rows):
    require_seed_set([row["seed"] for row in rows])
    summary = {"n_seeds": len(rows), "ensemble": False, "seed_selection": False,
               "dispersion": "sample standard deviation across fixed-split training seeds"}
    for role in ("train", "validation", "test"):
        summary[role] = {name: {"mean": statistics.mean(row[f"{role}_{name}"] for row in rows),
                               "sd": statistics.stdev(row[f"{role}_{name}"] for row in rows)}
                         for name in METRICS}
    return summary


def write_training_summary(out, protocol):
    """Report the fallback separately from the best actually trained epoch."""
    rows = []
    for seed in SEEDS:
        key = run_key(seed)
        result = completed_metrics(out, key, protocol)
        trained = min((entry for entry in read(out / "runs" / key / "history.json")
                       if entry["step"] > 0), key=lambda entry: entry["validation"]["nll"])
        row = {"seed": seed, "selected_epoch": result["selected_epoch"],
               "selected_kind": result["selected_kind"],
               "auxiliary_selected_epoch": result["auxiliary"]["selected_epoch"],
               "auxiliary_completed_epochs": result["auxiliary"]["completed_epochs"],
               "completed_epochs": result["completed_epochs"],
               "best_trained_epoch": trained["epoch"]}
        for role in ("train", "validation"):
            for name in METRICS:
                row[f"{role}_{name}"] = normalized_metrics(result[role])[name]
                row[f"best_trained_{role}_{name}"] = normalized_metrics(trained[role])[name]
        rows.append(row)
    write_csv(rows, out / "per_seed_train_validation.csv")
    write({"n_seeds": len(rows), "test_evaluated": False,
           "trained_dynamics_selected": sum(row["selected_epoch"] > 0 for row in rows),
           "best_trained_is_diagnostic_only": True,
           "mean": {key: statistics.mean(row[key] for row in rows)
                    for key in rows[0] if any(key.startswith(prefix) for prefix in
                                            ("train_", "validation_", "best_trained_train_", "best_trained_validation_"))}},
          out / "summary.json")


def score_test(out, protocol, plan, cohort, test_indices):
    require_test_ready(out, protocol, plan)
    verify_identity(protocol)
    rows, baseline, private_predictions = [], None, {}
    for key, job in plan["jobs"].items():
        metrics, probability = predict_export(job, cohort, test_indices, protocol["device"])
        training = completed_metrics(out, key, protocol)
        record = {"model": "D01" if key == "D01" else "weak_four_stage", "seed": job["seed"],
                  "selected_step": job["selected_step"], "selected_kind": job["selected_kind"],
                  "selected_epoch": training.get("selected_epoch"),
                  "completed_steps": training.get("completed_steps"),
                  "completed_epochs": training.get("completed_epochs"),
                  "stop_reason": training.get("stop_reason"),
                  "pretrain_steps": training.get("auxiliary", {}).get("completed_steps"),
                  "pretrain_selected_step": training.get("auxiliary", {}).get("selected_step"),
                  "pretrain_selected_epoch": training.get("auxiliary", {}).get("selected_epoch"),
                  "pretrain_completed_epochs": training.get("auxiliary", {}).get("completed_epochs"),
                  "pretrain_stop_reason": training.get("auxiliary", {}).get("stop_reason"),
                  "batch_size": training.get("hyperparameters", {}).get("batch_size"),
                  "steps_per_epoch": training.get("steps_per_epoch"),
                  "test_n": metrics["n"], "test_positive": metrics["positive"]}
        for role, values in (("train", normalized_metrics(training["train"])),
                             ("validation", normalized_metrics(training["validation"])), ("test", metrics)):
            for name in METRICS:
                record[f"{role}_{name}"] = values[name]
        if key == "D01":
            baseline = record
        else:
            rows.append(record)
        private_predictions[key] = probability
    write_csv(rows, out / "per_seed.csv")
    write_csv([baseline], out / "baseline.csv")
    summary = aggregate_seed_metrics(rows)
    summary.update({"baseline": baseline, "test_patients": len(test_indices),
                    "test_positive": rows[0]["test_positive"], "concept_validated": False,
                    "historical_test_exposure": True, "independent_external_validation": False})
    write(summary, out / "summary.json")
    np.savez_compressed(out / "test_predictions.private.npz", row_index=test_indices.cpu().numpy(),
                        patient_ids=np.asarray([cohort.ids[i] for i in test_indices.tolist()]),
                        labels=cohort.tensors["binary"][test_indices].cpu().numpy(), **private_predictions)
    return summary


def verify_completed_study(out, protocol):
    path = out / "completion.json"
    if not path.exists():
        return False
    verify_identity(protocol)
    receipt = read(path)
    if receipt["protocol_sha256"] != file_sha256(out / "study_protocol.json"):
        raise ValueError("Completed study protocol mismatch")
    for name, digest in receipt["artifacts_sha256"].items():
        if file_sha256(out / name) != digest:
            raise ValueError("Completed study artifact changed")
    for key in [run_key(seed) for seed in protocol["seeds"]] + ["D01"]:
        if completed_metrics(out, key, protocol) is None:
            raise ValueError("Missing training receipt in completed study")
    return True


def main(argv=None):
    args = parse_args(argv)
    os.umask(0o077)
    args.out = args.out.resolve()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with study_lock(args.out):
        protocol = bind_protocol(args)
        if verify_completed_study(args.out, protocol):
            print(json.dumps({"status": "reused_completed_study", "out": str(args.out)}), flush=True)
            return 0
        if (args.out / "status.json").exists():
            attempts = args.out / "coordinator_attempts"
            attempts.mkdir(exist_ok=True)
            shutil.copy2(args.out / "status.json", attempts / f"{time.time_ns()}.json")
        status = {"status": "running", "pid": os.getpid(), "started_at_unix": time.time(),
                  "completed_seeds": [], "stage": "training", "test_evaluated": False,
                  "diagnostic": args.diagnostic, "concept_validated": False,
                  "training_schedule": protocol["training_schedule"]}
        write(status, args.out / "status.json")
        try:
            verify_identity(protocol)
            from stageworld_tcwm import four_stage_training
            from stageworld_tcwm.research_baselines import fit_baseline
            if not Path(four_stage_training.__file__).resolve().is_relative_to(ROOT):
                raise ValueError("Training must execute original repository source")
            cohort = TimelineCohort.load(args.data_root / "cohort.pt")
            roles = split_indices(cohort, read(args.data_root / "split.json"))
            if {role: len(indices) for role, indices in roles.items()} != FIXED_COUNTS:
                raise ValueError("Invalid frozen patient split")
            for seed in protocol["seeds"]:
                key = run_key(seed)
                status["active_seed"] = seed
                write(status, args.out / "status.json")
                if completed_metrics(args.out, key, protocol) is None:
                    preserve_incomplete_run(args.out, key)
                    verify_identity(protocol)
                    four_stage_training.train_four_stage(
                        cohort, roles["train"], roles["validation"], args.out / "runs" / key,
                        seed=seed, device=args.device, diagnostic=args.diagnostic)
                    run_receipt(args.out, key, TRAIN_FILES, protocol)
                status["completed_seeds"].append(seed)
                write(status, args.out / "status.json")
                print(json.dumps({"event": "seed_complete", "seed": seed}), flush=True)
            if completed_metrics(args.out, "D01", protocol) is None:
                preserve_incomplete_run(args.out, "D01")
                fit_baseline("D01", cohort, roles["train"], roles["validation"],
                             args.out / "runs" / "D01", seed=17)
                run_receipt(args.out, "D01", ("metrics.json", "inference.joblib"), protocol)
            if args.diagnostic:
                status.update(stage="diagnostic_complete", status="pass", completed_at_unix=time.time())
                write(status, args.out / "status.json")
                artifacts = ("status.json",)
            else:
                plan = freeze_exports(args.out, protocol)
                status["stage"] = "validation_reproduction"
                write(status, args.out / "status.json")
                reproduce_validation(args.out, protocol, plan, cohort, roles["validation"])
                if args.train_validation_only:
                    write_training_summary(args.out, protocol)
                    status.update(stage="train_validation_complete", status="pass", test_evaluated=False,
                                  completed_at_unix=time.time())
                    artifacts = ("status.json", "evaluation_plan.json", "validation_reproduction.json",
                                 "per_seed_train_validation.csv", "summary.json")
                else:
                    status["stage"] = "test_evaluation"
                    write(status, args.out / "status.json")
                    score_test(args.out, protocol, plan, cohort, roles["test"])
                    status.update(stage="complete", status="pass", test_evaluated=True,
                                  completed_at_unix=time.time())
                    artifacts = ("status.json", "evaluation_plan.json", "validation_reproduction.json",
                                 "per_seed.csv", "baseline.csv", "summary.json", "test_predictions.private.npz")
                write(status, args.out / "status.json")
            verify_identity(protocol)
            write({"protocol_sha256": file_sha256(args.out / "study_protocol.json"),
                   "artifacts_sha256": {name: file_sha256(args.out / name) for name in artifacts},
                   "test_evaluated": status["test_evaluated"], "completed_at_unix": time.time()},
                  args.out / "completion.json")
            print(json.dumps({"status": "pass", "test_evaluated": status["test_evaluated"],
                              "out": str(args.out)}), flush=True)
            return 0
        except BaseException as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}", failed_at_unix=time.time())
            write(status, args.out / "status.json")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
