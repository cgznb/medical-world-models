#!/usr/bin/env python
"""Terminal supervision and S1 pathology pilots on the frozen 651-patient split."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stageworld_tcwm.data import file_sha256, write_json
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_training import (
    TimelineTrainConfig, source_manifest, train_timeline, verify_source,
)
from run_modality import DEFAULT_DATA_ROOT, FIXED_COUNTS, validate_fixed_data

ARMS = {"terminal_control": 0.0, "terminal_align001": 0.01, "terminal_align01": 0.1}
REPORT_ARMS = {"report_control": 0.0, "report_weight001": 0.01, "report_weight01": 0.1}
ALL_ARMS = {**ARMS, **REPORT_ARMS}
OPTIMIZATION_ARMS = {
    "compact_control": {"terminal_clinical_anchor": False, "s1_report_concepts": False},
    "anchored_compact": {"terminal_clinical_anchor": True, "s1_report_concepts": False},
    "anchored_report": {"terminal_clinical_anchor": True, "s1_report_concepts": True},
}
ALL_ARMS.update(OPTIMIZATION_ARMS)
REPORT_DATA_ROOT = Path(os.environ.get("GASTRIC_REPORT_DATA_ROOT", "private_data/report_fixed651"))
FROZEN_SPLIT_SHA256 = "2b6b17fe70291c94673e0fbf31ac482828a55b318acc7407e13f216151edad48"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("A", "B-report-pilot", "optimization-v1"), default="A")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ALL_ARMS))
    parser.add_argument("--parallel-jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-arm", choices=tuple(ALL_ARMS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    available = {"A": ARMS, "B-report-pilot": REPORT_ARMS,
                 "optimization-v1": OPTIMIZATION_ARMS}[args.suite]
    args.arms = args.arms or list(available)
    args.data_root = args.data_root or (DEFAULT_DATA_ROOT if args.suite == "A" else REPORT_DATA_ROOT)
    if len(set(args.arms)) != len(args.arms) or args.threads < 1:
        parser.error("Arms must be unique and threads must be positive")
    if not args.worker_arm and set(args.arms) - set(available):
        parser.error("Selected arms must belong to the requested suite")
    return args


def configurations(args):
    filename = {"A": "state_v1.json", "B-report-pilot": "report_concepts_v1.json",
                "optimization-v1": "optimization_v1.json"}[args.suite]
    base = json.loads((ROOT / "configs" / "terminal" / filename).read_text())
    result = {}
    for arm in args.arms:
        if args.suite == "optimization-v1":
            model = TimelineConfig.from_dict(dict(base["model"], **OPTIMIZATION_ARMS[arm]))
            weights = {"report_concept_weight": .01 if model.s1_report_concepts else 0.0}
        else:
            model = TimelineConfig.from_dict(base["model"])
            weights = {"alignment_weight": ARMS[arm]} if args.suite == "A" else {"report_concept_weight": REPORT_ARMS[arm]}
        training = TimelineTrainConfig(**dict(base["training"], **weights, device=args.device))
        training.validate()
        result[arm] = {"model": asdict(model), "training": asdict(training)}
    return result


def select_validation_arm(records):
    if not records:
        raise ValueError("No completed arms")
    scores = {arm: float(metrics["validation"]["selection_nll"]) for arm, metrics in records.items()}
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("All validation scores must be finite")
    return min(scores, key=lambda arm: (scores[arm], list(ALL_ARMS).index(arm)))


def evaluate_selected_test(out, identity, selection):
    """Evaluate the validation-selected export once, after binding its selection."""
    from stageworld_tcwm.timeline_data import TimelineCohort, split_indices
    from stageworld_tcwm.timeline_losses import FixedCTMoments
    from stageworld_tcwm.timeline_model import TimelineModel
    from stageworld_tcwm.timeline_training import (
        checkpoint_head_status, collect_predictions, set_head_training_status,
    )

    if identity["diagnostic"] or identity["suite"] != "optimization-v1":
        raise ValueError("Selected-only test scoring requires a completed optimization study")
    selection_path = out / "selection.json"
    if json.loads(selection_path.read_text()) != selection:
        raise ValueError("Test scoring requires the frozen validation selection")
    records = json.loads((out / "results.json").read_text())
    winner = select_validation_arm(records)
    if winner != selection["selected_arm"] or set(records) != set(identity["arms"]):
        raise ValueError("Test scoring requires all arms and the validation-selected winner")
    run = out / winner
    verify_source(ROOT, identity["source_sha256"])
    data = Path(identity["data_root"])
    for name, expected in identity["data_sha256"].items():
        if file_sha256(data / name) != expected:
            raise ValueError("Data changed after validation selection")
    exported = torch.load(run / "inference.pt", map_location="cpu", weights_only=True)
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if (exported["contract_id"] != best["contract_id"]
            or exported["selected_step"] != best["selected_step"]
            or set(exported["model_state"]) != set(best["model_state"])
            or any(not torch.equal(value, best["model_state"][key])
                   for key, value in exported["model_state"].items())):
        raise ValueError("Inference export does not match the validation-selected checkpoint")
    config = identity["configurations"][winner]
    if (exported["contract"]["source_sha256"] != identity["source_sha256"]
            or exported["contract"]["cohort_sha256"] != identity["data_sha256"]["cohort.pt"]
            or exported["contract"]["split_sha256"] != FROZEN_SPLIT_SHA256
            or json.loads(json.dumps(exported["model_config"])) != config["model"]
            or exported["contract"]["training"] != config["training"]
            or exported["selected_step"] != selection["selected_step"]
            or file_sha256(run / "inference.pt") != selection["selected_inference_sha256"]):
        raise ValueError("Selected export is not bound to this frozen study")
    if exported["contract"]["training"]["evaluate_test"]:
        raise ValueError("Optimization workers must not evaluate test before arm selection")
    binding = {"selected_arm": winner, "selected_step": exported["selected_step"],
               "contract_id": exported["contract_id"],
               "selection_sha256": file_sha256(selection_path),
               "inference_sha256": file_sha256(run / "inference.pt"),
               "used_for_selection": False, "historical_test_exposure": True}
    destination = out / "selected_test_metrics.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if any(existing.get(key) != value for key, value in binding.items()):
            raise ValueError("Existing test evaluation is bound to a different selection")
        if not (out / "selected_test_predictions.json").exists():
            raise ValueError("Existing test evaluation is missing its predictions")
        if existing.get("predictions_sha256") != file_sha256(out / "selected_test_predictions.json"):
            raise ValueError("Existing test predictions changed after evaluation")
        return existing
    model_config = TimelineConfig.from_dict(config["model"])
    expected_heads = checkpoint_head_status(model_config, TimelineTrainConfig(**config["training"]),
                                            exported["selected_step"])
    if (exported["optimizer_steps"] != exported["selected_step"]
            or exported.get("head_training_status") != expected_heads
            or exported["metadata"].get("head_training_status") != expected_heads
            or exported["metadata"].get("report_concept_head_trained") != expected_heads["report_concept_head_trained"]):
        raise ValueError("Export training status does not match its selected checkpoint")
    device = torch.device(config["training"]["device"])
    torch.set_num_threads(identity["threads"])
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = TimelineModel(model_config).to(device)
    model.load_state_dict(exported["model_state"])
    set_head_training_status(model, exported["head_training_status"])
    model.training_recurrence_probability = exported["metadata"]["training_recurrence_probability"]
    targets = FixedCTMoments(model_config.image_dim, anchor_dim=model_config.hidden,
                             report_concepts=model_config.s1_report_concepts).to(device)
    targets.load_state_dict(exported["target_statistics"])
    cohort = TimelineCohort.load(data / "cohort.pt")
    split = json.loads((data / "split.json").read_text())
    roles = split_indices(cohort, split)
    if {role: len(indices) for role, indices in roles.items()} != FIXED_COUNTS:
        raise ValueError("Selected test scoring requires the frozen 456/65/130 split")
    trained = exported["metadata"].get("report_concept_head_trained", False)
    rows, metrics = collect_predictions(model, cohort, roles["test"],
                                        config["training"]["batch_size"], device, targets,
                                        report_concept_head_trained=trained)
    if len(rows) != 130 or any(row["query_order"] != 3 for row in rows):
        raise ValueError("Selected test predictions must contain one terminal score per patient")
    write_json(rows, out / "selected_test_predictions.json")
    result = dict(binding, metrics=metrics, evaluated_at_unix=time.time(),
                  predictions_sha256=file_sha256(out / "selected_test_predictions.json"))
    write_json(result, destination)
    return result


def worker(args):
    identity = json.loads((args.out / "study_protocol.json").read_text())
    if args.worker_arm not in identity["configurations"]:
        raise ValueError("Worker is not part of the bound study")
    verify_source(ROOT, identity["source_sha256"])
    data = Path(identity["data_root"])
    for name, expected in identity["data_sha256"].items():
        if file_sha256(data / name) != expected:
            raise ValueError("Study data changed before worker launch")
    config = identity["configurations"][args.worker_arm]
    torch.set_num_threads(identity["threads"])
    run = args.out / args.worker_arm
    train_timeline(data / "cohort.pt", data / "split.json",
                   TimelineConfig.from_dict(config["model"]), TimelineTrainConfig(**config["training"]),
                   run, ROOT, identity["source_sha256"],
                   resume=args.resume and (run / "contract.json").exists(),
                   diagnostic=identity["diagnostic"])
    return 0


def main(argv=None):
    args = parse_args(argv)
    os.umask(0o077)
    if args.worker_arm:
        return worker(args)
    validate_fixed_data(args.data_root)
    if file_sha256(args.data_root / "split.json") != FROZEN_SPLIT_SHA256:
        raise ValueError("Stage A must reuse the exact frozen seed17 456/65/130 split")
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    with (args.out / "study.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "schema": "terminal-state-study-v1", "protocol": "fixed651_712", "suite": args.suite,
            "source_root": str(ROOT), "source_sha256": source_manifest(ROOT),
            "python": sys.executable, "data_root": str(args.data_root.resolve()),
            "data_sha256": {name: file_sha256(args.data_root / name) for name in ("cohort.pt", "split.json")},
            "split_counts": FIXED_COUNTS, "split_ratio": [7, 1, 2], "split_seed": 17,
            "configurations": configurations(args), "arms": args.arms,
            "threads": args.threads, "parallel_jobs": args.parallel_jobs, "diagnostic": args.diagnostic,
            "checkpoint_selection": "validation_terminal_BCE", "arm_selection": "validation_terminal_BCE",
            "test_used_for_selection": False, "historical_test_exposure": True,
            "independent_external_validation": False, "causal_effects_identified": False,
            "terminal_boundary": "last_retrospective_treatment_summary_not_fixed_followup",
        }
        if args.suite == "B-report-pilot":
            from stageworld_tcwm.timeline_data import TimelineCohort
            metadata = TimelineCohort.load(args.data_root / "cohort.pt").metadata.get("s1_report_concepts")
            if not metadata or metadata.get("clinically_adjudicated") is not False:
                raise ValueError("The B report pilot requires the audited weak S1 pathology sidecar")
            identity.update(report_concept_contract=metadata, longitudinal_semantics_validated=False,
                            pilot_scope="evidence_backed_automatically_extracted_S1_specimen_targets")
        if args.suite == "optimization-v1":
            identity.update(test_evaluation="once_after_validation_arm_selection",
                            clinical_anchor="training_only_C1_logistic_terminal_prognosis",
                            initialization_selection=True,
                            hypothesis="reduce_overfitting_and_preserve_available_clinical_signal",
                            comparison_limit="reused_small_validation_set_not_unbiased_generalization")
        identity = json.loads(json.dumps(identity))
        protocol = args.out / "study_protocol.json"
        if protocol.exists():
            if not args.resume or json.loads(protocol.read_text()) != identity:
                raise ValueError("Existing study requires --resume with identical source, data and configuration")
        elif any(path.name != "study.lock" for path in args.out.iterdir()):
            raise ValueError("Refusing to overwrite an unbound study directory")
        else:
            write_json(identity, protocol)
        status = {"status": "in_progress", "pid": os.getpid(), "started_at_unix": time.time(),
                  "total_jobs": len(args.arms), "completed_jobs": 0, "running_arms": []}
        write_json(status, args.out / "status.json")
        pending, running, records = list(args.arms), {}, {}
        try:
            while pending or running:
                while pending and len(running) < args.parallel_jobs:
                    arm = pending.pop(0)
                    command = [sys.executable, "-u", str(Path(__file__).resolve()),
                               "--suite", args.suite, "--out", str(args.out.resolve()), "--worker-arm", arm]
                    if args.resume:
                        command.append("--resume")
                    with (args.out / f"{arm}.log").open("ab", buffering=0) as log:
                        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT)
                    running[arm] = process
                    print(json.dumps({"event": "started", "arm": arm, "pid": process.pid}), flush=True)
                for arm, process in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"{arm} failed with exit {code}; inspect {arm}.log")
                    records[arm] = json.loads((args.out / arm / "metrics.json").read_text())
                    del running[arm]
                    print(json.dumps({"event": "completed", "arm": arm}), flush=True)
                    write_json(records, args.out / "results.json")
                status.update(completed_jobs=len(records), running_arms=list(running))
                write_json(status, args.out / "status.json")
                if running:
                    time.sleep(0.5)
            winner = select_validation_arm(records)
            selection = {"selected_arm": winner, "selection_rule": "minimum_validation_terminal_BCE",
                         "validation_scores": {arm: value["validation"]["selection_nll"] for arm, value in records.items()},
                         "test_used_for_selection": False, "diagnostic": args.diagnostic,
                         "inference_path": str((args.out / winner / "inference.pt").resolve())}
            if args.suite == "optimization-v1":
                selection.update(selected_step=records[winner]["selected_step"],
                                 selected_inference_sha256=file_sha256(args.out / winner / "inference.pt"))
            if (args.out / "selection.json").exists():
                if json.loads((args.out / "selection.json").read_text()) != selection:
                    raise ValueError("Refusing to replace an existing frozen arm selection")
            write_json(selection, args.out / "selection.json")
            verify_source(ROOT, identity["source_sha256"])
            if args.suite == "optimization-v1" and not args.diagnostic:
                evaluate_selected_test(args.out, identity, selection)
            status.update(status="pass", completed_at_unix=time.time(), selected_arm=winner)
            write_json(status, args.out / "status.json")
            print(json.dumps(status), flush=True)
            return 0
        except BaseException as error:
            for process in running.values():
                if process.poll() is None:
                    process.terminate()
            for process in running.values():
                process.wait()
            status.update(status="fail", error_type=type(error).__name__, error=str(error))
            write_json(status, args.out / "status.json")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
