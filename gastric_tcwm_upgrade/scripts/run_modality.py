#!/usr/bin/env python
"""Locked modality-only training, defaulting to the fixed 651-patient 7:1:2 split."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stageworld_tcwm.data import file_sha256, fingerprint, write_json
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_data import TimelineCohort, split_indices
from stageworld_tcwm.timeline_training import (
    TimelineTrainConfig, source_manifest, train_timeline, verify_source,
)

ARMS = ("dynamic_h128", "static_h128", "dynamic_h64")
FIXED_PROTOCOL = "fixed651_712"
LEGACY_PROTOCOL = "legacy-nested-521"
DEFAULT_DATA_ROOT = Path(os.environ.get("GASTRIC_DATA_ROOT", "private_data/modality_fixed651"))
FIXED_COUNTS = {"train": 456, "validation": 65, "test": 130}


def validate_fixed_data(data_root):
    cohort = TimelineCohort.load(data_root / "cohort.pt")
    split = json.loads((data_root / "split.json").read_text())
    roles = split_indices(cohort, split)
    counts = {role: len(indices) for role, indices in roles.items()}
    if len(cohort.ids) != 651 or counts != FIXED_COUNTS:
        raise ValueError("The fixed 651-patient protocol requires train456/validation65/test130")
    if (cohort.metadata.get("protocol") != FIXED_PROTOCOL or
            cohort.metadata.get("split_seed") != 17 or
            cohort.metadata.get("split_ratio") != [7, 1, 2]):
        raise ValueError("Use the audited fixed651_712 preparation with seed17 and ratio7:1:2")
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=(FIXED_PROTOCOL, LEGACY_PROTOCOL), default=FIXED_PROTOCOL)
    parser.add_argument("--data-root", type=Path, help=f"Default: {DEFAULT_DATA_ROOT}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--folds", nargs="+", type=int, choices=(0, 1, 2),
                        help="Historical replay only; requires --protocol legacy-nested-521")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--diagnostic", action="store_true", help="Two real optimizer updates; no test/outer evaluation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.protocol == FIXED_PROTOCOL and args.folds is not None:
        parser.error("The default fixed651_712 protocol has no folds")
    if args.protocol == LEGACY_PROTOCOL and args.data_root is None:
        parser.error("Historical replay requires an explicit --data-root")
    args.data_root = args.data_root or DEFAULT_DATA_ROOT
    args.arms = args.arms or ([ARMS[0]] if args.diagnostic or args.protocol == FIXED_PROTOCOL else list(ARMS))
    if args.protocol == LEGACY_PROTOCOL:
        args.folds = args.folds or ([0] if args.diagnostic else [0, 1, 2])
    else:
        args.folds = []
    if len(set(args.arms)) != len(args.arms) or len(set(args.folds)) != len(args.folds):
        parser.error("Arm and fold lists must contain unique entries")
    if args.diagnostic and (len(args.arms) != 1 or len(args.folds) > 1):
        parser.error("A diagnostic is restricted to one arm and one partition")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.umask(0o077)
    torch.set_num_threads(args.threads)
    arms, folds = args.arms, args.folds
    fixed = args.protocol == FIXED_PROTOCOL
    counts = validate_fixed_data(args.data_root) if fixed else None
    partitions = [(None, args.data_root)] if fixed else [
        (fold, args.data_root / f"fold-{fold}") for fold in folds]
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    with (args.out / "study.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configurations = {}
        for arm in arms:
            config = json.loads((ROOT / "configs" / "modality" / f"{arm}.json").read_text())
            model = TimelineConfig.from_dict(config["model"])
            training = TimelineTrainConfig(**config["training"])
            training.device = args.device
            training.validate()
            configurations[arm] = {"model": asdict(model), "training": asdict(training)}
        source = source_manifest(ROOT)
        identity = {"schema": "modality-timeline-study-v2", "source_root": str(ROOT),
                    "python": sys.executable, "source_sha256": source, "configurations": configurations,
                    "arms": arms, "folds": folds, "diagnostic": args.diagnostic, "threads": args.threads,
                    "fold_hashes": {str(fold): {name: file_sha256(args.data_root / f"fold-{fold}" / name)
                                                for name in ("cohort.pt", "split.json")} for fold in folds},
                    "original_holdout65_accessed": False, "ct1_assimilation": False,
                    "blocked_arms": {"C": "No verified calendar event dates",
                                     "D": "Calendar arm unavailable and CT1 assimilation disabled"}}
        if fixed:
            del identity["fold_hashes"]
            del identity["folds"]
            del identity["original_holdout65_accessed"]
            identity.update(protocol=FIXED_PROTOCOL, cohort_size=651, split_counts=counts,
                            split_ratio=[7, 1, 2], split_seed=17,
                            data_root=str(args.data_root.resolve()),
                            data_hashes={name: file_sha256(args.data_root / name)
                                         for name in ("cohort.pt", "split.json")},
                            historical_holdouts_repartitioned=True,
                            independent_external_validation=False,
                            checkpoint_selection_role="validation", final_evaluation_role="test")
        identity = json.loads(json.dumps(identity))
        protocol = args.out / "study_protocol.json"
        if protocol.exists():
            if json.loads(protocol.read_text()) != identity or not args.resume:
                raise ValueError("Existing study needs --resume with identical source/data/configuration")
        elif any(path.name != "study.lock" for path in args.out.iterdir()):
            raise ValueError("Refusing to overwrite an unbound study directory")
        else:
            write_json(identity, protocol)
        study_id = fingerprint(identity)
        records = []
        started = time.monotonic()
        status = {"status": "in_progress", "pid": os.getpid(), "study_id": study_id,
                  "total_jobs": len(arms) * len(partitions), "completed_jobs": 0,
                  "diagnostic": args.diagnostic, "started_at_unix": time.time()}
        if fixed:
            status.update(protocol=FIXED_PROTOCOL, split_counts=counts)
        write_json(status, args.out / "status.json")
        try:
            for arm in arms:
                config = configurations[arm]
                for fold, data in partitions:
                    verify_source(ROOT, source)
                    run = args.out / arm if fixed else args.out / arm / f"fold-{fold}"
                    status.update(current_arm=arm, elapsed_seconds=time.monotonic() - started)
                    if not fixed:
                        status["current_fold"] = fold
                    write_json(status, args.out / "status.json")
                    print(json.dumps({"event": "starting", "arm": arm, "protocol": args.protocol,
                                      "fold": fold, "source_root": str(ROOT), "output": str(run)}), flush=True)
                    metrics = train_timeline(data / "cohort.pt", data / "split.json",
                                             TimelineConfig.from_dict(config["model"]),
                                             TimelineTrainConfig(**config["training"]), run,
                                             ROOT, source, resume=args.resume and (run / "contract.json").exists(),
                                             diagnostic=args.diagnostic)
                    record = {"arm": arm, "metrics": metrics}
                    record.update({"protocol": FIXED_PROTOCOL} if fixed else {"fold": fold})
                    records.append(record)
                    status["completed_jobs"] = len(records)
                    write_json(records, args.out / "results.json")
                    write_json(status, args.out / "status.json")
            status.update(status="pass", elapsed_seconds=time.monotonic() - started,
                          completed_at_unix=time.time())
            write_json(status, args.out / "status.json")
            print(json.dumps(status), flush=True)
            return 0
        except BaseException as error:
            status.update(status="fail", error_type=type(error).__name__, error=str(error), failed_at_unix=time.time())
            write_json(status, args.out / "status.json")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
