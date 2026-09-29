#!/usr/bin/env python
"""Locked next-round training on the existing nested development folds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stageworld_tcwm.config import load_config, config_dict
from stageworld_tcwm.data import file_sha256, fingerprint, write_json
from evaluate_ct_cv import load_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("G0", "G1", "G2", "G3"), default=["G0", "G1"])
    parser.add_argument("--fold-indices", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--mc-samples", type=int, choices=(32, 64), default=64)
    parser.add_argument("--max-supervised-steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    os.umask(0o077)
    if len(set(args.cases)) != len(args.cases) or len(set(args.fold_indices)) != len(args.fold_indices):
        raise ValueError("Cases/folds must be unique")
    folds, _ = load_folds(args.folds)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "study.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configs = {}
        for case in args.cases:
            model, train = load_config(ROOT / "configs/next" / f"{case}.json")
            train.samples_eval, train.device = args.mc_samples, args.device
            if args.max_supervised_steps is not None:
                train.max_supervised_steps = args.max_supervised_steps
                train.max_optimizer_steps = args.max_supervised_steps
            configs[case] = config_dict(model, train.validate())
        identity = {"schema": "tcwm-next-round-v1", "source_root": str(ROOT),
            "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p)
                for directory in (ROOT / "src", ROOT / "scripts") for p in sorted(directory.rglob("*.py"))},
            "fold_hashes": [fold["hashes"] for fold in folds], "configs": configs,
            "fold_indices": args.fold_indices, "python": sys.executable,
            "original_holdouts_scored": False, "development_only": True}
        protocol = args.out / "study_protocol.json"
        if protocol.exists() and json.loads(protocol.read_text()) != identity:
            raise ValueError("Source/data/config changed; use a new study directory")
        write_json(identity, protocol)
        records = []
        for case in args.cases:
            config_path = args.out / "configs" / f"{case}.json"
            write_json(configs[case], config_path)
            for index in args.fold_indices:
                if any(file_sha256(ROOT / name) != expected
                       for name, expected in identity["source_sha256"].items()):
                    raise ValueError("Source changed after study lock; stop before mixing implementations")
                data = args.folds / f"fold-{index}"
                run = args.out / case / f"fold-{index}"
                run.parent.mkdir(parents=True, exist_ok=True)
                run.mkdir(exist_ok=True)
                provenance = {"study_id": fingerprint(identity), "case": case, "fold": index,
                              "config_sha256": file_sha256(config_path)}
                provenance_path = run / "study_provenance.json"
                if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
                    raise ValueError("Run provenance differs from locked study")
                write_json(provenance, provenance_path)
                command = [sys.executable, "-u", str(ROOT / "scripts/run_tcwm.py"),
                    "--threads", str(args.threads), "train", "--data", str(data / "cohort.pt"),
                    "--split", str(data / "split.json"), "--config", str(config_path), "--out", str(run)]
                if (run / "contract.json").exists():
                    command.append("--resume")
                print(f"Starting {case} fold-{index} from {ROOT}", flush=True)
                with (run.parent / f"fold-{index}.log").open("a") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if any(file_sha256(ROOT / name) != expected
                       for name, expected in identity["source_sha256"].items()):
                    raise ValueError("Source changed during a training run; results require a fresh study")
                record = {"case": case, "fold": index, "exit_code": result.returncode}
                if result.returncode == 0:
                    record["training"] = json.loads((run / "training_report.json").read_text())
                records.append(record)
                write_json(records, args.out / "study_status.json")
                if result.returncode:
                    raise RuntimeError(f"Training failed; inspect {run.parent / f'fold-{index}.log'}")
                summary = record["training"]
                print(json.dumps({"case": case, "fold": index,
                    **{key: summary[key] for key in ("optimizer_steps", "supervised_steps",
                        "selected_supervised_steps", "selected_kind", "best_validation_nll", "stop_reason")}}), flush=True)


if __name__ == "__main__":
    main()
