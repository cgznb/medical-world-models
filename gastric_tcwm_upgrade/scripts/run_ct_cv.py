#!/usr/bin/env python
"""Run a fixed nested development study within the original training patients."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stageworld_tcwm.data import file_sha256, write_json

CASES = ("token_control", "token_prior", "direct8", "direct32", "predictive8", "predictive32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "study.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        for name in args.cases:
            for fold in range(3):
                path = args.folds / f"fold-{fold}"
                run = args.out / name / f"fold-{fold}"
                run.parent.mkdir(parents=True, exist_ok=True)
                record = {"case": name, "fold": fold,
                          "started_utc": datetime.now(timezone.utc).isoformat()}
                print(f"starting {name} fold {fold}", flush=True)
                command = [sys.executable, "-u", str(ROOT / "scripts/run_local.py"),
                           "--data-dir", str(path), "--out", str(run), "--config",
                           str(ROOT / "configs" / f"ct_cv_{name}.json")]
                with (run.parent / f"fold-{fold}.log").open("a") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                record.update(exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
                records.append(record)
                write_json(records, args.out / "study_status.json")
                if result.returncode:
                    raise RuntimeError(f"Study failed: {name} fold {fold}; inspect the case log")
                report = json.loads((run / "training_report.json").read_text())
                print(json.dumps({"case": name, "fold": fold, "training": report}), flush=True)


if __name__ == "__main__":
    main()
