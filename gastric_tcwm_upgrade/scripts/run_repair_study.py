#!/usr/bin/env python
"""Run a fixed validation-only gastric repair study on one GPU, sequentially."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stageworld_tcwm.data import write_json

CASES = ("schedule", "compact", "aux_balance", "observation", "anchored",
         "anchored_seed29", "anchored_seed43")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.cases:
        record = {"case": name, "started_utc": datetime.now(timezone.utc).isoformat()}
        with (args.out / (name + ".log")).open("a") as log:
            command = [sys.executable, "-u", str(ROOT / "scripts/run_local.py"),
                       "--data-dir", str(args.data_dir), "--out", str(args.out / name),
                       "--config", str(ROOT / "configs" / ("repair_" + name + ".json"))]
            print("starting " + name, flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        records.append(record)
        write_json(records, args.out / "study_status.json")
        if result.returncode:
            raise RuntimeError("Repair study failed at " + name + "; see its log")
        stats = json.loads((args.out / name / "evaluation_validation.json").read_text())
        print(json.dumps({"case": name, "validation": stats}), flush=True)


if __name__ == "__main__":
    main()
