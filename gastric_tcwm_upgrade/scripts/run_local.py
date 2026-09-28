#!/usr/bin/env python
"""Run the local adaptation under a process lock with restartable training."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stageworld_tcwm.data import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = open(args.out / "runner.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = {str(p.relative_to(ROOT)): file_sha256(p)
               for directory in (ROOT / "src", ROOT / "scripts")
               for p in sorted(directory.rglob("*.py"))}
    identity = {"source_sha256": sources, "config_sha256": file_sha256(args.config),
                "python": sys.executable, "evaluate_test": args.evaluate_test}
    contract_path = args.out / "runner_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != identity:
        raise ValueError("Runner source/config/evaluation contract changed; use a new run directory")
    write_json(identity, contract_path)
    status = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "phase": "train"}
    status_path = args.out / "runner_status.json"
    write_json(status, status_path)
    base = [sys.executable, "-u", str(ROOT / "scripts/run_tcwm.py"), "--threads", "4"]
    data = ["--data", str(args.data_dir / "cohort.pt"), "--split", str(args.data_dir / "split.json")]
    try:
        command = base + ["train", *data, "--config", str(args.config), "--out", str(args.out)]
        if (args.out / "contract.json").exists():
            command.append("--resume")
        subprocess.run(command, check=True)
        for role in (["validation", "test"] if args.evaluate_test else ["validation"]):
            status["phase"] = "evaluate_" + role
            write_json(status, status_path)
            if not (args.out / f"evaluation_{role}.json").exists():
                subprocess.run(base + ["evaluate", *data, "--run", str(args.out),
                                       "--role", role, "--samples", "32", "--device", "cuda"], check=True)
        status.update(status="complete", phase="complete")
    except BaseException as exc:
        status.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        traceback.print_exc()
        raise
    finally:
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status, status_path)


if __name__ == "__main__":
    main()
