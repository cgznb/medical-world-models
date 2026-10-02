#!/usr/bin/env python
"""Emit aggregate B/C readiness without modifying data, splits or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from stageworld_tcwm.data import file_sha256, write_json
from stageworld_tcwm.terminal_readiness import audit_terminal_readiness, require_training_ready
from stageworld_tcwm.timeline_data import TimelineCohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--original-pool", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, help="Optional verified measured-clinical-concepts-v1 sidecar")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-stage", choices=("B", "C"), help="Exit 2 after writing the report when this stage is blocked")
    args = parser.parse_args()
    sources = {"cohort": args.data_root / "cohort.pt", "split": args.data_root / "split.json",
               "original_pool": args.original_pool, "events": args.events}
    if args.concepts:
        sources["concepts"] = args.concepts
    if args.out.resolve() in {path.resolve() for path in sources.values()}:
        parser.error("Output must not overwrite any source artifact")
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    cohort = TimelineCohort.load(sources["cohort"])
    split = json.loads(sources["split"].read_text())
    original = torch.load(args.original_pool, map_location="cpu", weights_only=True)
    events = torch.load(args.events, map_location="cpu", weights_only=True)
    concepts = torch.load(args.concepts, map_location="cpu", weights_only=True) if args.concepts else None
    report = audit_terminal_readiness(cohort, split, original, events, concepts=concepts)
    if hashes != {name: file_sha256(path) for name, path in sources.items()}:
        raise ValueError("Readiness source artifacts changed during the audit")
    report["sources"] = {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                         for name, path in sources.items()}
    write_json(report, args.out)
    print(json.dumps({"report": str(args.out.resolve()), "B_ready": report["stages"]["B"]["training_ready"],
                      "C_ready": report["stages"]["C"]["training_ready"], "patients": len(cohort.ids)}, sort_keys=True))
    if args.require_stage:
        try:
            require_training_ready(report, args.require_stage)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
