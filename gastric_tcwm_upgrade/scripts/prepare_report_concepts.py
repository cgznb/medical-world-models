#!/usr/bin/env python
"""Prepare immutable S1 report targets while preserving the fixed651 cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from stageworld_tcwm.data import file_sha256, fingerprint, write_json
from stageworld_tcwm.report_concept_data import (attach_report_concepts, extract_report_concepts,
                                               load_report_schema)
from stageworld_tcwm.timeline_data import TimelineCohort, split_indices

FROZEN_SPLIT_FILE_SHA256 = "2b6b17fe70291c94673e0fbf31ac482828a55b318acc7407e13f216151edad48"


def prepare(data_root, report_root, original_pool, out):
    data_root, report_root, original_pool, out = map(Path, (data_root, report_root, original_pool, out))
    if out.exists():
        raise ValueError("Report-target output is immutable; choose a new nonexistent directory")
    source_files = {"cohort": data_root / "cohort.pt", "split": data_root / "split.json",
                    "split_protocol": data_root / "split_protocol.json", "preparation": data_root / "preparation.json",
                    "original_pool": original_pool, "report_manifest": report_root / "manifest.json",
                    "report_progress": report_root / "progress.json"}
    hashes = {name: file_sha256(path) for name, path in source_files.items()}
    if hashes["split"] != FROZEN_SPLIT_FILE_SHA256:
        raise ValueError("Require the exact existing seed17 456/65/130 split file")
    cohort = TimelineCohort.load(source_files["cohort"])
    split = json.loads(source_files["split"].read_text())
    original = torch.load(original_pool, map_location="cpu", weights_only=True)
    preparation = json.loads(source_files["preparation"].read_text())
    if (cohort.ids != original["ids"] or cohort.metadata.get("source_pool_id") != original["artifact_id"]
            or len(cohort) != 651 or {role: len(ids) for role, ids in split.items()}
            != {"train": 456, "validation": 65, "test": 130}
            or preparation.get("cohort_sha256") != hashes["cohort"]
            or preparation.get("split_file_sha256") != hashes["split"]):
        raise ValueError("Prepared source membership, artifact binding or fixed split differs")
    split_indices(cohort, split)
    manifest = json.loads(source_files["report_manifest"].read_text())
    progress = json.loads(source_files["report_progress"].read_text())
    validator, validator_provenance = load_report_schema()
    record_hashes = []

    def load_record(patient):
        path = report_root / "records" / f"{patient}.json"
        record_hashes.append(file_sha256(path))
        return json.loads(path.read_text())

    targets, observed, quality = extract_report_concepts(cohort, manifest, progress, load_record, validator)
    provenance = {"source_report_root": str(report_root.resolve()), "source_column": "AU",
                  "report_schema": validator.SCHEMA_VERSION, "validator": validator_provenance,
                  "records_sha256": fingerprint(record_hashes), "records": len(record_hashes),
                  "automatic_extraction": True, "extraction_identity_sha256": fingerprint(manifest.get("api", {})),
                  "raw_workbook_read": False, "report_text_or_evidence_exported": False,
                  "source_files": {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                                   for name, path in source_files.items()}}
    augmented = attach_report_concepts(cohort, split, targets, observed, quality, provenance)
    repeated_hashes = [file_sha256(report_root / "records" / f"{patient}.json") for patient in cohort.ids]
    if (hashes != {name: file_sha256(path) for name, path in source_files.items()}
            or repeated_hashes != record_hashes
            or file_sha256(validator_provenance["path"]) != validator_provenance["sha256"]):
        raise ValueError("Source changed during report preparation")
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700)
    augmented.save(out / "cohort.pt")
    for name in ("split.json", "split_protocol.json"):
        shutil.copyfile(data_root / name, out / name)
    report = {**preparation, "cohort_sha256": file_sha256(out / "cohort.pt"),
              "parent_preparation_sha256": hashes["preparation"],
              "s1_report_concepts": augmented.metadata["s1_report_concepts"],
              "original_tensors_unchanged": True, "clinical_encoder_unchanged": True,
              "weak_s1_report_pilot_ready": augmented.metadata["s1_report_concepts"]["weak_s1_report_pilot_ready"],
              "longitudinal_concept_readiness": False, "causal_strategy_readiness": False}
    write_json(report, out / "preparation.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--original-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.data_root, args.reports, args.original_pool, args.out)
    print(json.dumps({"status": "passed", "output": str(args.out.resolve()),
                      "weak_s1_report_pilot_ready": report["weak_s1_report_pilot_ready"],
                      "coverage": report["s1_report_concepts"]["coverage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
