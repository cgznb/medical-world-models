#!/usr/bin/env python
"""Adapt the existing patient split and raw Generated651 cache without resplitting."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from stageworld_tcwm.data import Cohort, file_sha256, split_indices, write_json
from stageworld_tcwm.legacy import convert_legacy
from stageworld_tcwm.support import fit_support, support_flags


def normalize_split(document, pool, events=None):
    split = document.get("patient_ids", document)
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("The existing split must contain train/validation/test")
    members = [patient for role in split.values() for patient in role]
    if len(members) != len(set(members)) or set(members) != set(pool["ids"]):
        raise ValueError("Patient split overlaps or differs from the source cohort")
    identities = {pool["artifact_id"]}
    if events is not None:
        if events.get("source_pool_id") != pool["artifact_id"]:
            raise ValueError("The event cache belongs to a different source pool")
        identities.add(events["artifact_id"])
    if document.get("pool_id", pool["artifact_id"]) not in identities:
        raise ValueError("The existing split belongs to a different source pool")
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--legacy-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(4)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory; prepared cohorts are immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.legacy_source.resolve().parent))
    sys.path.insert(0, str(args.legacy_source.resolve()))
    raw = torch.load(args.pool, map_location="cpu", weights_only=True)
    events = torch.load(args.events, map_location="cpu", weights_only=True)
    document = json.loads(args.split.read_text())
    split = normalize_split(document, raw, events)
    split_path = args.out / "split.json"
    write_json(split, split_path)
    cohort_path = args.out / "cohort.pt"
    result = convert_legacy(args.pool, args.events, split, cohort_path,
                            acknowledge_retrospective=True)
    cohort = Cohort.load(cohort_path)
    rows = split_indices(cohort, split)
    for new, old in (("ct0", "ct0"), ("ct1", "ct1_tokens"),
                     ("interval_days", "interval")):
        torch.testing.assert_close(cohort.tensors[new], raw[old], rtol=0, atol=0)
    for name, column in (("pcr", 0), ("binary", 1)):
        torch.testing.assert_close(cohort.tensors[name], raw["labels"][:, column], rtol=0, atol=0)
        assert torch.equal(cohort.tensors[name + "_valid"], raw["valid"][:, column])
    assert set(cohort.encoders["fit_ids"]) == set(split["train"])
    sources = {"pool": args.pool, "events": args.events, "existing_split": args.split}
    source_files = {
        str(path.relative_to(args.legacy_source)): file_sha256(path)
        for path in sorted((args.legacy_source / "stageworld").rglob("*.py"))
    }
    release_helper = args.legacy_source.parent / "research_release.py"
    if release_helper.exists():
        source_files["../research_release.py"] = file_sha256(release_helper)
    provenance = {
        "inputs": {key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                   for key, path in sources.items()},
        "legacy_source": str(args.legacy_source.resolve()),
        "legacy_source_sha256": source_files,
        "split_seed": document.get("seed"),
        "split_algorithm": document.get("algorithm"),
        "same_patient_membership_as_existing_split": True,
        "raw_features_labels_and_masks_unchanged": True,
        "encoders_fit_on_train_only": True,
    }
    cohort.metadata["audited_upstream_commit"] = cohort.metadata.pop("source_commit")
    cohort.metadata["local_adaptation"] = provenance
    cohort.save(cohort_path)
    support = fit_support(cohort.batch(rows["train"]))
    partitions = {}
    for role, indices in rows.items():
        batch = cohort.batch(indices)
        flags = support_flags(support, batch)
        partitions[role] = {
            "patients": len(indices),
            "ct0_present": int(batch["image_valid"][:, 0].sum()),
            "ct1_present": int(batch["image_valid"][:, 1].sum()),
            "recurrence_positive": int(batch["binary"][batch["binary_valid"]].sum()),
            "recurrence_missing": int((~batch["binary_valid"]).sum()),
            "pcr_positive": int(batch["pcr"][batch["pcr_valid"]].sum()),
            "pcr_missing": int((~batch["pcr_valid"]).sum()),
            "interval_min_days": float(batch["interval_days"].min()),
            "interval_max_days": float(batch["interval_days"].max()),
            "interval_over_365_days": int((batch["interval_days"] > 365).sum()),
            "unseen_treatment_names_patients": int((batch["unseen_treatment_names_count"] > 0).sum()),
            "support_warnings": {name: sum(name in item["warnings"] for item in flags)
                                 for name in ("unseen_treatment_surgery_combination",
                                              "sparse_combination_support",
                                              "interval_outside_training_range")},
        }
    report = {**result, "status": "passed", "partitions": partitions,
              "cohort_sha256": file_sha256(cohort_path), "provenance": provenance,
              "limitations": [
                  "Retrospective treatment and CT interval are explicit scenarios, not baseline-known facts.",
                  "Recorded recurrence is binary status, not a fixed-horizon recurrence risk.",
                  "No verified survival sidecar or postoperative image observations are available in this cache.",
                  "Without new postoperative observations, S1 and S2 use the same information.",
                  "The existing cohort has been used for model development; held-out scoring is internal validation.",
                  "Long CT intervals are preserved and reported, never reinterpreted as follow-up time.",
              ]}
    write_json(report, args.out / "preparation.json")
    print(json.dumps({key: report[key] for key in ("status", "patients", "partitions")}, indent=2))


if __name__ == "__main__":
    main()
