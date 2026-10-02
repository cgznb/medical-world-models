#!/usr/bin/env python
"""Prepare immutable modality folds from audited caches; never open raw workbooks."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from stageworld_tcwm.data import Cohort, file_sha256, fingerprint, write_json
from stageworld_tcwm.modality_schema import SCHEMA, SCHEMA_ENABLED
from stageworld_tcwm.modality_data import (LEGACY_MODE, aggregate_modality_audit,
                                          legacy_modality_rows, ordinal_event_tensors)
from stageworld_tcwm.timeline_data import TimelineCohort, split_indices


def convert_fold(old, old_split, raw, events, *, acknowledge_loss_of_information=False):
    """Use cached scalar methods and fold clinical32; discard every drug input."""
    outer = list(old.metadata.get("outer_evaluation_ids", []))
    excluded = set(old.metadata.get("excluded_ids", []))
    if not outer or not excluded or set(outer) & excluded:
        raise ValueError("Require the audited nested folds with explicitly excluded holdouts")
    split = {"train": list(old_split["train"]), "validation": list(old_split["validation"]),
             "outer_evaluation": outer}
    selected = set(patient for patients in split.values() for patient in patients)
    if selected & excluded or selected | excluded != set(old.ids):
        raise ValueError("Excluded original holdouts must be physically removed")
    if set(old_split["test"]) != set(outer) | excluded:
        raise ValueError("Old test-role membership does not match audited nested partitions")
    if old.encoders.get("fit_ids") != split["train"]:
        raise ValueError("Frozen clinical encoder fit IDs differ from the original fold")
    if events.get("source_pool_id") != raw["artifact_id"]:
        raise ValueError("Event source pool provenance mismatch")
    if (old.metadata.get("original_pool_id") != raw["artifact_id"] or
            old.metadata.get("original_event_id") != events["artifact_id"]):
        raise ValueError("Frozen fold source artifact identity mismatch")
    if len(set(raw["ids"])) != len(raw["ids"]) or set(raw["ids"]) != set(old.ids):
        raise ValueError("Frozen fold and original cache patients differ")
    event_ids = events["patient_ids"]
    if len(set(event_ids)) != len(event_ids) or set(event_ids) != set(raw["ids"]):
        raise ValueError("Audited legacy event patient binding mismatch")
    ids = [patient for patient in old.ids if patient in selected]
    old_lookup = {patient: index for index, patient in enumerate(old.ids)}
    event_lookup = {patient: index for index, patient in enumerate(event_ids)}
    rows = torch.tensor([old_lookup[patient] for patient in ids])
    raw_lookup = {patient: index for index, patient in enumerate(raw["ids"])}
    raw_rows = torch.tensor([raw_lookup[patient] for patient in ids])
    event_values = events["events"][[event_lookup[patient] for patient in ids]]
    for current, source in (("ct0", "ct0"), ("ct1", "ct1_tokens")):
        if not torch.equal(old.tensors[current][rows], raw[source][raw_rows]):
            raise ValueError("Frozen fold CT features differ from the verified original cache")
    for name, column in (("pcr", 0), ("binary", 1)):
        if (not torch.equal(old.tensors[name][rows], raw["labels"][raw_rows, column]) or
                not torch.equal(old.tensors[name + "_valid"][rows], raw["valid"][raw_rows, column])):
            raise ValueError("Frozen fold outcomes differ from the verified original cache")
    methods = legacy_modality_rows(raw["treatments"], ids,
                                   acknowledge_loss_of_information=acknowledge_loss_of_information)
    tensors = {name: old.tensors[name][rows].clone() for name in
               ("ct0", "ct1", "clinical", "image_valid", "binary", "binary_valid", "pcr", "pcr_valid")}
    tensors.update(ordinal_event_tensors(methods, event_values))
    metadata = {
        "time_basis": "ordinal_stage", "schema_enabled": list(SCHEMA_ENABLED),
        "source_mode": LEGACY_MODE, "raw_workbook_read": False,
        "compound_record_policy": "retrospective_replay_no_prospective_intermediate_claim",
        "query_names": ["S0", "S1", "S2_replay", "S3_retrospective_final"],
        "query_prospective_supported": [False, False, False, False],
        "legacy_final": "S3_retrospective_final_if_present",
        "endpoint_definition": "recorded_recurrence_metastasis_status_not_fixed_horizon",
        "ct1_policy": "supervision_only_pre_assimilation_after_neoadjuvant_summary",
        "calendar_training_supported": False, "original_holdouts_excluded": len(excluded),
        "source_pool_id": raw["artifact_id"], "source_event_id": events["artifact_id"],
        "clinical_fit_ids_sha256": fingerprint(sorted(split["train"])),
        "warnings": [
            "Cached modality values may already have been changed by named-treatment conflicts; unknowns cannot be recovered.",
            "This is not equivalent to reconstruction from original modality columns.",
            "AQ surgery and CE postoperative chemotherapy source columns are verified in the producer; their dates and availability are unknown.",
            "Intermediate states are retrospective ordinal replay, not prospectively verified clinical landmarks.",
            "Postoperative chemotherapy summaries neither establish treatment start/completion nor persistent active treatment.",
            "Radiotherapy is disabled; no missing source is recoded as confirmed absence.",
        ],
    }
    cohort = TimelineCohort({"schema": SCHEMA, "ids": ids, "tensors": tensors,
                             "metadata": metadata,
                             "encoders": {"clinical": old.encoders["clinical"], "fit_ids": split["train"]}})
    indices = split_indices(cohort, split)
    report = aggregate_modality_audit(methods, event_values)
    report["partitions"] = {
        role: {"patients": len(index), "recurrence_positive": int(tensors["binary"][index].sum())}
        for role, index in indices.items()}
    report.update({"source_clinical_features_reused_exactly": True,
                   "clinical_fit_ids_exact_match": True, "drug_support_imported": False,
                   "old_4x82_sliced": False, "excluded_original_holdouts": len(excluded)})
    return cohort, split, report


def verify_source_hashes(folds, pool, events):
    manifest = json.loads((Path(folds) / "preparation.json").read_text())
    for name, path in (("pool", pool), ("events", events)):
        if file_sha256(path) != manifest["sources"][name]["sha256"]:
            raise ValueError(f"The {name} cache differs from the frozen source manifest")
    for index in range(3):
        report = manifest["folds"][index]
        for filename, key in (("cohort.pt", "cohort_sha256"), ("split.json", "split_sha256")):
            if file_sha256(Path(folds) / f"fold-{index}" / filename) != report[key]:
                raise ValueError("Frozen fold source checksum mismatch")
    return manifest


def verify_existing(folds, pool, events, out):
    """Read-only revalidation of immutable output against stronger source checks."""
    folds, pool, events, out = map(Path, (folds, pool, events, out))
    verify_source_hashes(folds, pool, events)
    raw = torch.load(pool, map_location="cpu", weights_only=True)
    original_events = torch.load(events, map_location="cpu", weights_only=True)
    manifest = json.loads((out / "preparation.json").read_text())
    for index in range(3):
        source, destination = folds / f"fold-{index}", out / f"fold-{index}"
        old = Cohort.load(source / "cohort.pt")
        old_split = json.loads((source / "split.json").read_text())
        rebuilt, split, _ = convert_fold(old, old_split, raw, original_events,
                                         acknowledge_loss_of_information=True)
        actual = TimelineCohort.load(destination / "cohort.pt")
        if actual.ids != rebuilt.ids or json.loads((destination / "split.json").read_text()) != split:
            raise ValueError("Prepared patient membership differs from exact source conversion")
        split_indices(actual, split)
        for name, value in rebuilt.tensors.items():
            torch.testing.assert_close(actual.tensors[name], value, rtol=0, atol=0, equal_nan=True)
        for filename, key in (("cohort.pt", "cohort_sha256"), ("split.json", "split_sha256")):
            if file_sha256(destination / filename) != manifest["folds"][index][key]:
                raise ValueError("Prepared immutable artifact checksum mismatch")
    return {"status": "passed", "mode": "read_only_revalidation", "folds": 3,
            "source_checksums_verified": True, "features_labels_and_membership_exact": True}


def prepare(folds, pool, events, out, *, acknowledge_loss_of_information=False):
    if not acknowledge_loss_of_information:
        raise ValueError("Use --acknowledge-legacy-loss-of-information")
    folds, pool, events, out = map(Path, (folds, pool, events, out))
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; historical artifacts are immutable")
    os.umask(0o077)
    verify_source_hashes(folds, pool, events)
    raw = torch.load(pool, map_location="cpu", weights_only=True)
    source_events = torch.load(events, map_location="cpu", weights_only=True)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    sources = {"pool": {"path": str(pool.resolve()), "sha256": file_sha256(pool)},
               "events": {"path": str(events.resolve()), "sha256": file_sha256(events)}}
    reference_ids = None
    for index in range(3):
        folder = folds / f"fold-{index}"
        old = Cohort.load(folder / "cohort.pt")
        old_split = json.loads((folder / "split.json").read_text())
        cohort, split, report = convert_fold(old, old_split, raw, source_events,
                                            acknowledge_loss_of_information=True)
        if reference_ids is not None and set(cohort.ids) != reference_ids:
            raise ValueError("Prepared folds do not share the same 521-person development cohort")
        reference_ids = set(cohort.ids)
        cohort.metadata["sources"] = {**sources, "fold": {
            "path": str(folder.resolve()), "cohort_sha256": file_sha256(folder / "cohort.pt"),
            "split_sha256": file_sha256(folder / "split.json")}}
        destination = out / f"fold-{index}"
        destination.mkdir()
        cohort.save(destination / "cohort.pt")
        write_json(split, destination / "split.json")
        report.update({"fold": index, "cohort_sha256": file_sha256(destination / "cohort.pt"),
                       "split_sha256": file_sha256(destination / "split.json")})
        write_json(report, destination / "preparation.json")
        reports.append(report)
    result = {"schema": SCHEMA, "status": "passed", "source_mode": LEGACY_MODE,
              "raw_workbook_read": False, "sources": sources, "folds": reports}
    write_json(result, out / "preparation.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acknowledge-legacy-loss-of-information", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.verify_only:
        print(json.dumps(verify_existing(args.folds, args.pool, args.events, args.out), indent=2))
        return
    result = prepare(args.folds, args.pool, args.events, args.out,
                     acknowledge_loss_of_information=args.acknowledge_legacy_loss_of_information)
    print(json.dumps({"status": result["status"], "source_mode": result["source_mode"],
                      "fold_counts": [row["partitions"] for row in result["folds"]]}, indent=2))


if __name__ == "__main__":
    main()
