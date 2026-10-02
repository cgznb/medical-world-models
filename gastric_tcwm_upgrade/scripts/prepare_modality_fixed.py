#!/usr/bin/env python
"""Prepare all 651 patients once with the fixed seed-17 456/65/130 split."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch

from stageworld_tcwm.data import file_sha256, fingerprint, write_json
from stageworld_tcwm.modality_data import (LEGACY_MODE, aggregate_modality_audit,
                                          legacy_modality_rows, ordinal_event_tensors)
from stageworld_tcwm.modality_schema import SCHEMA, SCHEMA_ENABLED
from stageworld_tcwm.timeline_data import TimelineCohort

PROTOCOL = "fixed651_712"
SEED = 17
COUNTS = {"train": 456, "validation": 65, "test": 130}
CLINICAL_FIELDS = ("sex", "age", "bmi", "ct_stage", "cn_stage", "cm_stage")


def validate_fixed_split(split, ids):
    if set(split) != set(COUNTS) or any(len(split[role]) != count for role, count in COUNTS.items()):
        raise ValueError("fixed651_712 requires exactly 456 train, 65 validation and 130 test patients")
    flat = [patient for role in COUNTS for patient in split[role]]
    if len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError("Fixed patient partitions must be disjoint and cover all 651 patients")
    return split


def make_fixed_split(ids, labels, valid):
    """Stable ID ordering makes fixed seed membership independent of cache row order."""
    if len(ids) != 651 or len(set(ids)) != 651 or not all(isinstance(p, str) and p for p in ids):
        raise ValueError("The fixed protocol requires all 651 unique source patients")
    if labels.shape != (651, 2) or valid.shape != (651, 2) or valid.dtype != torch.bool:
        raise ValueError("Expected pCR and recurrence labels with their validity masks")
    if not bool(valid.all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("The complete651 source must have both labels; no further filtering is allowed")
    lookup = {patient: index for index, patient in enumerate(ids)}
    ordered = sorted(ids)
    indices = torch.tensor([lookup[p] for p in ordered], dtype=torch.long)
    y = labels[indices].long().cpu().numpy()
    strata = y[:, 0] + 2 * y[:, 1]
    train, holdout = train_test_split(np.arange(651), train_size=456, test_size=195,
                                     stratify=strata, random_state=SEED)
    validation, test = train_test_split(holdout, train_size=65, test_size=130,
                                        stratify=strata[holdout], random_state=SEED)
    split = {role: sorted(ordered[index] for index in rows)
             for role, rows in (("train", train), ("validation", validation), ("test", test))}
    return validate_fixed_split(split, ids)


def load_clinical_api(source_root):
    source = Path(source_root).resolve()
    if (source / "src" / "stageworld").is_dir():
        source = source / "src"
    expected = source / "stageworld" / "data" / "baseline_clinical.py"
    if not expected.is_file():
        raise ValueError("Clinical source root must contain stageworld/data/baseline_clinical.py")
    sys.path.insert(0, str(source.parent))
    sys.path.insert(0, str(source))
    module = importlib.import_module("stageworld.data.baseline_clinical")
    if Path(module.__file__).resolve() != expected:
        raise ValueError("The loaded clinical encoder differs from the requested source tree")
    if tuple(module.CT6_FIELD_NAMES) != CLINICAL_FIELDS:
        raise ValueError("Only the audited six baseline clinical fields are permitted")
    return module


def convert_fixed(raw, events, clinical_api, *, split=None, acknowledge_loss_of_information=False):
    if not acknowledge_loss_of_information:
        raise ValueError("Explicitly acknowledge legacy-loss-of-information migration")
    ids = list(raw["ids"])
    expected_split = make_fixed_split(ids, raw["labels"], raw["valid"])
    if split is None:
        split = expected_split
    else:
        validate_fixed_split(split, ids)
        if any(set(split[role]) != set(expected_split[role]) for role in COUNTS):
            raise ValueError("Reused membership differs from the locked seed-17 joint-stratified split")
        split = {role: sorted(split[role]) for role in COUNTS}
    if (raw["ct0"].shape != (651, 27, 768) or raw["ct1_tokens"].shape != raw["ct0"].shape
            or raw["ct0_valid"].shape != (651,) or raw["ct1_valid"].shape != (651,)
            or raw["ct0_valid"].dtype != torch.bool or raw["ct1_valid"].dtype != torch.bool
            or not bool(raw["ct0_valid"].all()) or not bool(raw["ct1_valid"].all())):
        raise ValueError("Complete651 requires both original CT feature sets; no patient may be dropped")
    if not all(torch.isfinite(raw[name]).all() for name in ("ct0", "ct1_tokens")):
        raise ValueError("Nonfinite original CT features")
    if events.get("source_pool_id") != raw["artifact_id"]:
        raise ValueError("Event artifact does not belong to the source pool")
    event_ids = list(events["patient_ids"])
    if len(event_ids) != 651 or len(set(event_ids)) != 651 or set(event_ids) != set(ids):
        raise ValueError("Event patient binding differs from all 651 source patients")
    lookup = {patient: index for index, patient in enumerate(event_ids)}
    aligned_events = events["events"][[lookup[patient] for patient in ids]]
    if set(raw["clinical"]) != set(ids) or set(raw["treatments"]) != set(ids):
        raise ValueError("Cached clinical and modality rows must exactly cover the complete source")
    if tuple(clinical_api.CT6_FIELD_NAMES) != CLINICAL_FIELDS:
        raise ValueError("Clinical encoder does not implement the audited six fields")
    clinical_rows = {patient: {name: raw["clinical"][patient].get(name) for name in CLINICAL_FIELDS}
                     for patient in ids}
    training_rows = {patient: clinical_rows[patient] for patient in split["train"]}
    transform = clinical_api.fit_clinical_transform(
        training_rows, set(split["train"]), schema_version=clinical_api.CT6_CLINICAL_SCHEMA)
    if (transform.get("training_patients") != 456 or transform.get("fit_split") != "train"
            or transform.get("schema_version") != clinical_api.CT6_CLINICAL_SCHEMA):
        raise ValueError("Clinical transform was not fitted to exactly the new training partition")
    clinical = clinical_api.encode_baseline([clinical_rows[patient] for patient in ids], transform).ridge_features().float()
    methods = legacy_modality_rows(raw["treatments"], ids, acknowledge_loss_of_information=True)
    tensors = {
        "ct0": raw["ct0"].float().clone(), "ct1": raw["ct1_tokens"].float().clone(),
        "clinical": clinical,
        "image_valid": torch.stack((raw["ct0_valid"], raw["ct1_valid"]), 1),
        "binary": raw["labels"][:, 1].float().clone(), "binary_valid": raw["valid"][:, 1].clone(),
        "pcr": raw["labels"][:, 0].float().clone(), "pcr_valid": raw["valid"][:, 0].clone(),
        **ordinal_event_tensors(methods, aligned_events),
    }
    metadata = {
        "protocol": PROTOCOL, "split_protocol": PROTOCOL, "split_seed": SEED,
        "split_ratio": [7, 1, 2], "split_counts": dict(COUNTS),
        "historical_holdouts_repartitioned": True, "historical_holdouts_preserved": False,
        "independent_external_validation": False, "patients_excluded": 0,
        "time_basis": "ordinal_stage", "schema_enabled": list(SCHEMA_ENABLED),
        "source_mode": LEGACY_MODE, "raw_workbook_read": False,
        "compound_record_policy": "retrospective_replay_no_prospective_intermediate_claim",
        "query_names": ["S0", "S1", "S2_replay", "S3_retrospective_final"],
        "query_prospective_supported": [False, False, False, False],
        "legacy_final": "S3_retrospective_final_if_present",
        "endpoint_definition": "recorded_recurrence_metastasis_status_not_fixed_horizon",
        "ct1_policy": "supervision_only_pre_assimilation_after_neoadjuvant_summary",
        "calendar_training_supported": False,
        "source_pool_id": raw["artifact_id"], "source_event_id": events["artifact_id"],
        "clinical_fields": list(CLINICAL_FIELDS),
        "clinical_fit_ids_sha256": fingerprint(split["train"]),
        "split_sha256": fingerprint(split),
        "warnings": [
            "All 651 patients are repartitioned under the user-authorized fixed 7:1:2 protocol; historical holdouts are reused.",
            "This internal test partition is not a new independent or development-naive external cohort.",
            "Cached methods may have prior named-treatment conflict information loss; original fields cannot be recovered.",
            "Surgery and postoperative chemotherapy are retrospective summaries with unknown occurrence and availability dates.",
            "No prospective intermediate, ongoing active treatment, radiotherapy or calendar horizon is established.",
        ],
    }
    cohort = TimelineCohort({"schema": SCHEMA, "ids": ids, "tensors": tensors,
                             "metadata": metadata, "encoders": {"clinical": transform, "fit_ids": split["train"]}})
    row_lookup = {patient: index for index, patient in enumerate(ids)}
    report = aggregate_modality_audit(methods, aligned_events)
    partitions = {}
    for role, members in split.items():
        rows = torch.tensor([row_lookup[patient] for patient in members])
        labels = raw["labels"][rows].long()
        partitions[role] = {"patients": len(members), "pcr_positive": int(labels[:, 0].sum()),
                            "recurrence_positive": int(labels[:, 1].sum()),
                            "joint_pcr_plus_2recurrence_counts": torch.bincount(labels[:, 0] + 2*labels[:, 1], minlength=4).tolist()}
    report.update({"protocol": PROTOCOL, "split_seed": SEED, "split_ratio": [7, 1, 2],
                   "partitions": partitions, "patients_excluded": 0,
                   "historical_holdouts_repartitioned": True, "independent_external_validation": False,
                   "clinical_fields": list(CLINICAL_FIELDS), "clinical_fit_patients": 456,
                   "clinical_transform_sha256": fingerprint(transform),
                   "clinical_fit_ids_sha256": fingerprint(split["train"]),
                   "drug_support_imported": False, "old_4x82_sliced": False})
    return cohort, split, report


def source_contract(pool, events, clinical_api):
    clinical_path = Path(clinical_api.__file__).resolve()
    source = clinical_path.parents[2]
    return {"pool": {"path": str(Path(pool).resolve()), "sha256": file_sha256(pool)},
            "events": {"path": str(Path(events).resolve()), "sha256": file_sha256(events)},
            "clinical_source_root": str(source),
            "clinical_source_sha256": {str(path.relative_to(source)): file_sha256(path)
                                       for path in sorted(source.rglob("*.py"))}}


def prepare(pool, events, clinical_source_root, out, *, split_path=None,
            acknowledge_loss_of_information=False, verify_only=False):
    if not acknowledge_loss_of_information:
        raise ValueError("Use --acknowledge-legacy-loss-of-information")
    pool, events, out = map(Path, (pool, events, out))
    if not verify_only and out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; fixed partitions are immutable")
    clinical_api = load_clinical_api(clinical_source_root)
    sources = source_contract(pool, events, clinical_api)
    raw = torch.load(pool, map_location="cpu", weights_only=True)
    event_data = torch.load(events, map_location="cpu", weights_only=True)
    reuse_path = Path(split_path) if split_path else out / "split.json" if verify_only else None
    reused = json.loads(reuse_path.read_text()) if reuse_path else None
    cohort, split, report = convert_fixed(raw, event_data, clinical_api, split=reused,
                                          acknowledge_loss_of_information=True)
    cohort.metadata["sources"] = sources
    if sources != source_contract(pool, events, clinical_api):
        raise ValueError("Source artifacts changed during conversion")
    protocol = {"protocol": PROTOCOL, "schema": SCHEMA, "seed": SEED,
                "ratio": [7, 1, 2], "counts": dict(COUNTS),
                "stratification": "pcr_plus_2_recurrence",
                "algorithm": "sorted_ids_then_sklearn_stratified_456_195_then_65_130_seed17_both",
                "sklearn_version": sklearn.__version__, "membership_sha256": fingerprint(split),
                "selection_roles": ["train", "validation"], "locked_evaluation_role": "test",
                "historical_holdouts_repartitioned": True, "independent_external_validation": False}
    if verify_only:
        actual = TimelineCohort.load(out / "cohort.pt")
        saved_report = json.loads((out / "preparation.json").read_text())
        if actual.ids != cohort.ids or actual.metadata != cohort.metadata or actual.encoders != cohort.encoders:
            raise ValueError("Prepared cohort does not match the exact fixed-source conversion")
        for name, values in cohort.tensors.items():
            torch.testing.assert_close(actual.tensors[name], values, rtol=0, atol=0, equal_nan=True)
        if json.loads((out / "split_protocol.json").read_text()) != protocol:
            raise ValueError("Frozen split protocol differs")
        for filename, key in (("cohort.pt", "cohort_sha256"), ("split.json", "split_file_sha256"),
                              ("split_protocol.json", "protocol_file_sha256")):
            if file_sha256(out / filename) != saved_report[key]:
                raise ValueError("Prepared fixed artifact checksum mismatch")
        return {"status": "passed", "mode": "read_only_revalidation", "protocol": PROTOCOL,
                "patients": 651, "partitions": report["partitions"]}
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=True)
    cohort.save(out / "cohort.pt")
    write_json(split, out / "split.json")
    write_json(protocol, out / "split_protocol.json")
    report.update({"status": "passed", "schema": SCHEMA, "sources": sources,
                   "source_membership_order_preserved": True,
                   "cohort_sha256": file_sha256(out / "cohort.pt"),
                   "split_file_sha256": file_sha256(out / "split.json"),
                   "protocol_file_sha256": file_sha256(out / "split_protocol.json")})
    write_json(report, out / "preparation.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--clinical-source-root", "--legacy-source", dest="clinical_source_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--acknowledge-legacy-loss-of-information", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = prepare(args.pool, args.events, args.clinical_source_root, args.out,
                     split_path=args.split, acknowledge_loss_of_information=args.acknowledge_legacy_loss_of_information,
                     verify_only=args.verify_only)
    print(json.dumps({key: report[key] for key in ("status", "protocol", "patients", "partitions")}, indent=2))


if __name__ == "__main__":
    main()
