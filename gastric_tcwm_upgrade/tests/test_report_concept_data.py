"""Synthetic report evidence checks, without patient records or external calls."""
from copy import deepcopy
import math

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.data import fingerprint
from stageworld_tcwm.modality_schema import SCHEMA, SCHEMA_ENABLED
from stageworld_tcwm.report_concept_data import (S1_CONCEPT_NAMES, attach_report_concepts,
                                                extract_report_concepts, load_report_schema,
                                                validate_report_concepts)
from stageworld_tcwm.timeline_data import TimelineCohort


@pytest.fixture
def reports():
    validator, _ = load_report_schema()
    ids = ["synthetic-a", "synthetic-b", "synthetic-c"]
    split = {"train": [ids[0]], "validation": [ids[1]], "test": [ids[2]]}
    t = modality_batch(3)
    t["modality_value"] = t["modality_value"].bool()
    t["pcr_valid"][0] = False
    t["modality_value"][1, 0] = False
    cohort = TimelineCohort({"schema": SCHEMA, "ids": ids, "tensors": t,
                             "metadata": {"time_basis": "ordinal_stage", "schema_enabled": SCHEMA_ENABLED,
                                          "source_pool_id": "synthetic-source", "split_sha256": fingerprint(split)},
                             "encoders": {"fit_ids": split["train"]}})
    text = "Synthetic specimen: viable primary present, viable fraction 25%, positive nodes 3, diameter 2 cm."
    manifest = {"schema_version": validator.SCHEMA_VERSION, "source_pool_id": "synthetic-source",
                "patient_ids": ids, "role": "training_only_auxiliary_supervision", "artifact_id": "synthetic-manifest",
                "source_workbook": {"column": "AU"},
                "records": [{"patient_id": patient, "eligible": True, "report": text} for patient in ids]}
    progress = {"manifest_id": manifest["artifact_id"], "status": "completed"}
    records = {}
    for patient in ids:
        result = validator.empty_result("valid")
        for name, value, evidence in zip(S1_CONCEPT_NAMES, ("present", 25, 3, 2),
                                         ("viable primary present", "viable fraction 25%", "positive nodes 3", "diameter 2 cm")):
            result["findings"][name] = {"status": "observed", "value": value, "evidence": [evidence]}
        records[patient] = {"manifest_id": manifest["artifact_id"], "patient_id": patient,
                            "status": "completed", "result": result}
    records[ids[2]]["result"]["findings"]["primary_tumour_max_diameter_cm"] = {
        "status": "ambiguous", "value": None, "evidence": []}
    return cohort, split, manifest, progress, records, validator


def extract(fixture):
    cohort, _, manifest, progress, records, validator = fixture
    return extract_report_concepts(cohort, manifest, progress, records.__getitem__, validator)


def test_transforms_missing_masks_and_nac_applicability_without_pcr_label(reports):
    targets, observed, quality = extract(reports)
    torch.testing.assert_close(targets["s1_concepts"][0], torch.tensor([1., .25, math.log1p(3), math.log1p(2)]))
    assert targets["s1_concept_valid"].tolist() == [[True, True, True, True], [False, False, True, True], [True, True, True, False]]
    assert torch.isnan(targets["s1_concepts"][1, :2]).all()
    assert torch.isnan(targets["s1_concepts"][2, 3])
    assert observed.sum(0).tolist() == [3, 3, 3, 2]
    assert quality["applicable_counts"] == [2, 2, 3, 2]


def test_attach_preserves_original_tensors_encoder_and_split(reports):
    cohort, split, *_ = reports
    before = deepcopy(cohort.tensors)
    targets, observed, quality = extract(reports)
    augmented = attach_report_concepts(cohort, split, targets, observed, quality, {"synthetic": True})
    for name, value in before.items():
        torch.testing.assert_close(augmented.tensors[name], value, rtol=0, atol=0, equal_nan=True)
    assert augmented.ids == cohort.ids and augmented.encoders == cohort.encoders
    assert augmented.metadata["split_sha256"] == cohort.metadata["split_sha256"]
    contract = augmented.metadata["s1_report_concepts"]
    assert contract["weak_s1_report_pilot_ready"] and not contract["longitudinal_concept_readiness"]
    assert contract["target_stage"] == 1 and contract["available_stage"] == 2
    assert not contract["clinically_adjudicated"]
    assert "s1_concepts" not in cohort.tensors


@pytest.mark.parametrize("change", ["wrong_source", "wrong_order", "incomplete", "unsupported_evidence", "inferred_extra_target", "wrong_units_value"])
def test_invalid_sources_or_findings_cannot_enter_targets(reports, change):
    _, _, manifest, progress, records, _ = reports
    if change == "wrong_source":
        manifest["source_pool_id"] = "different"
    elif change == "wrong_order":
        manifest["patient_ids"] = list(reversed(manifest["patient_ids"]))
    elif change == "incomplete":
        progress["status"] = "in_progress"
    elif change == "unsupported_evidence":
        records["synthetic-a"]["result"]["findings"][S1_CONCEPT_NAMES[0]]["evidence"] = ["invented finding"]
    elif change == "inferred_extra_target":
        records["synthetic-a"]["result"]["findings"]["pcr"] = {"status": "observed", "value": 1, "evidence": []}
    else:
        records["synthetic-a"]["result"]["findings"][S1_CONCEPT_NAMES[1]]["value"] = 101
    with pytest.raises(ValueError):
        extract(reports)


def test_contract_rejects_relabeling_specimen_as_postoperative_burden(reports):
    cohort, split, *_ = reports
    targets, observed, quality = extract(reports)
    result = attach_report_concepts(cohort, split, targets, observed, quality, {})
    result.metadata["s1_report_concepts"]["target_stage"] = 2
    with pytest.raises(ValueError, match="semantics"):
        validate_report_concepts(result.tensors, result.metadata)


def test_conflict_and_missing_are_not_negative_labels(reports):
    records = reports[4]
    for index, status in enumerate(("missing", "ambiguous", "conflict")):
        records["synthetic-a"]["result"]["findings"][S1_CONCEPT_NAMES[index]] = {
            "status": status, "value": None, "evidence": []}
    targets, _, _ = extract(reports)
    assert not targets["s1_concept_valid"][0, :3].any()
    assert torch.isnan(targets["s1_concepts"][0, :3]).all()
