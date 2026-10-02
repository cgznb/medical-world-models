import copy
import sys
from pathlib import Path

import pytest
import torch

from stageworld_tcwm.modality_schema import (MODALITY_COLUMNS, MODALITY_ID, SCHEMA,
                                           parse_modality_flag, status_ids)
from stageworld_tcwm.modality_data import (legacy_modality_rows, ordinal_event_tensors,
                                          read_modality_rows)
from stageworld_tcwm.timeline_data import TimelineCohort, split_indices


@pytest.mark.parametrize("negative", [0, 2])
def test_column_specific_negative_and_missing(negative):
    assert parse_modality_flag(1, "n", negative) == 1
    assert parse_modality_flag(negative, "n", negative) == 0
    for value, kind in ((None, "n"), (True, "b"), ("=1", "f"), ("#VALUE!", "e"),
                        (float("nan"), "n"), ("unexpected", "s"), (2 if negative == 0 else 0, "n")):
        assert parse_modality_flag(value, kind, negative) is None


def _methods():
    return {name: 1 if name in ("chemotherapy", "hipec") else 0
            for name, _, _, _ in MODALITY_COLUMNS}


def test_cache_mode_requires_ack_and_ignores_names():
    cached = {"P1": {"methods": _methods(), "drugs": {"x": 1}, "regimens": {"x": 1}}}
    with pytest.raises(ValueError, match="acknowledge"):
        legacy_modality_rows(cached, ["P1"])
    original = legacy_modality_rows(cached, ["P1"], acknowledge_loss_of_information=True)
    cached["P1"]["drugs"] = object()
    cached["P1"]["regimens"] = object()
    assert original == legacy_modality_rows(cached, ["P1"], acknowledge_loss_of_information=True)


def test_raw_whitelist_drug_string_independence(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "synthetic.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet["A2"] = "\u5e8f\u5217\u53f7"
    sheet["A3"] = 1
    sheet["AD3"] = "arbitrary drug regimen text"
    for name, column, negative, header in MODALITY_COLUMNS:
        sheet[f"{column}2"] = header
        sheet[f"{column}3"] = 1 if name == "chemotherapy" else negative
    book.save(path)
    class Pseudonymizer:
        def token(self, namespace, key, prefix):
            return prefix + key
    before = read_modality_rows(path, {"P1"}, Pseudonymizer())
    sheet["AD3"] = "=unrelated_formula()"
    book.save(path)
    after = read_modality_rows(path, {"P1"}, Pseudonymizer())
    assert before == after
    assert after[0]["methods"]["chemotherapy"] == 1
    assert after[0]["methods"]["immunotherapy"] == 0
    sheet["AF3"] = None
    book.save(path)
    assert read_modality_rows(path, {"P1"}, Pseudonymizer())[0]["methods"]["immunotherapy"] is None


def test_radiotherapy_disabled_and_unknown_not_absent():
    value = torch.zeros(1, 7, dtype=torch.bool)
    known = value.clone()
    applicable = value.clone()
    applicable[:, 0] = True
    assert status_ids(value, known, applicable)[0, 0] == 0
    known[:, 0] = True
    assert status_ids(value, known, applicable)[0, 0] == 1
    value[:, 0] = True
    assert status_ids(value, known, applicable)[0, 0] == 2
    applicable[:, 1] = True
    with pytest.raises(ValueError, match="Radiotherapy"):
        status_ids(value, known, applicable)


def _payload(events=None):
    rows = [{"patient_id": "P1", "methods": _methods()}]
    events = torch.tensor([[1, 1, 1]]) if events is None else events
    t = ordinal_event_tensors(rows, events)
    t.update(ct0=torch.zeros(1, 27, 8), ct1=torch.ones(1, 27, 8), clinical=torch.zeros(1, 32),
             image_valid=torch.ones(1, 2, dtype=torch.bool), binary=torch.ones(1), pcr=torch.zeros(1),
             binary_valid=torch.ones(1, dtype=torch.bool), pcr_valid=torch.ones(1, dtype=torch.bool))
    return {"schema": SCHEMA, "ids": ["P1"], "tensors": t,
            "metadata": {"time_basis": "ordinal_stage"}, "encoders": {}}


def test_surgery_postop_separate_scopes_and_unknown_record_no_s3():
    t = TimelineCohort(_payload()).tensors
    assert torch.equal(t["modality_applicable"][0, 1], torch.tensor([0, 0, 0, 0, 0, 0, 1]).bool())
    assert torch.equal(t["modality_applicable"][0, 2], torch.tensor([1, 0, 0, 0, 0, 0, 0]).bool())
    assert t["phase"][0, 0] != t["phase"][0, 2]
    assert t["operation"][0, 2] == 3
    unknown = TimelineCohort(_payload(torch.tensor([[1, 1, 2]]))).tensors
    assert not unknown["event_mask"][0, 2]
    assert not unknown["query_mask"][0, 3]


def test_unknown_calendar_dates_and_legacy_scalers_rejected():
    payload = _payload()
    payload["metadata"]["time_basis"] = "calendar_days"
    with pytest.raises(ValueError, match="verified"):
        TimelineCohort(payload)
    payload = _payload()
    payload["encoders"]["treatment_support"] = {}
    with pytest.raises(ValueError, match="drug/scaler/support"):
        TimelineCohort(payload)
    payload = _payload()
    payload["tensors"]["occurred_at"][:] = 0
    with pytest.raises(ValueError, match="fabricated"):
        TimelineCohort(payload)


def test_fold_conversion_physically_excludes_holdouts_and_imports_only_clinical(cohort):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_modality import convert_fold
    ids = cohort.ids
    train, validation, outer, excluded = ids[:24], ids[24:30], ids[30:40], ids[40:]
    cohort.metadata.update(outer_evaluation_ids=outer, excluded_ids=excluded)
    cohort.metadata.update(original_pool_id="synthetic-pool", original_event_id="synthetic-events")
    cohort.encoders = {"clinical": {"synthetic": True}, "fit_ids": train, "treatment_support": {"discard": True}}
    split = {"train": train, "validation": validation, "test": outer + excluded}
    raw = {"artifact_id": "synthetic-pool", "ids": ids,
           "treatments": {patient: {"methods": _methods(), "drugs": object()} for patient in ids},
           "ct0": cohort.tensors["ct0"].clone(), "ct1_tokens": cohort.tensors["ct1"].clone(),
           "labels": torch.stack((cohort.tensors["pcr"], cohort.tensors["binary"]), 1),
           "valid": torch.stack((cohort.tensors["pcr_valid"], cohort.tensors["binary_valid"]), 1)}
    events = {"artifact_id": "synthetic-events", "source_pool_id": "synthetic-pool",
              "patient_ids": ids, "events": torch.ones(len(ids), 3, dtype=torch.long)}
    converted, new_split, report = convert_fold(cohort, split, raw, events, acknowledge_loss_of_information=True)
    assert len(converted) == 40
    assert not set(converted.ids) & set(excluded)
    assert set(converted.encoders) == {"clinical", "fit_ids"}
    assert set(new_split) == {"train", "validation", "outer_evaluation"}
    assert report["old_4x82_sliced"] is False
    assert report["drug_support_imported"] is False
    assert torch.equal(converted.tensors["clinical"], cohort.tensors["clinical"][:40])
    raw["ct0"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="CT features differ"):
        convert_fold(cohort, split, raw, events, acknowledge_loss_of_information=True)


def test_query_order_counts_consumed_slots_and_rejects_pad():
    payload = _payload()
    payload["tensors"]["event_order"] *= 10
    TimelineCohort(payload)
    payload["tensors"]["query_order"] *= 10
    with pytest.raises(ValueError, match="consumed slots"):
        TimelineCohort(payload)
    payload = _payload(torch.tensor([[1, 1, 2]]))
    payload["tensors"]["query_mask"][0, 3] = True
    with pytest.raises(ValueError, match="PAD"):
        TimelineCohort(payload)


def test_calendar_plan_may_be_available_before_future_occurrence():
    payload = _payload()
    payload["metadata"]["time_basis"] = "calendar_days"
    t = payload["tensors"]
    t["occurred_at"] = torch.tensor([[1., 10., 20.]])
    t["available_at"] = torch.tensor([[1., 10., 11.]])
    t["role"][0, 2] = 1
    TimelineCohort(payload)
    t["role"][0, 2] = 0
    with pytest.raises(ValueError, match="before occurrence"):
        TimelineCohort(payload)


def test_duplicate_clinical_fit_membership_rejected():
    payload = _payload()
    payload["ids"] = ["P1", "P2", "P3"]
    payload["tensors"] = {key: value.repeat(3, *([1] * (value.ndim - 1)))
                          for key, value in payload["tensors"].items()}
    payload["encoders"] = {"fit_ids": ["P1", "P1"]}
    cohort = TimelineCohort(payload)
    split = {"train": ["P1"], "validation": ["P2"], "outer_evaluation": ["P3"]}
    with pytest.raises(ValueError, match="exactly"):
        split_indices(cohort, split)


def test_raw_patient_join_normalization():
    from stageworld_tcwm.modality_data import _patient_key
    assert _patient_key(3.0, "n") == "3"
    assert _patient_key(" legacy/path/007 ", "s") == "007"
    assert _patient_key(" legacy\\path\\007 ", "s") == "007"
    for value in ("NA", "NaN", "missing", "..", float("nan"), True):
        assert _patient_key(value, "n") is None


def test_modality_mapping_insertion_order_cannot_create_treatment_sequence():
    methods = _methods()
    original = [{"patient_id": "P1", "methods": methods}]
    reordered = [{"patient_id": "P1", "methods": dict(reversed(list(methods.items())))}]
    source_events = torch.ones(1, 3, dtype=torch.long)
    before = ordinal_event_tensors(original, source_events)
    after = ordinal_event_tensors(reordered, source_events)
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0, equal_nan=True)
