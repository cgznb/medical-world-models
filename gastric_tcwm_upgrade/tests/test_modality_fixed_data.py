import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_modality_fixed", ROOT / "scripts" / "prepare_modality_fixed.py")
fixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixed)


@pytest.fixture
def source():
    ids = [f"SYN-{index:04d}" for index in range(651)]
    labels = torch.tensor([[index % 2, (index // 2) % 2] for index in range(651)], dtype=torch.float32)
    raw = {"artifact_id": "synthetic-complete651", "ids": ids,
           "ct0": torch.zeros(1, 27, 768).expand(651, -1, -1),
           "ct1_tokens": torch.ones(1, 27, 768).expand(651, -1, -1),
           "ct0_valid": torch.ones(651, dtype=torch.bool), "ct1_valid": torch.ones(651, dtype=torch.bool),
           "labels": labels, "valid": torch.ones(651, 2, dtype=torch.bool),
           "clinical": {patient: {"sex": "male" if index % 2 else "female", "age": 40. + index % 30,
                                  "bmi": None if index == 0 else 20. + index % 8,
                                  "ct_stage": "3", "cn_stage": "1", "cm_stage": "0"}
                        for index, patient in enumerate(ids)},
           "treatments": {patient: {"methods": {"chemotherapy": 1, "immunotherapy": 0,
                                                  "targeted": None, "interventional": 0, "hipec": 0}}
                          for patient in ids}}
    events = {"artifact_id": "synthetic-events", "source_pool_id": raw["artifact_id"],
              "patient_ids": list(reversed(ids)), "events": torch.ones(651, 3, dtype=torch.long)}
    return raw, events


@pytest.fixture
def clinical_api():
    configured = os.environ.get("GASTRIC_CLINICAL_SOURCE_ROOT")
    if not configured:
        pytest.skip("Set GASTRIC_CLINICAL_SOURCE_ROOT for optional legacy encoder integration")
    source = Path(configured)
    if not (source / "stageworld" / "data" / "baseline_clinical.py").is_file():
        pytest.skip("The audited original six-field encoder is unavailable")
    return fixed.load_clinical_api(source)


def test_split_has_all651_exact_joint_stratified_counts_and_is_order_stable(source):
    raw, _ = source
    split = fixed.make_fixed_split(raw["ids"], raw["labels"], raw["valid"])
    assert {role: len(members) for role, members in split.items()} == fixed.COUNTS
    assert len(set(patient for members in split.values() for patient in members)) == 651
    repeated = fixed.make_fixed_split(list(reversed(raw["ids"])), raw["labels"].flip(0), raw["valid"].flip(0))
    assert repeated == split
    for role, members in split.items():
        rows = torch.tensor([raw["ids"].index(patient) for patient in members])
        labels = raw["labels"][rows].long()
        counts = torch.bincount(labels[:, 0] + 2*labels[:, 1], minlength=4)
        assert int(counts.max() - counts.min()) <= 1


def test_joint_split_rejects_missing_labels_instead_of_dropping_patients(source):
    raw, _ = source
    raw["valid"][0, 0] = False
    with pytest.raises(ValueError, match="no further filtering"):
        fixed.make_fixed_split(raw["ids"], raw["labels"], raw["valid"])


def test_fixed_conversion_fits_only_new_train_and_ignores_drug_data(source, clinical_api, monkeypatch):
    raw, events = source
    split = fixed.make_fixed_split(raw["ids"], raw["labels"], raw["valid"])
    actual_fit = clinical_api.fit_clinical_transform
    calls = []
    def audited_fit(rows, training_ids, **kwargs):
        assert set(rows) == training_ids == set(split["train"])
        assert all(set(row) == set(fixed.CLINICAL_FIELDS) for row in rows.values())
        calls.append(len(rows))
        return actual_fit(rows, training_ids, **kwargs)
    monkeypatch.setattr(clinical_api, "fit_clinical_transform", audited_fit)
    # Fail loudly if named-treatment encoders are accidentally introduced.
    compact = __import__("stageworld.data.treatment_compact", fromlist=["encode_compact"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Named treatment fitting/encoding must not be called")
    monkeypatch.setattr(compact, "fit_name_support", forbidden)
    monkeypatch.setattr(compact, "encode_compact", forbidden)
    monkeypatch.setattr(compact, "normalize_treatment", forbidden)
    for patient in raw["ids"]:
        raw["treatments"][patient]["drugs"] = object()
        raw["treatments"][patient]["regimens"] = object()
        raw["clinical"][patient]["postoperative_leak"] = object()
    cohort, actual_split, report = fixed.convert_fixed(raw, events, clinical_api,
                                                       acknowledge_loss_of_information=True)
    assert calls == [456]
    assert actual_split == split
    assert len(cohort) == 651
    assert cohort.tensors["clinical"].shape == (651, 32)
    assert cohort.encoders["fit_ids"] == split["train"]
    assert set(cohort.encoders) == {"clinical", "fit_ids"}
    assert cohort.metadata["protocol"] == "fixed651_712"
    assert cohort.metadata["historical_holdouts_repartitioned"] is True
    assert report["patients_excluded"] == 0
    transform = copy.deepcopy(cohort.encoders["clinical"])
    train_rows = torch.tensor([raw["ids"].index(patient) for patient in split["train"]])
    train_features = cohort.tensors["clinical"][train_rows].clone()
    del cohort
    for patient in split["validation"] + split["test"]:
        raw["clinical"][patient]["age"] = 10000.
        raw["clinical"][patient]["bmi"] = 9999.
    changed, _, _ = fixed.convert_fixed(raw, events, clinical_api, acknowledge_loss_of_information=True)
    assert changed.encoders["clinical"] == transform
    assert torch.equal(changed.tensors["clinical"][train_rows], train_features)


def test_reused_partition_is_fixed_and_acknowledgement_required(source, clinical_api):
    raw, events = source
    with pytest.raises(ValueError, match="acknowledge"):
        fixed.convert_fixed(raw, events, clinical_api)
    split = fixed.make_fixed_split(raw["ids"], raw["labels"], raw["valid"])
    split["train"][0], split["test"][0] = split["test"][0], split["train"][0]
    with pytest.raises(ValueError, match="locked seed-17"):
        fixed.convert_fixed(raw, events, clinical_api, split=split, acknowledge_loss_of_information=True)


def test_invalid_ct_or_event_source_does_not_silently_change_cohort(source, clinical_api):
    raw, events = source
    raw["ct0_valid"][0] = False
    with pytest.raises(ValueError, match="no patient may be dropped"):
        fixed.convert_fixed(raw, events, clinical_api, acknowledge_loss_of_information=True)
    raw["ct0_valid"][0] = True
    events["source_pool_id"] = "different-source"
    with pytest.raises(ValueError, match="does not belong"):
        fixed.convert_fixed(raw, events, clinical_api, acknowledge_loss_of_information=True)


def test_fixed_artifacts_are_reusable_and_immutable(tmp_path, source, clinical_api):
    raw, events = source
    pool_path, event_path, out = tmp_path / "pool.pt", tmp_path / "events.pt", tmp_path / "prepared"
    torch.save(raw, pool_path)
    torch.save(events, event_path)
    source_root = Path(clinical_api.__file__).resolve().parents[2]
    report = fixed.prepare(pool_path, event_path, source_root, out, acknowledge_loss_of_information=True)
    assert report["status"] == "passed"
    assert {path.name for path in out.iterdir()} == {"cohort.pt", "split.json", "split_protocol.json", "preparation.json"}
    with pytest.raises(ValueError, match="immutable"):
        fixed.prepare(pool_path, event_path, source_root, out, acknowledge_loss_of_information=True)
    checked = fixed.prepare(pool_path, event_path, source_root, out, acknowledge_loss_of_information=True, verify_only=True)
    assert checked["mode"] == "read_only_revalidation"
    assert checked["patients"] == 651
