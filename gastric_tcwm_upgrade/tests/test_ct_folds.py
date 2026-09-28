import importlib.util
from pathlib import Path
import copy
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "prepare_ct_folds",Path(__file__).resolve().parents[1]/"scripts/prepare_ct_folds.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def split_case():
    ids = [f"p{index}" for index in range(18)]
    original = {"train":ids[:12],"validation":ids[12:15],"test":ids[15:]}
    labels = torch.tensor([0,1]*6+[float("nan")]*6)
    valid = torch.tensor([True]*12+[False]*6)
    return ids,original,labels,valid


def test_internal_folds_exclude_original_holdouts_and_cover_train_once():
    ids,original,labels,valid = split_case()
    folds = adapter.make_internal_splits(original,ids,labels,valid)
    selected = []
    for fold in folds:
        assert len(fold["train"])==8 and len(fold["validation"])==4
        assert set(fold["train"]).isdisjoint(fold["validation"])
        assert set(fold["train"]+fold["validation"])==set(original["train"])
        assert fold["test"]==original["validation"]+original["test"]
        indices = [ids.index(patient) for patient in fold["validation"]]
        assert int(labels[indices].sum())==2
        selected.extend(fold["validation"])
    assert len(selected)==len(set(selected))==12
    assert set(selected)==set(original["train"])
    labels[12:] = 12345
    valid[12:] = True
    assert adapter.make_internal_splits(original,ids,labels,valid)==folds


@pytest.mark.parametrize("problem",["overlap","missing_label","insufficient_classes"])
def test_invalid_internal_partition_inputs_fail(problem):
    ids,original,labels,valid = split_case()
    if problem=="overlap":
        original["test"].append(original["train"][0])
    elif problem=="missing_label":
        valid[0] = False
    else:
        labels[:12] = 0
    with pytest.raises(ValueError):
        adapter.make_internal_splits(original,ids,labels,valid)


def test_excluded_labels_redacted_without_modifying_source_tensors(cohort):
    ids = cohort.ids
    split = {"train":ids[:24],"validation":ids[24:36],"test":ids[36:]}
    before = copy.deepcopy(cohort.tensors)
    original_binary = cohort.tensors["binary"]
    original_valid = cohort.tensors["binary_valid"]
    adapter.exclude_holdout_labels(cohort,split)
    for name in ("binary","pcr"):
        assert torch.equal(cohort.tensors[name][:36],before[name][:36])
        assert torch.equal(cohort.tensors[name+"_valid"][:36],before[name+"_valid"][:36])
        assert not bool(cohort.tensors[name][36:].any())
        assert not bool(cohort.tensors[name+"_valid"][36:].any())
    assert not bool(cohort.tensors["prefix_valid"][36:].any())
    assert torch.equal(original_binary,before["binary"])
    assert torch.equal(original_valid,before["binary_valid"])
    assert cohort.metadata["excluded_scoring_permitted"] is False
    assert cohort.metadata["excluded_indices"]==list(range(36,48))


def test_encoder_audit_refits_only_fold_training_rows(cohort,monkeypatch):
    ids = cohort.ids
    split = {"train":ids[:24],"validation":ids[24:36],"test":ids[36:]}
    clinical = {patient:{"patient_id":patient,"value":float(index)} for index,patient in enumerate(ids)}
    treatment = copy.deepcopy(clinical)
    seen = []
    def fit_clinical(rows,fit_ids,schema_version):
        assert set(rows)==fit_ids==set(split["train"])
        seen.append("clinical")
        return {"mean":sum(row["value"] for row in rows.values())/len(rows)}
    def fit_treatment(rows,fit_ids):
        assert {row["patient_id"] for row in rows}==fit_ids==set(split["train"])
        seen.append("treatment")
        return {"count":len(rows)}
    class Encoded:
        def __init__(self,rows,transform):
            self.rows,self.transform = rows,transform
        def ridge_features(self):
            return torch.tensor([[row["value"]-self.transform["mean"]]*32 for row in self.rows])
    def encode_treatment(rows,support):
        return torch.full((len(rows),4,82),float(support["count"])),[]
    clinical_transform = {"mean":11.5}
    treatment_support = {"count":24}
    cohort.encoders = {"fit_ids":split["train"],"clinical":clinical_transform,"treatment_support":treatment_support}
    cohort.tensors["clinical"] = Encoded(list(clinical.values()),clinical_transform).ridge_features()
    cohort.tensors["treatment"] = encode_treatment(list(treatment.values()),treatment_support)[0]
    monkeypatch.setattr(adapter,"original_encoders",lambda:(Encoded,fit_clinical,"test",encode_treatment,fit_treatment))
    result = adapter.verify_fold_encoders(cohort,{"clinical":clinical,"treatments":treatment},split)
    assert result["fit_patients"]==24 and seen==["clinical","treatment"]
    cohort.encoders["fit_ids"] = split["train"]+split["validation"]
    with pytest.raises(ValueError,match="exactly this fold"):
        adapter.verify_fold_encoders(cohort,{"clinical":clinical,"treatments":treatment},split)


def test_nested_selection_never_uses_outer_holdout_or_original_holdouts():
    ids,original,labels,valid = split_case()
    outer_folds = adapter.make_internal_splits(original,ids,labels,valid)
    evaluation = []
    for index,outer in enumerate(outer_folds):
        split,outer_ids = adapter.add_nested_selection(outer,ids,labels,valid,20260928+index)
        assert len(split["train"])==6 and len(split["validation"])==2
        assert set(split["train"]+split["validation"])==set(outer["train"])
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["train"]+split["validation"]).isdisjoint(outer_ids)
        assert set(split["test"])==set(outer_ids+original["validation"]+original["test"])
        changed_labels,changed_valid = labels.clone(),valid.clone()
        other_rows = [ids.index(patient) for patient in split["test"]]
        changed_labels[other_rows] = float("nan")
        changed_valid[other_rows] = False
        assert adapter.add_nested_selection(outer,ids,changed_labels,changed_valid,20260928+index)==(split,outer_ids)
        evaluation.extend(outer_ids)
    assert len(evaluation)==len(set(evaluation))==len(original["train"])
    assert set(evaluation)==set(original["train"])


def test_nested_redaction_preserves_outer_evaluation_targets(cohort):
    ids = cohort.ids
    split = {"train":ids[:18],"validation":ids[18:24],"test":ids[24:]}
    outer_ids,excluded_ids = ids[24:36],ids[36:]
    before = copy.deepcopy(cohort.tensors)
    adapter.exclude_holdout_labels(cohort,split,excluded_ids)
    for name in ("binary","pcr","binary_valid","pcr_valid","prefix_valid"):
        assert torch.equal(cohort.tensors[name][:36],before[name][:36])
        assert not bool(cohort.tensors[name][36:].any())
    assert set(outer_ids).isdisjoint(cohort.metadata["excluded_ids"])
    assert cohort.metadata["excluded_test_role"] is False
    with pytest.raises(ValueError,match="test role"):
        adapter.exclude_holdout_labels(cohort,split,split["train"])


def test_nested_encoder_binding_rejects_outer_patients(cohort):
    ids = cohort.ids
    split = {"train":ids[:18],"validation":ids[18:24],"test":ids[24:]}
    cohort.encoders = {"fit_ids":split["train"]+[ids[24]]}
    with pytest.raises(ValueError,match="exactly this fold"):
        adapter.verify_fold_encoders(cohort,{},split)
