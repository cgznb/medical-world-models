import copy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import stat

import pytest
import torch
from torch import nn

from stageworld_tcwm.data import Cohort, file_sha256, fingerprint, write_json
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.model import model_from_config
from stageworld_tcwm.synthetic import synthetic_cohort

spec = importlib.util.spec_from_file_location(
    "evaluate_ct_cv", Path(__file__).resolve().parents[1] / "scripts/evaluate_ct_cv.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.fixture
def nested_runs(tmp_path, config):
    original = synthetic_cohort(n=36, image_dim=config.image_dim, seed=19)
    original.tensors["binary"] = torch.arange(36).remainder(2).float()
    original.tensors["pcr"] = torch.arange(36).remainder(3).eq(0).float()
    permitted, excluded = original.ids[:30], original.ids[30:]
    folders, runs = tmp_path / "folds", tmp_path / "runs"
    for index in range(3):
        cohort = copy.deepcopy(original)
        outer = permitted[index * 10:(index + 1) * 10]
        remaining = [patient for patient in permitted if patient not in outer]
        split = {"train": remaining[:16], "validation": remaining[16:], "test": outer + excluded}
        outer_indices = [cohort.ids.index(patient) for patient in outer]
        excluded_indices = list(range(30, 36))
        for name in ("binary", "pcr", "binary_valid", "pcr_valid", "prefix_valid"):
            cohort.tensors[name][excluded_indices] = 0
        cohort.encoders["fit_ids"] = split["train"]
        cohort.metadata.update({
            "nested_selection": True, "outer_evaluation_ids": outer,
            "outer_evaluation_indices": outer_indices, "excluded_ids": excluded,
            "excluded_indices": excluded_indices, "excluded_scoring_permitted": False,
            "inner_fold": {"index": index, "folds": 3, "nested_selection": True,
                           "original_train_membership_sha256": fingerprint(sorted(permitted))}})
        folder = folders / f"fold-{index}"
        cohort.save(folder / "cohort.pt")
        write_json(split, folder / "split.json")
        write_json({"cohort_sha256": file_sha256(folder / "cohort.pt"),
                    "split_sha256": file_sha256(folder / "split.json")}, folder / "preparation.json")
        train_rows = [cohort.ids.index(patient) for patient in split["train"]]
        for case, architecture in (("old", "token_world"), ("new", "predictive_ct")):
            cfg = replace(config, architecture=architecture, ct_rank=4, clinical_anchor=True)
            model = model_from_config(cfg)
            model.fit_statistics(cohort.batch(train_rows))
            model.fit_clinical_anchors(cohort.batch(train_rows))
            contract = {"cohort_sha256": file_sha256(folder / "cohort.pt"),
                        "split": fingerprint(split), "model": asdict(cfg), "train": {"seed": 17}}
            contract_id = fingerprint(contract)
            target = runs / case / f"fold-{index}"
            target.mkdir(parents=True)
            write_json({"id": contract_id, "contract": contract}, target / "contract.json")
            torch.save({"schema": "tcwm-inference-v1", "model_config": asdict(cfg),
                        "model_state": model.state_dict(), "locked_selection": True,
                        "selected_epoch": 0, "contract_id": contract_id, "support": {},
                        "metadata": cohort.metadata, "encoders": cohort.encoders}, target / "inference.pt")
    return folders, runs, permitted


def test_outer_only_evaluation_loads_both_model_aliases_and_exports_private_oof(nested_runs, tmp_path, monkeypatch):
    folders, runs, permitted = nested_runs
    original_batch = Cohort.batch

    def guarded_batch(self, indices, device="cpu"):
        assert not set(torch.as_tensor(indices).tolist()) & set(self.metadata["excluded_indices"])
        return original_batch(self, indices, device)

    monkeypatch.setattr(Cohort, "batch", guarded_batch)
    destination = tmp_path / "output"
    report = evaluation.evaluate(folders, runs, ["old", "new"], destination,
                                 samples=2, batch_size=4)
    assert report["outer_oof_patients"] == 30
    assert report["original_validation_or_test_scored"] is False
    assert report["clinical_baseline"]["oof"]["patients"] == 30
    assert report["cases"]["old"]["absolute_ct1"]["supported"] is False
    future = report["cases"]["new"]["absolute_ct1"]
    assert future["patients"] == 30 and future["elements"] == 30 * 8
    assert future["prediction_mse"] == pytest.approx(future["copy_ct0_mse"])
    for case in report["cases"].values():
        assert case["oof"]["stages"]["S0"]["ct1_permutation"]["maximum_absolute"] == 0
        assert case["oof"]["pcr"]["ct1_permutation"]["maximum_absolute"] == 0
    private = torch.load(destination / "oof_probabilities.pt", weights_only=True)
    for values in private["cases"].values():
        assert values["patient_ids"] == sorted(permitted)
        assert len(values["patient_ids"]) == len(set(values["patient_ids"]))
    text = (destination / "report.json").read_text()
    assert all(patient not in text for patient in permitted)
    assert stat.S_IMODE((destination / "oof_probabilities.pt").stat().st_mode) == 0o600


def test_rejects_nonredacted_excluded_labels_and_index_mismatch(nested_runs):
    folders, _, _ = nested_runs
    cohort = Cohort.load(folders / "fold-0/cohort.pt")
    split = json.loads((folders / "fold-0/split.json").read_text())
    cohort.tensors["binary_valid"][cohort.metadata["excluded_indices"][0]] = True
    with pytest.raises(ValueError, match="redacted"):
        evaluation.validate_fold(cohort, split, 0)
    cohort.tensors["binary_valid"][cohort.metadata["excluded_indices"][0]] = False
    cohort.metadata["outer_evaluation_indices"][0] = cohort.metadata["excluded_indices"][0]
    with pytest.raises(ValueError, match="indices"):
        evaluation.validate_fold(cohort, split, 0)


def test_run_requires_same_inner_fit_membership_and_fold_hash(nested_runs):
    folders, runs, _ = nested_runs
    folds, _ = evaluation.load_folds(folders)
    path = runs / "old/fold-0/inference.pt"
    predictor = Predictor(path)
    predictor.bundle["encoders"]["fit_ids"] = predictor.bundle["encoders"]["fit_ids"][:-1]
    with pytest.raises(ValueError, match="training"):
        evaluation.verify_run(predictor, path, folds[0])
    predictor = Predictor(path)
    folds[0]["hashes"]["cohort"] = "wrong"
    with pytest.raises(ValueError, match="different cohort"):
        evaluation.verify_run(predictor, path, folds[0])


def test_pool_rejects_duplicate_or_missing_outer_patients():
    chunk = {"patient_ids": ["a", "b"], "probability": torch.ones(2, 2)}
    with pytest.raises(ValueError, match="exactly once"):
        evaluation.pool_outer([chunk, chunk], ["a", "b"])
    with pytest.raises(ValueError, match="exactly once"):
        evaluation.pool_outer([chunk], ["a", "b", "c"])
    result = evaluation.pool_outer([chunk], ["b", "a"])
    assert result["patient_ids"] == ["b", "a"]


def test_mean_prefix_nll_weights_patients_equally_not_pooled_prefixes():
    values = {"patient_ids": ["a", "b"], "binary": torch.tensor([0., 1.]),
              "pcr": torch.tensor([0., 1.]), "binary_valid": torch.tensor([True, True]),
              "pcr_valid": torch.tensor([True, True]),
              "prefix_valid": torch.tensor([[True, True, True], [True, False, False]]),
              "probability": torch.full((2, 3), .5), "permuted_probability": torch.full((2, 3), .5),
              "nll": torch.tensor([[1., 2., 3.], [8., 99., 99.]]),
              "pcr_probability": torch.full((2,), .5), "permuted_pcr_probability": torch.full((2,), .5),
              "pcr_nll": torch.ones(2)}
    report = evaluation.summarize(values)
    assert report["mean_prefix_nll"] == 5.0
    assert set(report["stages"]) == {"S0", "S1", "S2"}


def test_absolute_ct_metrics_pool_errors_instead_of_averaging_fold_r2():
    first = evaluation.ct_error_sums(torch.tensor([[1.]]), torch.tensor([[2.]]),
                                    torch.tensor([[0.]]), torch.tensor([0.]), torch.tensor([True]))
    second = evaluation.ct_error_sums(torch.tensor([[2.], [2.]]), torch.tensor([[4.], [4.]]),
                                     torch.zeros(2, 1), torch.tensor([1.]), torch.ones(2, dtype=torch.bool))
    pooled = evaluation.ct_summary([first, second])
    assert pooled["prediction_mse"] == pytest.approx(3.0)
    assert pooled["prediction_r2_vs_training_mean"] == pytest.approx(1 - 9 / 22)
    assert pooled["copy_ct0_r2_vs_training_mean"] == pytest.approx(1 - 36 / 22)
    assert "delta" not in pooled["target"].lower()


def test_ct_training_mean_uses_only_supplied_fit_rows(cohort):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.device_parameter = nn.Parameter(torch.zeros(()))

        def encode_image(self, images):
            return images.mean((1, 2))[:, None]

    cohort.tensors["ct1"][0].fill_(1)
    cohort.tensors["ct1"][1].fill_(3)
    cohort.tensors["ct1"][2:].fill_(1000)
    mean = evaluation.target_training_mean(Encoder(), cohort, torch.tensor([0, 1]), 1, 1)
    torch.testing.assert_close(mean, torch.tensor([2.], dtype=torch.float64))


def test_ct1_leak_into_s0_is_a_hard_failure(nested_runs):
    folders, _, _ = nested_runs
    folds, _ = evaluation.load_folds(folders)

    class Leaky(nn.Module):
        def __init__(self):
            super().__init__()
            self.device_parameter = nn.Parameter(torch.zeros(()))

        def forward(self, batch, samples, **kwargs):
            future = batch["ct1"].mean((1, 2))
            return {"predictions": future[:, None, None].expand(-1, 2, samples),
                    "pcr_logits": torch.zeros(len(future), samples)}

    with pytest.raises(ValueError, match="S0 changed"):
        evaluation.model_outer(Leaky(), folds[0], 2, 4, 17, 19)
