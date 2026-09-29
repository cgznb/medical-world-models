import copy
import importlib.util
from pathlib import Path
import stat

import numpy as np
import pytest
import torch

from stageworld_tcwm.data import Cohort, file_sha256, fingerprint, write_json
from stageworld_tcwm.diagnostic_baselines import (
    FixedFeatures, fit_fold, fit_logistic, logistic_objective, paired_comparison,
)
from stageworld_tcwm.synthetic import synthetic_cohort


spec = importlib.util.spec_from_file_location(
    "run_diagnostic_baselines", Path(__file__).resolve().parents[1] / "scripts/run_diagnostic_baselines.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def diagnostic_folds(tmp_path):
    original = synthetic_cohort(n=36, image_dim=16, seed=19)
    original.tensors["binary"] = torch.arange(36).remainder(2).float()
    permitted, excluded = original.ids[:30], original.ids[30:]
    folders = tmp_path / "folds"
    for index in range(3):
        cohort = copy.deepcopy(original)
        outer = permitted[index * 10:(index + 1) * 10]
        remaining = [patient for patient in permitted if patient not in outer]
        split = {"train": remaining[:16], "validation": remaining[16:], "test": outer + excluded}
        for name in ("binary", "pcr", "binary_valid", "pcr_valid", "prefix_valid"):
            cohort.tensors[name][30:] = 0
        cohort.encoders["fit_ids"] = split["train"]
        cohort.metadata.update({
            "nested_selection": True, "outer_evaluation_ids": outer,
            "outer_evaluation_indices": [cohort.ids.index(patient) for patient in outer],
            "excluded_ids": excluded, "excluded_indices": list(range(30, 36)),
            "excluded_scoring_permitted": False,
            "inner_fold": {"index": index, "folds": 3, "nested_selection": True,
                           "original_train_membership_sha256": fingerprint(sorted(permitted))}})
        folder = folders / f"fold-{index}"
        cohort.save(folder / "cohort.pt")
        write_json(split, folder / "split.json")
        write_json({"cohort_sha256": file_sha256(folder / "cohort.pt"),
                    "split_sha256": file_sha256(folder / "split.json")}, folder / "preparation.json")
    return folders, permitted


def test_logistic_matches_explicit_regularized_stationary_equations():
    x = torch.randn(60, 4, dtype=torch.float64)
    y = (torch.arange(60) % 3 == 0).double()
    penalty = .1
    result = fit_logistic(x, y, penalty)
    residual = result.logits(x).sigmoid() - y
    weight_gradient = x.T @ residual / len(y) + penalty * result.parameters[:-1]
    assert weight_gradient.abs().max() < 1e-6
    assert residual.mean().abs() < 1e-6
    assert result.report["converged"]
    assert result.parameters.dtype == torch.float64
    theta = result.parameters.clone().requires_grad_()
    objective = logistic_objective(theta, x, y, penalty)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(result.logits(x), y)
    expected += penalty / 2 * result.parameters[:-1].square().sum()
    torch.testing.assert_close(objective.detach(), expected)
    again = fit_logistic(x, y, penalty)
    torch.testing.assert_close(again.parameters, result.parameters, rtol=0, atol=0)


def test_intercept_not_penalized_even_under_strong_shrinkage():
    x = torch.zeros(10, 3)
    y = torch.tensor([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])
    fitted = fit_logistic(x, y, 100.)
    torch.testing.assert_close(fitted.logits(x).sigmoid(), torch.full((10,), .3, dtype=torch.float64))
    assert torch.equal(fitted.parameters[:-1], torch.zeros(3, dtype=torch.float64))


def test_projection_only_fits_training_and_has_shared_target_space(cohort):
    rows = torch.arange(24)
    first = FixedFeatures(4).fit(cohort.batch(rows), cohort.ids[:24])
    before = first.transform(cohort.batch(torch.arange(24, 48)))
    cohort.tensors["ct1"][24:] += 1000
    cohort.tensors["clinical"][24:] += 1000
    second = FixedFeatures(4).fit(cohort.batch(rows), cohort.ids[:24])
    assert first.target_space_id == second.target_space_id
    torch.testing.assert_close(first.basis, second.basis, rtol=0, atol=0)
    assert first.provenance["fit_patients"] == 24
    assert first.encode_image(cohort.tensors["ct0"][:1]).shape == (1, 8)
    assert before["B3"].shape == (24, 361 + 9)
    assert before["B4"].shape == (24, 361 + 18)
    assert FixedFeatures(8).fit(cohort.batch(rows), cohort.ids[:24]).target_space_id != first.target_space_id


def test_information_boundaries_exclude_ct1_and_illegal_observations(cohort):
    transform = FixedFeatures(4).fit(cohort.batch(torch.arange(24)), cohort.ids[:24])
    batch = cohort.batch(torch.arange(24, 30))
    batch["ct1_available_stage"][:3] = 2
    original = transform.transform(batch)
    batch["ct1"] += 10000
    batch["binary"] = 1 - batch["binary"]
    batch["pcr"] = 1 - batch["pcr"]
    altered = transform.transform(batch)
    for name in ("B1", "B2", "B3"):
        torch.testing.assert_close(original[name], altered[name], rtol=0, atol=0)
    torch.testing.assert_close(original["B4"][:3], altered["B4"][:3], rtol=0, atol=0)
    assert not torch.equal(original["B4"][3:], altered["B4"][3:])


def test_outer_labels_never_select_lambda_or_prevalence(diagnostic_folds):
    folders, _ = diagnostic_folds
    folds, _ = runner.load_folds(folders)
    fold = folds[0]
    first, predictions, state = fit_fold(fold, 4)
    outer = fold["rows"]["outer_evaluation"]
    fold["cohort"].tensors["binary"][outer] = 1 - fold["cohort"].tensors["binary"][outer]
    second, changed, updated = fit_fold(fold, 4)
    for name in ("B1", "B2", "B3", "B4"):
        torch.testing.assert_close(state["classifiers"][name], updated["classifiers"][name], rtol=0, atol=0)
        assert first["models"][name]["selected_fit"]["lambda"] == second["models"][name]["selected_fit"]["lambda"]
    assert np.array_equal(predictions["probabilities"]["B0"], changed["probabilities"]["B0"])
    assert state["training_prevalence"] == .5
    fold["cohort"].encoders["fit_ids"].append(fold["cohort"].metadata["outer_evaluation_ids"][0])
    with pytest.raises(ValueError, match="exactly the inner"):
        fit_fold(fold, 4)


def test_paired_bootstrap_identical_predictions_have_zero_difference():
    y = np.tile([0., 1.], 12)
    p = np.linspace(.1, .9, len(y))
    result = paired_comparison(y, p, p, np.repeat(np.arange(3), 8), bootstrap=20)
    assert all(value == 0 for value in result["delta_new_minus_reference"].values())
    assert all(value["low"] == value["high"] == 0 for value in result["bootstrap_95_percent"].values())
    assert result["folds_improved"]["auroc"] == 0


def test_cli_contract_omits_holdouts_and_exports_private_case_differences(diagnostic_folds, tmp_path, monkeypatch):
    folders, permitted = diagnostic_folds
    original_batch = Cohort.batch

    def guard(self, indices, device="cpu"):
        assert not set(torch.as_tensor(indices).tolist()) & set(self.metadata["excluded_indices"])
        return original_batch(self, indices, device)

    monkeypatch.setattr(Cohort, "batch", guard)
    destination = tmp_path / "diagnostics"
    report = runner.run(folders, destination, ranks=[4], bootstrap=10)
    assert report["original_validation_or_test_scored"] is False
    assert report["full_outer_oof"] is True
    assert report["results"]["4"]["pooled_outer"]["B4"]["n"] == 30
    private = torch.load(destination / "paired_predictions_private.pt", weights_only=True)
    values = private["ranks"]["4"]
    assert len(set(values["case_keys"])) == 30
    assert all(patient not in repr(private) for patient in permitted)
    assert all(patient not in (destination / "report.json").read_text() for patient in permitted)
    assert set(values["paired_deltas"]) == {"B2_vs_B1", "B3_vs_B2", "B4_vs_B3"}
    assert stat.S_IMODE((destination / "paired_predictions_private.pt").stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="new output"):
        runner.run(folders, destination, ranks=[4], bootstrap=0)


def test_partial_fold_run_is_explicitly_not_full_oof(diagnostic_folds, tmp_path):
    folders, _ = diagnostic_folds
    report = runner.run(folders, tmp_path / "partial", ranks=[4], fold_indices=[1], bootstrap=0)
    assert report["full_outer_oof"] is False
    assert report["fold_indices"] == [1]
    assert report["results"]["4"]["pooled_outer"]["B0"]["n"] == 10
