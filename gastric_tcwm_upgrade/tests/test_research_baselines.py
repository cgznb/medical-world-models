"""Anonymous engineering checks for fit boundaries and deployable baseline exports."""
from copy import deepcopy
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from stageworld_tcwm.research_baselines import fit_baseline, predict_baseline


class TinyCohort(SimpleNamespace):
    def __len__(self):
        return len(self.ids)


def cohort_fixture():
    generator = torch.Generator().manual_seed(81)
    n = 44
    x = torch.zeros(n, 32)
    for start, size in ((0, 3), (7, 12), (19, 9), (28, 4)):
        category = torch.randint(size, (n,), generator=generator)
        x[torch.arange(n), start + category] = 1
    x[:, 3] = torch.randn(n, generator=generator)
    x[:, 5] = torch.randn(n, generator=generator)
    x[:, [4, 6]] = 1
    x[2, 5:7] = 0
    ids = [f"anonymous_{i}" for i in range(n)]
    return TinyCohort(ids=ids, encoders={"fit_ids": ids[:30], "clinical": {
        "schema_version": "gastric-baseline-6-fields-v1", "fit_split": "train",
        "training_patients": 30,
        "continuous": {"age": {"mean": 60., "scale": 10.}, "bmi": {"mean": 23., "scale": 3.}}}},
        tensors={"clinical": x, "ct0": torch.randn(n, 27, 10, generator=generator),
                 "ct1": torch.randn(n, 27, 10, generator=generator),
                 "binary": torch.arange(n).remainder(2).float(), "binary_valid": torch.ones(n, dtype=torch.bool),
                 "pcr": torch.arange(n).remainder(2).float(), "image_valid": torch.ones(n, 2, dtype=torch.bool)})


@pytest.mark.parametrize("experiment", ["D00", "D01", "D02", "D03", "D04"])
def test_fit_never_reads_future_or_unsupplied_rows_and_export_roundtrips(tmp_path, experiment):
    cohort = cohort_fixture()
    train, validation = torch.arange(30), torch.arange(30, 38)
    first = fit_baseline(experiment, cohort, train, validation, tmp_path / "first")
    artifact = joblib.load(tmp_path / "first" / "inference.joblib")
    expected = predict_baseline(artifact, cohort, validation)
    changed = deepcopy(cohort)
    # Poison future data for every patient and all inputs/outcomes of held-out test rows.
    changed.tensors["ct1"][:] = float("nan")
    changed.tensors["pcr"][:] = float("nan")
    changed.tensors["image_valid"][:, 1] = False
    for key in ("clinical", "ct0", "binary"):
        changed.tensors[key][38:] = float("nan")
    second = fit_baseline(experiment, changed, train, validation, tmp_path / "second")
    assert first["train"] == second["train"]
    assert first["validation"] == second["validation"]
    assert first["test_scored"] is False
    np.testing.assert_array_equal(expected, predict_baseline(artifact, changed, validation))
    predictions = np.load(tmp_path / "first" / "predictions.private.npz")
    np.testing.assert_array_equal(expected, predictions["validation_probabilities"])
    assert not any("test" in key for key in predictions.files)
    assert (tmp_path / "first" / "predictions.private.npz").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "first" / "inference.joblib").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("experiment", ["D01", "D02", "D03", "D04"])
def test_validation_changes_cannot_change_fitted_transform_or_training_predictions(tmp_path, experiment):
    cohort = cohort_fixture()
    train, validation = torch.arange(30), torch.arange(30, 38)
    fit_baseline(experiment, cohort, train, validation, tmp_path / "first")
    changed = deepcopy(cohort)
    changed.tensors["clinical"][30:38, 3] += 1000
    changed.tensors["clinical"][30:38, 5] -= 1000
    changed.tensors["ct0"][30:38] *= 1000
    changed.tensors["binary"][30:38] = 1 - changed.tensors["binary"][30:38]
    fit_baseline(experiment, changed, train, validation, tmp_path / "second")
    first = joblib.load(tmp_path / "first" / "inference.joblib")
    second = joblib.load(tmp_path / "second" / "inference.joblib")
    np.testing.assert_array_equal(predict_baseline(first, cohort, train), predict_baseline(second, cohort, train))
    if experiment in ("D01", "D04"):
        np.testing.assert_array_equal(first["clinical_scaler"].mean_, second["clinical_scaler"].mean_)
    if experiment == "D02":
        assert first["gam"].description() == second["gam"].description()
        assert first["gam"].continuous_["age"]["spline"].n_features_out_ == 3
    elif experiment in ("D03", "D04"):
        np.testing.assert_array_equal(first["ct_pca"].components_, second["ct_pca"].components_)


def test_d01_matches_historical_train_only_scaling_and_d00_masks_labels(tmp_path):
    cohort = cohort_fixture()
    train, validation = torch.arange(30), torch.arange(30, 38)
    result = fit_baseline("D01", cohort, train, validation, tmp_path / "linear")
    reference = make_pipeline(StandardScaler(), LogisticRegression(
        C=1, solver="lbfgs", max_iter=2000, tol=1e-8, random_state=17)).fit(
        cohort.tensors["clinical"][:30].double().numpy(), cohort.tensors["binary"][:30].numpy())
    artifact = joblib.load(tmp_path / "linear" / "inference.joblib")
    np.testing.assert_array_equal(predict_baseline(artifact, cohort, validation),
                                  reference.predict_proba(cohort.tensors["clinical"][30:38].double().numpy())[:, 1])
    assert result["feature_count"] == 32
    assert "clinical_scaler" in artifact
    # The same fitted anchor can initialize a raw-clinical linear layer exactly.
    classifier, scaler = artifact["classifier"], artifact["clinical_scaler"]
    weight = classifier.coef_[0] / scaler.scale_
    bias = classifier.intercept_[0] - np.dot(weight, scaler.mean_)
    logits = cohort.tensors["clinical"][validation].double().numpy() @ weight + bias
    np.testing.assert_allclose(1 / (1 + np.exp(-logits)), predict_baseline(artifact, cohort, validation), atol=1e-14)
    cohort.tensors["binary_valid"][:10] = False
    cohort.tensors["binary"][:10] = float("nan")
    constant = fit_baseline("D00", cohort, train, validation, tmp_path / "constant")
    assert constant["training_prevalence"] == .5
    assert constant["fit_endpoint_patients"] == 20
    assert constant["status"] == "partially_completed"
    assert "blocked" in constant["subtasks"]["identity"]


def test_clinical_scaler_matches_historical_observed_endpoint_fit_mask(tmp_path):
    cohort = cohort_fixture()
    train, validation = torch.arange(30), torch.arange(30, 38)
    cohort.tensors["binary_valid"][:4] = False
    cohort.tensors["clinical"][:4, 3] += 1000
    fit_baseline("D01", cohort, train, validation, tmp_path / "masked")
    artifact = joblib.load(tmp_path / "masked" / "inference.joblib")
    expected = cohort.tensors["clinical"][4:30].double().numpy().mean(axis=0)
    np.testing.assert_array_equal(artifact["clinical_scaler"].mean_, expected)


def test_rejects_overlap_and_encoder_fit_mismatch(tmp_path):
    cohort = cohort_fixture()
    with pytest.raises(ValueError, match="disjoint"):
        fit_baseline("D01", cohort, torch.arange(30), torch.arange(29, 38), tmp_path / "overlap")
    with pytest.raises(ValueError, match="exactly"):
        fit_baseline("D01", cohort, torch.arange(29), torch.arange(30, 38), tmp_path / "bad_fit")
