"""Fixed, train-only D00--D04 references for the 2026-10-01 research queue.

These models predict the recorded terminal recurrence endpoint. They do not
implement concept dynamics, intermediate risk or counterfactual treatment effects.
Only explicitly supplied train/validation rows are read by ``fit_baseline``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
import torch

from .data import fingerprint
from .timeline_training import binary_metrics


EXPERIMENTS = ("D00", "D01", "D02", "D03", "D04")
CLINICAL_SCHEMA = "gastric-baseline-6-fields-v1"
CATEGORICAL_FIELDS = (
    ("sex", 0, ("missing", "male", "female")),
    ("ct_stage", 7, ("missing", "0", "is", "1", "1a", "1b", "2", "3", "4", "4a", "4b", "x")),
    ("cn_stage", 19, ("missing", "0", "1", "2", "3", "3a", "3b", "x", "+")),
    ("cm_stage", 28, ("missing", "0", "1", "x")),
)
CONTINUOUS_FIELDS = (("age", 3, 4), ("bmi", 5, 6))


def _indices(values, size):
    raw = torch.as_tensor(values)
    if raw.ndim != 1 or raw.dtype not in (torch.int32, torch.int64):
        raise ValueError("Patient indices must be one-dimensional integers")
    rows = raw.cpu().long()
    if not len(rows) or len(rows.unique()) != len(rows) or (rows < 0).any() or (rows >= size).any():
        raise ValueError("Patient indices must be nonempty, unique and in range")
    return rows


def _clinical(cohort, rows):
    value = cohort.tensors["clinical"][rows].detach().cpu().double().numpy()
    if value.shape != (len(rows), 32) or not np.isfinite(value).all():
        raise ValueError("Expected finite six-field clinical32 encoding")
    return value


def _ct0_moments(cohort, rows):
    if not bool(cohort.tensors["image_valid"][rows, 0].all()):
        raise ValueError("D03/D04 require observed CT0 for every supplied patient")
    image = cohort.tensors["ct0"][rows].detach().cpu().double().numpy()
    if image.ndim != 3 or image.shape[1] != 27 or not np.isfinite(image).all():
        raise ValueError("Expected finite CT0 features with 27 tokens")
    return np.concatenate((image.mean(axis=1), image.std(axis=1, ddof=0)), axis=1)


class NamedClinicalGAM:
    """Additive design with exactly three cubic B-spline columns per continuous field.

    df=3 means degree=3, two boundary knots at the observed training min/max,
    include_bias=False; extrapolation is constant. Basis columns are centered
    using training rows. Missing continuous values use the observed training
    mean and their own indicator. Fixed medical category one-hot main effects
    include a missing category; no interactions or outcome-dependent knots.
    """

    def __init__(self, encoder):
        if encoder.get("schema_version") != CLINICAL_SCHEMA:
            raise ValueError("D02 requires the audited six-field clinical encoder")
        self.encoder = encoder

    def _raw(self, x, name, value_column, observed_column):
        stats = self.encoder["continuous"][name]
        mean, scale = float(stats["mean"]), float(stats["scale"])
        if not np.isfinite([mean, scale]).all() or scale <= 0:
            raise ValueError("Invalid frozen clinical inverse transform")
        observed = x[:, observed_column]
        if not np.isin(observed, [0., 1.]).all():
            raise ValueError("Continuous clinical observation mask is not binary")
        return x[:, value_column] * scale + mean, observed.astype(bool)

    def fit(self, x):
        self.continuous_ = {}
        self.feature_names_ = []
        for name, value_column, observed_column in CONTINUOUS_FIELDS:
            raw, observed = self._raw(x, name, value_column, observed_column)
            known = raw[observed]
            mean = float(known.mean()) if len(known) else 0.
            knots = [float(known.min()), float(known.max())] if len(known) else [0., 0.]
            imputed = np.where(observed, raw, mean)[:, None]
            spline = None
            basis = np.zeros((len(x), 3))
            if knots[1] > knots[0]:
                spline = SplineTransformer(degree=3, knots=np.asarray(knots)[:, None],
                                           include_bias=False, extrapolation="constant")
                basis = spline.fit_transform(imputed)
                if basis.shape[1] != 3:
                    raise RuntimeError("The frozen df=3 spline contract changed")
            self.continuous_[name] = {"mean": mean, "knots": knots, "spline": spline,
                                      "basis_mean": basis.mean(axis=0),
                                      "observed_training_patients": int(observed.sum())}
            self.feature_names_.extend([f"{name}_spline_{j}" for j in range(3)] + [f"{name}_missing"])
        for name, _, categories in CATEGORICAL_FIELDS:
            self.feature_names_.extend(f"{name}={category}" for category in categories)
        self.transform(x)  # Validate the declared categorical contract before fitting.
        return self

    def transform(self, x):
        columns = []
        for name, value_column, observed_column in CONTINUOUS_FIELDS:
            raw, observed = self._raw(x, name, value_column, observed_column)
            state = self.continuous_[name]
            values = np.where(observed, raw, state["mean"])[:, None]
            basis = (state["spline"].transform(values) if state["spline"] is not None
                     else np.zeros((len(x), 3)))
            columns.extend((basis - state["basis_mean"], (~observed)[:, None].astype(float)))
        for _, start, categories in CATEGORICAL_FIELDS:
            encoded = x[:, start:start + len(categories)]
            if not np.isin(encoded, [0., 1.]).all() or not (encoded.sum(axis=1) == 1).all():
                raise ValueError("Clinical category must have exactly one active fixed-ontology value")
            columns.append(encoded)
        return np.concatenate(columns, axis=1)

    def description(self):
        return {"continuous_basis": "cubic B-spline df3; two observed-train boundary knots; "
                                    "include_bias=False; constant extrapolation; train-centered",
                "missing": "observed-train mean imputation plus explicit missing indicator",
                "categorical": "fixed medical ontology one-hot main effects, including missing",
                "interactions": False,
                "feature_names": self.feature_names_,
                "continuous_training_statistics": {
                    name: {key: value for key, value in state.items() if key in
                           ("mean", "knots", "observed_training_patients")}
                    for name, state in self.continuous_.items()}}


def _features(artifact, cohort, rows):
    experiment = artifact["experiment_id"]
    if experiment == "D00":
        return np.zeros((len(rows), 0))
    if experiment == "D01":
        return artifact["clinical_scaler"].transform(_clinical(cohort, rows))
    if experiment == "D02":
        return artifact["gam"].transform(_clinical(cohort, rows))
    moments = _ct0_moments(cohort, rows)
    image = artifact["ct_score_scaler"].transform(
        artifact["ct_pca"].transform(artifact["ct_moment_scaler"].transform(moments)))
    return image if experiment == "D03" else np.concatenate(
        (artifact["clinical_scaler"].transform(_clinical(cohort, rows)), image), axis=1)


def predict_baseline(artifact, cohort, indices):
    """Predict supplied rows from a trusted fitted artifact; reads only baseline inputs."""
    if artifact.get("format") != "gastric-research-baseline-joblib-v1":
        raise ValueError("Unknown baseline inference artifact")
    rows = _indices(indices, len(cohort))
    x = _features(artifact, cohort, rows)
    if artifact["classifier"] is None:
        return np.full(len(rows), artifact["training_prevalence"], dtype=np.float64)
    return artifact["classifier"].predict_proba(x)[:, 1]


def _private_file(path, writer):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        writer(stream)


def fit_baseline(experiment_id, cohort, train_indices, validation_indices, out_dir, seed=17):
    """Fit one predeclared reference and score only supplied train/validation rows.

    ``metrics.json`` is aggregate only. The inference joblib and patient-linked
    prediction npz are private local files (0600); never load untrusted joblib.
    This function neither accepts a test role nor selects hyperparameters.
    """
    if experiment_id not in EXPERIMENTS:
        raise ValueError("Expected one of D00--D04")
    rows = {"train": _indices(train_indices, len(cohort)),
            "validation": _indices(validation_indices, len(cohort))}
    if set(rows["train"].tolist()) & set(rows["validation"].tolist()):
        raise ValueError("Train and validation memberships must be disjoint")
    train_ids = [cohort.ids[index] for index in rows["train"].tolist()]
    fit_ids = cohort.encoders.get("fit_ids", [])
    if len(fit_ids) != len(train_ids) or set(fit_ids) != set(train_ids):
        raise ValueError("Clinical encoder must fit exactly the supplied training membership")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use a new empty baseline output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {role: cohort.tensors["binary"][index].detach().cpu().numpy() for role, index in rows.items()}
    valid = {role: cohort.tensors["binary_valid"][index].detach().cpu().numpy().astype(bool)
             for role, index in rows.items()}
    training_labels = labels["train"][valid["train"]]
    if not len(training_labels) or not np.isin(training_labels, [0, 1]).all():
        raise ValueError("Require at least one observed binary training endpoint")
    for role in rows:
        if not np.isin(labels[role][valid[role]], [0, 1]).all():
            raise ValueError("Observed recurrence labels must be binary")

    artifact = {"format": "gastric-research-baseline-joblib-v1", "experiment_id": experiment_id,
                "seed": int(seed), "classifier": None,
                "training_prevalence": float(training_labels.mean()),
                "clinical_encoder": cohort.encoders.get("clinical", {}),
                "fit_patients_sha256": fingerprint(sorted(train_ids)),
                "sklearn_version": sklearn.__version__}
    preprocessing = {"fit_scope": "supplied training patients only", "ct1_read": False,
                     "clinical32": "frozen train-only six-field encoding"}
    if experiment_id in ("D01", "D04"):
        # Exactly match ClinicalAnchor.fit: fit all 32 columns on training rows
        # with observed recurrence. This includes fixed one-hot and mask columns.
        artifact["clinical_scaler"] = StandardScaler().fit(_clinical(cohort, rows["train"])[valid["train"]])
        preprocessing["clinical32"] = (
            "frozen train-only six-field encoding -> StandardScaler of all 32 columns "
            "fitted on observed-recurrence training rows; matches historical ClinicalAnchor")
        preprocessing["clinical_scaler_fit_patients"] = int(valid["train"].sum())
    if experiment_id == "D02":
        artifact["gam"] = NamedClinicalGAM(artifact["clinical_encoder"]).fit(_clinical(cohort, rows["train"]))
        preprocessing["gam"] = artifact["gam"].description()
    if experiment_id in ("D03", "D04"):
        moments = _ct0_moments(cohort, rows["train"])
        if min(moments.shape[0] - 1, moments.shape[1]) < 16:
            raise ValueError("The fixed PCA16 reference requires at least 17 training patients and 16 features")
        artifact["ct_moment_scaler"] = StandardScaler().fit(moments)
        standardized = artifact["ct_moment_scaler"].transform(moments)
        artifact["ct_pca"] = PCA(n_components=16, svd_solver="full").fit(standardized)
        artifact["ct_score_scaler"] = StandardScaler().fit(artifact["ct_pca"].transform(standardized))
        preprocessing["ct0"] = {"pool": "concatenate 27-token mean and population SD",
                                "order": "train StandardScaler -> full-SVD PCA16 -> train score StandardScaler",
                                "pca_components": 16,
                                "pca_explained_variance_ratio_sum": float(artifact["ct_pca"].explained_variance_ratio_.sum())}
    x = _features(artifact, cohort, rows["train"])
    regularization = {"C": 1. if experiment_id == "D01" else .1,
                      "objective": "summed binary NLL + (1/(2*C))*sum(coefficients**2); unpenalized intercept"}
    convergence = {"solver": None, "iterations": 0, "converged": True}
    if experiment_id != "D00" and len(np.unique(training_labels)) == 2:
        classifier = LogisticRegression(C=regularization["C"], solver="lbfgs", max_iter=2000,
                                        tol=1e-8, random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x[valid["train"]], training_labels)
        convergence = {"solver": "lbfgs", "iterations": int(classifier.n_iter_.max()),
                       "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught)}
        artifact["classifier"] = classifier
    elif experiment_id != "D00":
        convergence["fallback"] = "training endpoint contains a single class; constant prevalence"
    metrics = {"experiment_id": experiment_id, "seed": int(seed), "status": "completed",
               "endpoint": "recorded_terminal_binary_recurrence_not_fixed_horizon",
               "fit_patients": len(train_ids), "fit_endpoint_patients": int(valid["train"].sum()),
               "fit_patients_sha256": artifact["fit_patients_sha256"],
               "preprocessing": preprocessing,
               "regularization": regularization if experiment_id != "D00" else None,
               "convergence": convergence, "test_scored": False,
               "selection": "fixed configuration; validation not used to fit or choose parameters",
               "trainable_dynamics": False, "concept_model_trained": False,
               "feature_count": int(x.shape[1]), "training_prevalence": artifact["training_prevalence"],
               "artifact": "inference.joblib", "artifact_format": artifact["format"]}
    private_predictions = {}
    for role, index in rows.items():
        probabilities = predict_baseline(artifact, cohort, index)
        metrics[role] = binary_metrics(labels[role][valid[role]], probabilities[valid[role]])
        private_predictions.update({f"{role}_ids": np.asarray([cohort.ids[i] for i in index.tolist()]),
                                    f"{role}_labels": labels[role], f"{role}_valid": valid[role],
                                    f"{role}_probabilities": probabilities})
    if experiment_id == "D00":
        metrics["subtasks"] = {"recurrence_prevalence": "completed",
                               "C0_C1_means": "blocked: measured radiology labels required",
                               "identity": "blocked: matching frozen E and radiology labels required",
                               "true_mean_change": "blocked: paired radiology labels and matching E required",
                               "calibrated_persistence": "blocked: paired radiology labels and matching E required"}
        metrics["status"] = "partially_completed"
    artifact["preprocessing"] = preprocessing
    artifact["regularization"] = metrics["regularization"]
    _private_file(out_dir / "inference.joblib", lambda stream: joblib.dump(artifact, stream))
    _private_file(out_dir / "predictions.private.npz", lambda stream: np.savez_compressed(stream, **private_predictions))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return metrics
