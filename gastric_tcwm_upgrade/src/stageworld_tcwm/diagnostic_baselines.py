"""Deterministic, train-only fixed-feature recurrence diagnostics.

The treatment descriptors and interval are an explicitly declared retrospective
scenario. B4 is an observed-CT1 diagnostic, not a baseline deployment model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

from .data import fingerprint
from .evaluation import binary_metrics


LAMBDAS = (1e-3, 1e-2, 1e-1, 1.)
INFORMATION = {
    "B0": "inner_training_recorded_recurrence_prevalence",
    "B1": "clinical32",
    "B2": "clinical32+explicit_treatment_scenario+query_interval",
    "B3": "clinical32+explicit_treatment_scenario+query_interval+CT0",
    "B4": "clinical32+explicit_treatment_scenario+query_interval+CT0+legal_observed_CT1",
}
PAIRS = (("B2", "B1"), ("B3", "B2"), ("B4", "B3"))


def tensor_hash(*values):
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def endpoint_mask(batch):
    # Identical endpoint eligibility across all five information sets.
    return batch["binary_valid"] & batch["prefix_valid"][:, :2].all(1)


def legal_images(batch):
    valid = batch["image_valid"].clone()
    valid[:, 1] &= batch["ct1_available_stage"] <= 1
    return valid


class FixedFeatures:
    """One shared PCA for CT0/CT1, followed by token mean/population SD."""

    def __init__(self, rank):
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("PCA rank must be a positive integer")
        self.rank = rank
        self.fitted = False

    def fit(self, training, fit_ids, encoder_provenance=None):
        if len(fit_ids) != len(training["clinical"]) or len(fit_ids) != len(set(fit_ids)):
            raise ValueError("Feature statistics require exactly the fitting patients")
        valid = legal_images(training)
        scans = torch.cat([training[name][valid[:, index]].double().cpu()
                           for index, name in enumerate(("ct0", "ct1"))])
        if len(scans) < 2 or self.rank > min(scans.shape[-1], len(scans) - 1):
            raise ValueError("Insufficient observed training scans for requested PCA rank")
        means = scans.mean(1)
        self.image_mean = means.mean(0)
        self.image_scale = means.std(0, unbiased=False).clamp_min(.05)
        standardized = (means - self.image_mean) / self.image_scale
        covariance = standardized.T @ standardized / len(means)
        _, eigenvectors = torch.linalg.eigh(covariance)
        self.basis = eigenvectors[:, -self.rank:].flip(1)
        pivots = self.basis.abs().argmax(0)
        signs = self.basis[pivots, torch.arange(self.rank)].sign()
        self.basis *= signs
        self.fit_patients_hash = fingerprint(sorted(fit_ids))
        self.fitted = True
        raw = self.raw_features(training)
        self.means, self.scales = {}, {}
        for name, values in raw.items():
            self.means[name] = values.mean(0)
            scale = values.std(0, unbiased=False)
            self.scales[name] = torch.where(scale > 1e-12, scale, 1.)
        self.target_space_id = fingerprint({
            "contract": "shared_training_scan_mean_PCA_then_token_mean_population_sd_v1",
            "rank": self.rank, "fit_patients_hash": self.fit_patients_hash,
            "projection_sha256": tensor_hash(self.image_mean, self.image_scale, self.basis),
        })
        self.provenance = {
            "rank": self.rank, "target_space_id": self.target_space_id,
            "fit_patients": len(fit_ids), "fit_patients_hash": self.fit_patients_hash,
            "projection_observations": len(scans),
            "projection_fit": "shared_CT0_and_legal_CT1_patient_scan_channel_means",
            "feature_level": "patient_aggregated_27_cached_tokens",
            "normalization": "train_scan_mean_channel_center_and_population_sd_floor_0.05; "
                             "shared_PCA; token_mean_and_population_sd; "
                             "train_patient_feature_center_and_population_sd",
            "missing_images": "zero_imputation_in_raw_projected_space_plus_observation_indicator",
            "encoder_provenance": encoder_provenance or {},
            "tabular_statistics_sha256": tensor_hash(self.means["B2"], self.scales["B2"]),
            "different_ranks_define_different_target_spaces": True,
        }
        return self

    def encode_image(self, image):
        if not self.fitted:
            raise RuntimeError("Fit features on training patients before transform")
        tokens = ((image.detach().double().cpu() - self.image_mean) / self.image_scale) @ self.basis
        return torch.cat((tokens.mean(1), tokens.std(1, unbiased=False)), 1)

    def raw_features(self, batch):
        clinical = batch["clinical"].detach().double().cpu()
        condition = torch.cat((clinical, batch["treatment"].detach().double().cpu().flatten(1),
                               torch.log1p(batch["interval_days"].double().cpu()[:, None] / 30)), 1)
        images = legal_images(batch).cpu()
        observed = []
        for index, name in enumerate(("ct0", "ct1")):
            encoded = self.encode_image(batch[name])
            observed.append(torch.cat((torch.where(images[:, index, None], encoded, 0.),
                                       images[:, index, None].double()), 1))
        return {"B1": clinical, "B2": condition,
                "B3": torch.cat((condition, observed[0]), 1),
                "B4": torch.cat((condition, *observed), 1)}

    def transform(self, batch):
        return {name: (values - self.means[name]) / self.scales[name]
                for name, values in self.raw_features(batch).items()}

    def state_dict(self):
        return {"rank": self.rank, "image_mean": self.image_mean,
                "image_scale": self.image_scale, "basis": self.basis,
                "means": self.means, "scales": self.scales, "provenance": self.provenance}


def logistic_objective(theta, features, labels, penalty):
    logits = features @ theta[:-1] + theta[-1]
    return F.binary_cross_entropy_with_logits(logits, labels) + penalty / 2 * theta[:-1].square().sum()


@dataclass
class LogisticFit:
    parameters: torch.Tensor
    report: dict

    def logits(self, features):
        return features.double().cpu() @ self.parameters[:-1] + self.parameters[-1]


def fit_logistic(features, labels, penalty, max_iter=2000):
    """FP64 full-batch L-BFGS: mean BCE + lambda/2 ||w||^2, free intercept."""
    x, y = features.detach().double().cpu(), labels.detach().double().cpu()
    if (x.ndim != 2 or y.shape != (len(x),) or not len(x) or penalty <= 0
            or max_iter < 1 or max_iter > 2000 or not math.isfinite(penalty)):
        raise ValueError("Invalid logistic fitting arguments")
    if not torch.isfinite(x).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("Logistic features and binary labels must be finite")
    if len(torch.unique(y)) != 2:
        raise ValueError("Finite unpenalized-intercept logistic fit requires both classes")
    parameters = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        parameters[-1] = torch.logit(y.mean())
    optimizer = torch.optim.LBFGS([parameters], max_iter=max_iter, max_eval=max_iter * 2,
                                  tolerance_grad=1e-9, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe", history_size=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = logistic_objective(parameters, x, y, penalty)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure()
    gradient = parameters.grad.detach()
    state = optimizer.state[parameters]
    report = {
        "objective": "mean_BCE+lambda/2*sum(weight_squared); intercept_unpenalized",
        "dtype": "float64", "solver": "torch_full_batch_LBFGS_strong_wolfe",
        "lambda": penalty, "max_iter": max_iter,
        "iterations": int(state["n_iter"]), "function_evaluations": int(state["func_evals"]),
        "objective_value": float(loss.detach()),
        "gradient_l2_norm": float(gradient.norm()), "gradient_max_abs": float(gradient.abs().max()),
        "converged": bool(torch.isfinite(loss) and gradient.abs().max() <= 1e-6),
        "convergence_definition": "finite_objective_and_gradient_max_abs_at_most_1e-6",
        "training_patients": len(y), "training_events": int(y.sum()),
        "features": x.shape[1],
    }
    return LogisticFit(parameters.detach().clone(), report)


def point_losses(labels, probability):
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {"nll": -(y * np.log(p) + (1 - y) * np.log1p(-p)), "brier": (p - y) ** 2}


def metrics(labels, probability):
    y, p = np.asarray(labels), np.asarray(probability)
    result = binary_metrics(y, p)
    bins = []
    for lower in np.linspace(0, .9, 10):
        upper = lower + .1
        selected = (p >= lower) & ((p < upper) if upper < .999 else (p <= 1))
        if selected.any():
            bins.append({"lower": float(lower), "upper": float(upper), "n": int(selected.sum()),
                         "predicted": float(p[selected].mean()), "observed": float(y[selected].mean())})
    result["calibration"] = {"fixed_probability_bins": bins,
        "mean_prediction": float(p.mean()), "observed_frequency": float(y.mean()),
        "mean_prediction_minus_observed": float(p.mean() - y.mean()),
        "expected_calibration_error": float(sum(b["n"] * abs(b["predicted"] - b["observed"])
                                                  for b in bins) / len(y))}
    return result


def paired_comparison(labels, newer, reference, folds, bootstrap=1000, seed=17):
    """Patient paired bootstrap stratified by outer fold, conditional on fits."""
    y, newer, reference, folds = [np.asarray(value) for value in (labels, newer, reference, folds)]
    if len(y) == 0 or any(value.shape != y.shape for value in (newer, reference, folds)):
        raise ValueError("Paired comparisons require identical nonempty patient axes")
    keys = ("nll", "brier", "auroc", "average_precision")

    def differences(indices):
        first = binary_metrics(y[indices], newer[indices])
        second = binary_metrics(y[indices], reference[indices])
        return {key: first[key] - second[key] if first[key] is not None and second[key] is not None else None
                for key in keys}

    actual = differences(np.arange(len(y)))
    by_fold = []
    groups = [np.flatnonzero(folds == fold) for fold in np.unique(folds)]
    for fold, rows in zip(np.unique(folds), groups):
        by_fold.append({"fold": int(fold), "patients": len(rows), "delta_new_minus_reference": differences(rows)})
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in keys}
    for _ in range(bootstrap):
        rows = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        for key, value in differences(rows).items():
            if value is not None:
                draws[key].append(value)
    intervals = {key: {"low": float(np.quantile(value, .025)), "high": float(np.quantile(value, .975)),
                       "valid_draws": len(value)} if value else None for key, value in draws.items()}
    directions = {key: sum(record["delta_new_minus_reference"][key] is not None and
                           record["delta_new_minus_reference"][key] * (-1 if key in ("nll", "brier") else 1) > 0
                           for record in by_fold) for key in keys}
    return {"delta_new_minus_reference": actual, "folds": by_fold,
            "folds_improved": directions, "bootstrap_95_percent": intervals,
            "bootstrap_draws": bootstrap, "bootstrap_seed": seed,
            "uncertainty_scope": "fixed_predictions_patient_paired_within_fold_bootstrap; "
                                 "excludes_retraining_and_candidate_selection_uncertainty"}


def fit_fold(fold, rank, max_iter=2000):
    cohort, rows = fold["cohort"], fold["rows"]
    batch = {role: cohort.batch(rows[role]) for role in ("train", "validation", "outer_evaluation")}
    masks = {role: endpoint_mask(values) for role, values in batch.items()}
    if any(not bool(mask.any()) for mask in masks.values()):
        raise ValueError("Each fitting, selection and outer partition needs eligible outcomes")
    training_ids = fold["split"]["train"]
    if (len(cohort.encoders.get("fit_ids", [])) != len(training_ids)
            or set(cohort.encoders.get("fit_ids", [])) != set(training_ids)):
        raise ValueError("Raw clinical/treatment encoders must fit exactly the inner training patients")
    features = FixedFeatures(rank).fit(batch["train"], training_ids, {
        "clinical_transform_sha256": fingerprint(cohort.encoders.get("clinical", {})),
        "treatment_support_sha256": fingerprint(cohort.encoders.get("treatment_support", {})),
        "fit_ids_exactly_match_inner_train": True,
    })
    encoded = {role: {name: value[masks[role]] for name, value in features.transform(values).items()}
               for role, values in batch.items()}
    targets = {role: values["binary"][masks[role]].double() for role, values in batch.items()}
    prevalence = float(targets["train"].mean())
    selected_models, records, probabilities = {}, {}, {}
    for name in INFORMATION:
        if name == "B0":
            selection = np.full(len(targets["validation"]), prevalence)
            outer = np.full(len(targets["outer_evaluation"]), prevalence)
            report = {"lambda": None, "iterations": 0, "converged": True,
                      "training_prevalence": prevalence, "gradient_l2_norm": None,
                      "gradient_max_abs": None, "solver": "training_prevalence_constant"}
            candidates = []
        else:
            fitted = []
            for penalty in LAMBDAS:
                classifier = fit_logistic(encoded["train"][name], targets["train"], penalty, max_iter)
                prediction = classifier.logits(encoded["validation"][name]).sigmoid().numpy()
                # Select on unpenalized patient NLL only; tied fits favor stronger shrinkage.
                score = float(F.binary_cross_entropy_with_logits(
                    classifier.logits(encoded["validation"][name]), targets["validation"]))
                fitted.append((score, -penalty, classifier, prediction))
            eligible = [candidate for candidate in fitted if candidate[2].report["converged"]]
            if not eligible:
                raise RuntimeError(f"No converged candidate for fold {fold['index']}, rank {rank}, {name}")
            _, _, classifier, selection = min(eligible, key=lambda candidate: candidate[:2])
            outer = classifier.logits(encoded["outer_evaluation"][name]).sigmoid().numpy()
            report = classifier.report
            candidates = [{**candidate[2].report, "selection_nll": candidate[0]} for candidate in fitted]
            selected_models[name] = classifier.parameters
        records[name] = {"information_set": INFORMATION[name], "selected_fit": report,
                         "lambda_candidates": candidates,
                         "selection": metrics(targets["validation"].numpy(), selection),
                         "outer": metrics(targets["outer_evaluation"].numpy(), outer)}
        probabilities[name] = outer
    outer_ids = [patient for patient, valid in zip(cohort.metadata["outer_evaluation_ids"],
                                                  masks["outer_evaluation"].tolist()) if valid]
    record = {"fold": fold["index"], "rank": rank, "target_space_id": features.target_space_id,
              "provenance": features.provenance, "source_hashes": fold["hashes"],
              "training_patients": len(targets["train"]), "training_events": int(targets["train"].sum()),
              "selection_patients": len(targets["validation"]),
              "outer_patients": len(outer_ids), "outer_events": int(targets["outer_evaluation"].sum()),
              "stage_weights": None, "stage_weights_reason": "one_patient_one_recorded_recurrence_target",
              "seed": 17, "mc_samples": 0, "baseline0_selected": None,
              "supervised_optimizer_updates": {name: value["selected_fit"]["iterations"] for name, value in records.items()},
              "selected_steps": {name: value["selected_fit"]["iterations"] for name, value in records.items()},
              "models": records}
    private = {"patient_ids": outer_ids, "labels": targets["outer_evaluation"].numpy(),
               "probabilities": probabilities, "fold": np.full(len(outer_ids), fold["index"])}
    state = {"features": features.state_dict(), "classifiers": selected_models,
             "training_prevalence": prevalence, "record": record}
    return record, private, state
