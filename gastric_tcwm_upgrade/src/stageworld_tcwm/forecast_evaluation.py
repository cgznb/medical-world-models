"""Leakage-restricted prior forecasting scores in a frozen training target space."""
import math

import torch

from .data import fingerprint
from .diagnostic_baselines import FixedFeatures, legal_images, tensor_hash
from .losses import feature_set_loss
from .monte_carlo import case_key_epsilon


PRIOR_INPUTS = frozenset({"ct0", "clinical", "treatment", "interval_days", "surgery", "image_valid", "entry"})


@torch.inference_mode()
def prior_feature_draws(model, batch, case_keys, samples=32, seed=17):
    """CT1, postoperative observations and labels are absent from the forward map."""
    inputs = {key: value for key, value in batch.items() if key in PRIOR_INPUTS}
    inputs["image_valid"] = inputs["image_valid"].clone()
    inputs["image_valid"][:, 1] = False
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    epsilon = case_key_epsilon(case_keys, samples, model.cfg.latent_dim, seed, device=device)
    output = model(inputs, samples=samples, seed=seed, epsilon=epsilon,
                   max_stage=0, compute_aux=False, return_diagnostics=True)
    state = output["diagnostics"]["injected"]
    baseline = torch.where(inputs["image_valid"][:, 0, None, None], inputs["ct0"], 0.)
    return baseline[:, None] + model.decoder(state) * model.image_scale


class ForecastSpace:
    """Fixed shared PCA with train-CT1 moment standardization for all forecasters."""

    def __init__(self, training, fit_ids, rank, ridge_lambda=.1):
        if ridge_lambda <= 0 or not math.isfinite(ridge_lambda):
            raise ValueError("Ridge penalty must be finite and positive")
        self.features = FixedFeatures(rank).fit(training, fit_ids)
        self.paired = legal_images(training).all(1)
        if int(self.paired.sum()) < 2:
            raise ValueError("Forecast statistics require at least two legal paired training observations")
        targets = self.features.encode_image(training["ct1"][self.paired])
        self.mean = targets.mean(0)
        self.scale = targets.std(0, unbiased=False).clamp_min(.05)
        targets = (targets - self.mean) / self.scale
        self.target_mean = targets.mean(0)
        centered = targets - self.target_mean
        self.covariance = centered.T @ centered / len(targets)
        self.covariance += torch.eye(targets.shape[1], dtype=torch.float64) * 1e-6
        self.cholesky = torch.linalg.cholesky(self.covariance)
        predictors = self.features.transform(training)["B3"][self.paired]
        self.predictor_mean = predictors.mean(0)
        predictors = predictors - self.predictor_mean
        gram = predictors.T @ predictors / len(predictors)
        gram += ridge_lambda * torch.eye(predictors.shape[1], dtype=torch.float64)
        self.ridge_weight = torch.linalg.solve(gram, predictors.T @ centered / len(predictors))
        self.training_raw_mean_tokens = training["ct1"][self.paired].double().mean((0, 1))[None, None]
        self.provenance = {
            **self.features.provenance,
            "projection_space_id": self.features.target_space_id,
            "target_normalization": "train_legal_paired_CT1_aggregated_mean_and_population_sd_floor_0.05",
            "target_moment_statistics_sha256": tensor_hash(self.mean, self.scale),
            "paired_training_patients": int(self.paired.sum()),
            "paired_fit_patients_hash": fingerprint(sorted(patient for patient, valid in zip(fit_ids, self.paired.tolist()) if valid)),
            "ridge_lambda": ridge_lambda,
            "ridge_objective": "mean_patient_squared_error_per_target_channel+lambda*sum(weight_squared); free_intercept",
            "ridge_selection": "preset_0.1_no_outer_or_original_holdout_tuning",
            "gaussian_covariance": "training_CT1_population_full_covariance_plus_1e-6_identity",
        }
        self.provenance["target_space_id"] = fingerprint({
            "projection_space_id": self.features.target_space_id,
            "normalization": self.provenance["target_normalization"],
            "statistics": self.provenance["target_moment_statistics_sha256"],
        })

    def encode(self, tokens):
        return (self.features.encode_image(tokens) - self.mean) / self.scale

    def encode_draws(self, tokens):
        b, samples, length, channels = tokens.shape
        return self.encode(tokens.reshape(b * samples, length, channels)).reshape(b, samples, -1)

    def ridge(self, batch):
        features = self.features.transform(batch)["B3"]
        return (features - self.predictor_mean) @ self.ridge_weight + self.target_mean

    def gaussian(self, case_keys, samples, seed):
        epsilon = case_key_epsilon(case_keys, samples, len(self.mean), seed, dtype=torch.float64)
        return epsilon @ self.cholesky.T + self.target_mean


def distribution_scores(draws, target):
    if draws.ndim != 3 or draws.shape[0] != len(target) or draws.shape[2:] != target.shape[1:] or draws.shape[1] < 2:
        raise ValueError("Distribution scoring requires [patient,K>=2,channel] draws and matching targets")
    draws, target = draws.double().cpu(), target.double().cpu()
    samples = draws.shape[1]
    distances = torch.cdist(draws, draws)
    diagonal = torch.arange(samples)
    distances[:, diagonal, diagonal] = 0
    energy = (draws - target[:, None]).norm(dim=-1).mean(1) - distances.sum((1, 2)) / (2 * samples * (samples - 1))
    lower, upper = torch.quantile(draws, torch.tensor([.05, .95], dtype=torch.float64), dim=1)
    return {"energy_score": energy,
            "marginal_90pct_coverage": ((target >= lower) & (target <= upper)).double().mean(1),
            "marginal_90pct_width": (upper - lower).mean(1),
            "mean_projected_variance": draws.var(1, unbiased=True).mean(1)}


def point_scores(points, target, training_mean):
    baseline_sse = (target - training_mean).square().sum()
    return {name: {"patients": len(target), "elements": target.numel(),
                   "sse": float((value - target).square().sum()),
                   "training_mean_sse": float(baseline_sse)} for name, value in points.items()}


def merge_point_scores(parts):
    total = {key: sum(part[key] for part in parts) for key in ("patients", "elements", "sse", "training_mean_sse")}
    return {**total, "mse": total["sse"] / total["elements"],
            "skill_vs_training_CT1_mean": 1 - total["sse"] / total["training_mean_sse"] if total["training_mean_sse"] else None}


@torch.inference_mode()
def score_feature_sets(draws, target, copy_ct0, training_mean, scale):
    """Set-based scores never compare CT0/CT1 tokens at matching indices."""
    draws, target, copy_ct0, scale = [value.detach().cpu().float() for value in (draws, target, copy_ct0, scale)]
    valid = torch.ones(len(target), dtype=torch.bool)
    b, samples, length, channels = draws.shape
    mean_tokens = training_mean.float().expand(b, length, channels)
    first_loss = feature_set_loss(draws[:, 0], target, valid, scale)
    repeated = target[:, None].expand(-1, samples, -1, -1).reshape(b * samples, length, channels)
    all_loss = feature_set_loss(draws.flatten(0, 1), repeated, torch.ones(b * samples, dtype=torch.bool), scale)
    generated_means = draws.mean(2) / scale
    generated_sd = draws.std(2, unbiased=False) / scale
    target_means = target.mean(1) / scale
    target_sd = target.std(1, unbiased=False) / scale
    return {
        "prior_single_sample_set_loss": float(first_loss),
        "prior_expected_sample_set_loss": float(all_loss),
        "copy_CT0_set_loss": float(feature_set_loss(copy_ct0, target, valid, scale)),
        "training_channel_mean_degenerate_set_loss": float(feature_set_loss(mean_tokens, target, valid, scale)),
        "prior_MC_channel_mean_mse": float((generated_means.mean(1) - target_means).square().mean()),
        "prior_MC_channel_sd_mse": float((generated_sd.mean(1) - target_sd).square().mean()),
        "prior_single_channel_mean_mse": float((generated_means[:, 0] - target_means).square().mean()),
        "prior_single_channel_sd_mse": float((generated_sd[:, 0] - target_sd).square().mean()),
    }
