"""Small, terminal-only research controls on the fixed 651-person split.

These are predictive baselines, not validated concept or treatment-effect models.
Only explicitly listed CT0/clinical/modality inputs reach the forward graph.
"""
from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .data import atomic_save, fingerprint, write_json
from .modality_schema import PHASES, OPERATIONS, status_ids


EXPERIMENTS = ("N00", "N01", "N02", "N13")
EXPORT_SCHEMA = "gastric-research-neural-v1"
INPUT_FIELDS = ("clinical", "ct0", "image_valid", "modality_value", "modality_known",
                "modality_applicable", "event_mask", "phase", "operation", "role",
                "event_order", "event_id", "query_mask", "query_order")
DEFAULTS = dict(hidden_dim=16, dropout=.1, learning_rate=.001, weight_decay=.05,
                batch_size=16, gradient_clip=1., validation_interval=10,
                minimum_steps=100, patience=8, min_delta=1e-5, max_steps=300)


def terminal_eligible(batch):
    """Factual complete S3 only; no endpoint supervision on invented paths."""
    mask = batch["event_mask"].bool()
    if mask.shape[1] != 3:
        raise ValueError("Research controls require exactly three factual stage slots")
    expected = torch.tensor([1, 2, 3], device=mask.device)
    ordered = (batch["phase"] == expected).all(1) & (batch["event_order"] == expected).all(1)
    ids = batch["event_id"]
    unique = (ids[:, 0] != ids[:, 1]) & (ids[:, 0] != ids[:, 2]) & (ids[:, 1] != ids[:, 2])
    factual = ((batch["role"] == 0) | (batch["role"] == 2)).all(1)
    query = (batch["query_mask"] & (batch["query_order"] == 3)).any(1)
    return mask.all(1) & ordered & unique & factual & query


class ResearchNeural(nn.Module):
    """N00 clinical MLP; N01/N02 static MIL; N13 small event GRU."""

    def __init__(self, experiment_id, image_dim=768, hidden_dim=16, dropout=.1):
        super().__init__()
        if experiment_id not in EXPERIMENTS:
            raise ValueError(f"Unknown research neural experiment: {experiment_id}")
        self.experiment_id = experiment_id
        self.config = dict(experiment_id=experiment_id, image_dim=image_dim,
                           hidden_dim=hidden_dim, dropout=dropout)
        self.register_buffer("image_mean", torch.zeros(image_dim))
        self.register_buffer("image_scale", torch.ones(image_dim))
        self.dropout = nn.Dropout(dropout)
        if experiment_id == "N00":
            self.clinical_encoder = nn.Linear(32, hidden_dim)
        else:
            self.image_encoder = nn.Linear(image_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim, 1, bias=False)
        if experiment_id == "N02":
            self.clinical_linear = nn.Linear(32, 1)
        if experiment_id == "N13":
            self.state_initializer = nn.Linear(hidden_dim + 32, hidden_dim)
            # Four distinct states: inapplicable, unknown, absent, present.
            self.event_cell = nn.GRUCell(7 * 4 + len(PHASES) + len(OPERATIONS), hidden_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    @torch.no_grad()
    def fit_initial_state(self, train):
        """Fit only train statistics and a valid, explicitly named step0 head."""
        valid = train["binary_valid"].bool() & terminal_eligible(train)
        if not valid.any():
            raise ValueError("No labelled factual terminal training patients")
        if self.experiment_id != "N00":
            observed = train["image_valid"][:, 0].bool()
            if not observed.any():
                raise ValueError("Image controls require observed training CT0")
            tokens = train["ct0"][observed].float()
            self.image_mean.copy_(tokens.mean((0, 1)))
            self.image_scale.copy_(tokens.std((0, 1), unbiased=False).clamp_min(1e-3))
        y = train["binary"][valid].double().cpu().numpy()
        prevalence = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        prevalence_logit = math.log(prevalence / (1 - prevalence))
        self.risk_head.weight.zero_()
        self.risk_head.bias.fill_(prevalence_logit)
        record = {"kind": "training_prevalence", "patients": int(valid.sum()),
                  "prevalence": prevalence, "clinical_secondary_standardization": False}
        if self.experiment_id == "N02":
            self.risk_head.bias.zero_()
            self.clinical_linear.weight.zero_()
            self.clinical_linear.bias.fill_(prevalence_logit)
            if len(np.unique(y)) == 2:
                clinical = train["clinical"][valid].double().cpu().numpy()
                scaler = StandardScaler().fit(clinical)
                fitted = LogisticRegression(C=1., solver="lbfgs", max_iter=2000,
                                            tol=1e-8, random_state=17)
                fitted.fit(scaler.transform(clinical), y)
                # Match the established D01/ClinicalAnchor exactly, folding its
                # train-only standardization into the explicit raw-input branch.
                raw_weight = fitted.coef_ / scaler.scale_[None, :]
                raw_bias = fitted.intercept_ - (raw_weight * scaler.mean_[None, :]).sum(1)
                self.clinical_linear.weight.copy_(torch.from_numpy(raw_weight))
                self.clinical_linear.bias.copy_(torch.from_numpy(raw_bias))
                if int(fitted.n_iter_[0]) >= 2000:
                    raise RuntimeError("Initial train-only clinical logistic did not converge")
                record.update(kind="train_only_clinical_logistic_C1", iterations=int(fitted.n_iter_[0]),
                              clinical_secondary_standardization=True,
                              standardization_policy="train StandardScaler(all32) folded into raw-input linear; historical D01",
                              fitted_mean=scaler.mean_.tolist(), fitted_scale=scaler.scale_.tolist())
        return record

    def image_state(self, batch):
        observed = batch["image_valid"][:, 0].bool()
        tokens = (batch["ct0"].float() - self.image_mean) / self.image_scale
        tokens = torch.where(observed[:, None, None], tokens, torch.zeros_like(tokens))
        hidden = torch.tanh(self.image_encoder(tokens))
        weights = self.attention(hidden).softmax(dim=1)
        pooled = (weights * hidden).sum(1)
        return torch.where(observed[:, None], pooled, torch.zeros_like(pooled))

    def event_states(self, batch):
        """Latent diagnostic states, without interpreting them as disease concepts."""
        if self.experiment_id != "N13":
            raise ValueError("Only N13 has a sequential event state")
        image = self.image_state(batch)
        state = F.gelu(self.state_initializer(torch.cat((image, batch["clinical"].float()), 1)))
        state = self.dropout(state)
        states = [state]
        status = status_ids(batch["modality_value"], batch["modality_known"],
                            batch["modality_applicable"])
        status = torch.where(batch["modality_applicable"], status + 1, 0)
        for index in range(batch["event_mask"].shape[1]):
            event = torch.cat((F.one_hot(status[:, index], 4).flatten(1),
                               F.one_hot(batch["phase"][:, index], len(PHASES)),
                               F.one_hot(batch["operation"][:, index], len(OPERATIONS))), 1).float()
            updated = self.event_cell(event, state)
            state = torch.where(batch["event_mask"][:, index, None], updated, state)
            states.append(state)
        return torch.stack(states, 1)

    def forward(self, batch):
        if self.experiment_id == "N00":
            hidden = F.gelu(self.clinical_encoder(batch["clinical"].float()))
        elif self.experiment_id == "N13":
            if not terminal_eligible(batch).all():
                raise ValueError("N13 only exposes complete factual S3 risk, not partial or hypothetical risk")
            hidden = self.event_states(batch)[:, -1]
        else:
            hidden = self.image_state(batch)
        logits = self.risk_head(self.dropout(hidden)).squeeze(-1)
        if self.experiment_id == "N02":
            logits = logits + self.clinical_linear(batch["clinical"].float()).squeeze(-1)
        return logits


def head_training_flags(experiment_id, step):
    return {"recurrence_head_optimized": step > 0,
            "clinical_logistic_fitted": experiment_id == "N02",
            "image_encoder_supervised_updates": max(0, step - 1) if experiment_id != "N00" else 0,
            "event_transition_supervised_updates": max(0, step - 1) if experiment_id == "N13" else 0,
            "concept_heads_exist": False, "pcr_head_exists": False,
            "ct1_forecast_head_exists": False, "validated_concept_dynamics": False,
            "treatment_strategy_interface_enabled": False,
            "intermediate_risk_interface_enabled": False}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _metrics(labels, logits):
    labels, logits = labels.double().cpu(), logits.double().cpu()
    probability = logits.sigmoid().numpy()
    y = labels.numpy()
    return {"n": len(y), "positive": int(labels.sum()),
            "nll": float(F.binary_cross_entropy_with_logits(logits, labels)),
            "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "ap": float(average_precision_score(y, probability)) if y.sum() else None,
            "brier": float(np.mean((probability - y) ** 2)),
            "mean_probability": float(probability.mean())}


def _subset(batch, rows):
    return {key: value[rows] for key, value in batch.items()}


@torch.no_grad()
def _evaluate(model, batch, chunk_size=64):
    model.eval()
    valid = batch["binary_valid"].bool() & terminal_eligible(batch)
    rows = valid.nonzero().flatten()
    if not len(rows):
        raise ValueError("No labelled factual terminal patients for evaluation")
    logits = torch.cat([model(_subset(batch, chunk)) for chunk in rows.split(chunk_size)])
    return _metrics(batch["binary"][rows], logits), {"rows": rows.cpu(), "logits": logits.cpu(),
            "labels": batch["binary"][rows].cpu(), "probability": logits.sigmoid().cpu()}


def load_research_neural_export(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("Unrecognized research neural export")
    model = ResearchNeural(**payload["config"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


def train_research_neural(experiment_id, cohort, train_indices, validation_indices,
                          out_dir, seed=17, device="cuda", max_steps=300, diagnostic=False):
    """Train/validation only. The caller owns fixed split/source locking.

    ``best.pt`` and ``last.pt`` include optimizer and RNG state for auditable
    continuation; ``inference.pt`` contains the selected model only. Short runs
    must explicitly opt into diagnostic mode and cannot become scientific runs.
    """
    if experiment_id not in EXPERIMENTS:
        raise ValueError("Unsupported new neural control")
    if max_steps < 1 or max_steps > 300 or (not diagnostic and max_steps != 300):
        raise ValueError("Formal controls use max_steps=300; shorter runs require diagnostic=True")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / name).exists() for name in ("best.pt", "last.pt", "inference.pt", "metrics.json")):
        raise FileExistsError("Refusing to overwrite an existing research neural run")
    train_rows, val_rows = (torch.as_tensor(rows, dtype=torch.long).cpu()
                            for rows in (train_indices, validation_indices))
    for rows in (train_rows, val_rows):
        if rows.ndim != 1 or not len(rows) or len(rows.unique()) != len(rows):
            raise ValueError("Require nonempty unique one-dimensional partition indices")
        if rows.min() < 0 or rows.max() >= len(cohort.ids):
            raise ValueError("Partition index outside cohort")
    if set(train_rows.tolist()) & set(val_rows.tolist()):
        raise ValueError("Training and validation patients overlap")
    fitted_ids = cohort.encoders.get("fit_ids", [])
    actual_ids = [cohort.ids[int(row)] for row in train_rows]
    if len(fitted_ids) != len(actual_ids) or set(fitted_ids) != set(actual_ids):
        raise ValueError("Cached clinical encoder must be fitted on exactly this training partition")
    if not diagnostic and (len(train_rows) != 456 or len(val_rows) != 65 or len(cohort.ids) != 651):
        raise ValueError("Formal research controls require the fixed 456/65/130 cohort")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    sampler = torch.Generator().manual_seed(seed)
    # Read only assigned train/validation rows and allowed fields; CT1, pCR,
    # pathology, outcome masks and held-out test rows never enter the model.
    fields = INPUT_FIELDS + ("binary", "binary_valid")
    batches = {role: {key: cohort.tensors[key][rows].to(device) for key in fields}
               for role, rows in (("train", train_rows), ("validation", val_rows))}
    for batch in batches.values():
        valid = batch["binary_valid"].bool()
        y = batch["binary"][valid]
        if not ((y == 0) | (y == 1)).all():
            raise ValueError("Valid recurrence outcomes must be binary")
        if not torch.isfinite(batch["clinical"]).all() or not torch.isfinite(batch["ct0"]).all():
            raise ValueError("Nonfinite allowed model inputs")
    model = ResearchNeural(experiment_id, image_dim=batches["train"]["ct0"].shape[-1]).to(device)
    baseline = model.fit_initial_state(batches["train"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULTS["learning_rate"],
                                 weight_decay=DEFAULTS["weight_decay"])
    param_count = sum(parameter.numel() for parameter in model.parameters())
    if param_count > 100000:
        raise ValueError("New research control exceeded frozen 100k parameter target")
    eligible = (batches["train"]["binary_valid"] & terminal_eligible(batches["train"])).nonzero().flatten()
    run = {"schema": EXPORT_SCHEMA, "experiment_id": experiment_id, "seed": seed,
           "diagnostic": diagnostic, "config": model.config, "hyperparameters": {**DEFAULTS, "max_steps": max_steps},
           "parameter_count": param_count, "trainable_parameter_count": param_count,
           "partition_counts": {"train": len(train_rows), "validation": len(val_rows)},
           "train_membership_sha256": fingerprint(sorted(actual_ids)),
           "validation_membership_sha256": fingerprint(sorted(cohort.ids[int(row)] for row in val_rows)),
           "clinical_preprocessing": "existing clinical32 fitted on exactly train; N02 initial D01 scaler folded into linear weights",
           "image_preprocessing": "CT0 training-patient/channel mean and population std; minimum scale .001",
           "initial_baseline": baseline, "test_evaluated": False,
           "information_boundary": "CT0/clinical; N13 additionally three retrospective factual modality events",
           "endpoint": "recorded binary recurrence at factual complete S3; no fixed follow-up horizon",
           "clinical_branch_policy": "N02 fitted train-only logistic, then jointly optimized explicit linear branch",
           "concept_measurements_used": False, "ct1_input_used": False,
           "historical_test_exposure": True}
    history, best_score, best_step, stale = [], float("inf"), 0, 0
    start = time.monotonic()
    permutation, cursor = torch.empty(0, dtype=torch.long), 0

    def checkpoint(step):
        return {**run, "step": step, "model": _cpu_tree(model.state_dict()),
                "optimizer": _cpu_tree(optimizer.state_dict()), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
                "sampler_rng": sampler.get_state(), "sampler_permutation": permutation.clone(),
                "sampler_cursor": cursor, "history": copy.deepcopy(history),
                "best_step": best_step, "best_validation_nll": best_score, "stale_checks": stale,
                "head_training_flags": head_training_flags(experiment_id, step)}

    last_metrics = None
    for step in range(max_steps + 1):
        train_loss = None
        gradient_norm = None
        if step:
            model.train()
            if cursor >= len(permutation):
                permutation = torch.randperm(len(eligible), generator=sampler)
                cursor = 0
            local = permutation[cursor:cursor + DEFAULTS["batch_size"]]
            cursor += len(local)
            batch = _subset(batches["train"], eligible[local.to(eligible.device)])
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), batch["binary"].float())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite recurrence loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            train_loss, gradient_norm = float(loss.detach()), float(norm)
        if step % DEFAULTS["validation_interval"] and step != max_steps:
            continue
        evaluated = {role: _evaluate(model, batch)[0] for role, batch in batches.items()}
        history.append({"step": step, **evaluated, "minibatch_loss": train_loss,
                        "gradient_norm_before_clip": gradient_norm,
                        "parameter_l2": math.sqrt(sum(float(p.detach().square().sum()) for p in model.parameters()))})
        improved = evaluated["validation"]["nll"] < best_score - DEFAULTS["min_delta"]
        if improved:
            best_score, best_step, stale = evaluated["validation"]["nll"], step, 0
            atomic_save(checkpoint(step), out_dir / "best.pt")
        else:
            stale += 1
        last_metrics = evaluated
        atomic_save(checkpoint(step), out_dir / "last.pt")
        write_json(history, out_dir / "history.json")
        write_json({"state": "running", "step": step, "best_step": best_step,
                    "best_validation_nll": best_score, "diagnostic": diagnostic,
                    "elapsed_seconds": time.monotonic() - start}, out_dir / "status.json")
        if step >= DEFAULTS["minimum_steps"] and stale >= DEFAULTS["patience"]:
            break

    selected, payload = load_research_neural_export(out_dir / "best.pt", device)
    evaluated, predictions = {}, {}
    for role, batch in batches.items():
        evaluated[role], predictions[role] = _evaluate(selected, batch)
    selected_kind = (baseline["kind"] if best_step == 0 else "trained_" + experiment_id)
    result = {**run, **evaluated, "initial": history[0], "last": last_metrics,
              "best_step": best_step, "completed_steps": step, "selected_kind": selected_kind,
              "step0_selected": best_step == 0, "head_training_flags": head_training_flags(experiment_id, best_step),
              "stop_reason": "early_stopping" if step < max_steps else "max_steps",
              "elapsed_seconds": time.monotonic() - start}
    inference = {key: payload[key] for key in ("schema", "config", "model", "step", "experiment_id", "seed")}
    inference.update(metrics=result, selected_kind=selected_kind,
                     head_training_flags=result["head_training_flags"])
    atomic_save(inference, out_dir / "inference.pt")
    atomic_save(predictions, out_dir / "predictions_private.pt")
    write_json(result, out_dir / "metrics.json")
    write_json({"state": "completed", "completed_steps": step, "best_step": best_step,
                "selected_kind": selected_kind, "diagnostic": diagnostic,
                "elapsed_seconds": result["elapsed_seconds"]}, out_dir / "status.json")
    return result
