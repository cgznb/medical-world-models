"""Patient-balanced query supervision and fixed, training-only CT targets."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def patient_query_bce(logits, labels, query_mask, label_mask):
    """Every labelled patient contributes equally, regardless of query count."""
    mask = query_mask.bool() & label_mask.bool()[:, None]
    valid = mask.any(1)
    safe_labels = torch.where(label_mask.bool(), labels, torch.zeros_like(labels))
    safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
    losses = F.binary_cross_entropy_with_logits(
        safe_logits, safe_labels[:, None].expand_as(logits).float(), reduction="none")
    per_patient = (losses * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_patient[valid].mean() if valid.any() else safe_logits.sum() * 0


class FixedCTMoments(nn.Module):
    """Fixed channel and set-moment coordinates fitted on training CT1 only."""

    ANCHOR_SEED = 17031

    def __init__(self, image_dim, anchor_dim=None, report_concepts=False):
        super().__init__()
        if not isinstance(report_concepts, bool):
            raise ValueError("Report concept targets must be explicitly enabled or disabled")
        if anchor_dim is not None and (isinstance(anchor_dim, bool)
                                       or not isinstance(anchor_dim, int) or anchor_dim < 1):
            raise ValueError("Anchor dimension must be a positive integer")
        self.anchor_dim = anchor_dim
        self.report_concepts = report_concepts
        self.register_buffer("image_mean", torch.zeros(image_dim))
        self.register_buffer("image_scale", torch.ones(image_dim))
        self.register_buffer("moment_mean", torch.zeros(2 * image_dim))
        self.register_buffer("moment_scale", torch.ones(2 * image_dim))
        self.register_buffer("fitted", torch.tensor(False))
        if anchor_dim is not None:
            generator = torch.Generator().manual_seed(self.ANCHOR_SEED)
            projection = torch.randn(2 * image_dim, anchor_dim, generator=generator)
            projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-6)
            self.register_buffer("anchor_projection", projection)
            self.register_buffer("anchor_mean", torch.zeros(anchor_dim))
            self.register_buffer("anchor_scale", torch.ones(anchor_dim))
        if report_concepts:
            self.register_buffer("report_mean", torch.zeros(4))
            self.register_buffer("report_scale", torch.ones(4))
            self.register_buffer("report_counts", torch.zeros(4, dtype=torch.long))
            self.register_buffer("report_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, tokens, valid):
        selected = tokens[valid.bool()].float()
        if not len(selected):
            raise ValueError("Forecast targets require observed CT1 in the training fold")
        self.image_mean.copy_(selected.mean((0, 1)))
        self.image_scale.copy_(selected.std((0, 1), unbiased=False).clamp_min(1e-3))
        moments = self.raw_moments(self.normalize(selected))
        self.moment_mean.copy_(moments.mean(0))
        self.moment_scale.copy_(moments.std(0, unbiased=False).clamp_min(0.05))
        if self.anchor_dim is not None:
            anchors = ((moments - self.moment_mean) / self.moment_scale) @ self.anchor_projection
            self.anchor_mean.copy_(anchors.mean(0))
            self.anchor_scale.copy_(anchors.std(0, unbiased=False).clamp_min(0.05))
        self.fitted.fill_(True)
        return self

    def normalize(self, tokens):
        return (tokens.float() - self.image_mean) / self.image_scale

    @staticmethod
    def raw_moments(tokens):
        return torch.cat((tokens.mean(-2), tokens.std(-2, unbiased=False)), -1)

    def forward(self, tokens):
        if not bool(self.fitted):
            raise ValueError("CT target statistics have not been fitted on training patients")
        return (self.raw_moments(self.normalize(tokens)) - self.moment_mean) / self.moment_scale

    @torch.no_grad()
    def anchor(self, tokens):
        """A fixed, detached target; no learned encoder can collapse its coordinates."""
        if self.anchor_dim is None:
            raise ValueError("This target space has no S1 anchor projection")
        projected = self(tokens) @ self.anchor_projection
        return ((projected - self.anchor_mean) / self.anchor_scale).detach()

    @torch.no_grad()
    def fit_report_concepts(self, values, valid):
        if not self.report_concepts:
            raise ValueError("Report concept targets are disabled")
        if values.ndim != 2 or values.shape[1] != 4 or valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("Report targets require values and boolean validity with shape [N,4]")
        if not torch.isfinite(values[valid]).all():
            raise ValueError("Observed report concepts must be finite")
        if not ((values[:, 0][valid[:, 0]] == 0) | (values[:, 0][valid[:, 0]] == 1)).all():
            raise ValueError("Residual viable primary report target must be binary")
        self.report_counts.copy_(valid.sum(0))
        if not bool((self.report_counts > 0).all()):
            raise ValueError("Each enabled report target needs an observed training example")
        for column in range(4):
            selected = values[:, column][valid[:, column]].float()
            self.report_mean[column].copy_(selected.mean())
            if column:
                self.report_scale[column].copy_(selected.std(unbiased=False).clamp_min(.05))
        self.report_fitted.fill_(True)
        return self

    @torch.no_grad()
    def report_targets(self, values, valid):
        if not self.report_concepts or not bool(self.report_fitted):
            raise ValueError("Report targets have not been fitted on training patients")
        safe = torch.where(valid, values.float(), self.report_mean)
        normalized = (safe - self.report_mean) / self.report_scale
        normalized[:, 0] = torch.where(valid[:, 0], values[:, 0], 0.)
        return normalized.detach()

    def decode_report(self, predictions):
        if not self.report_concepts or not bool(self.report_fitted):
            raise ValueError("Report targets have not been fitted on training patients")
        result = predictions.float() * self.report_scale + self.report_mean
        return torch.cat((predictions[:, :1].sigmoid(), result[:, 1:]), dim=1)


def scan_valid(batch, output=None):
    valid = batch["image_valid"][:, 1].bool()
    if "scan_event_index" in batch:
        valid = valid & (batch["scan_event_index"] >= 0)
    if "scan_mask" in batch:
        valid = valid & batch["scan_mask"].bool()
    if output is not None and "forecast_mask" in output:
        valid = valid & output["forecast_mask"].bool()
    if "role" in batch and "scan_event_index" in batch:
        before_scan = torch.arange(batch["role"].shape[1], device=valid.device)[None] < batch["scan_event_index"][:, None]
        hypothetical = (batch["role"] == 3) & batch["event_mask"].bool() & before_scan
        valid = valid & ~hypothetical.any(1)
    return valid


def eligible_query_mask(output, batch):
    mask = output["query_mask"].bool() & batch["query_mask"].bool()
    if output.get("objective", "legacy_multistage") == "terminal_state_v1":
        if "terminal_mask" not in output or output["terminal_mask"].shape != mask.shape:
            raise ValueError("Terminal supervision requires an explicit per-query terminal mask")
        mask = mask & output["terminal_mask"].bool()
    if "hypothetical" in output:
        mask = mask & ~output["hypothetical"].bool()
    return mask


def pcr_label_mask(output, batch):
    terminal = output.get("objective", "legacy_multistage") == "terminal_state_v1"
    expected = "post_neoadjuvant_s1" if terminal else "baseline"
    if output.get("pcr_cutoff", "baseline") != expected:
        raise ValueError(f"The pCR cutoff must be {expected} for this objective")
    valid = batch["pcr_valid"].bool()
    if terminal:
        if "pcr_mask" not in output or output["pcr_mask"].shape != valid.shape:
            raise ValueError("S1 pCR supervision requires an explicit applicability mask")
        valid = valid & output["pcr_mask"].bool()
    return valid


def report_concept_mask(output, batch):
    if output.get("objective") != "terminal_state_v1":
        raise ValueError("Report concept supervision requires terminal_state_v1")
    values, valid = batch["s1_concepts"], batch["s1_concept_valid"]
    if values.ndim != 2 or values.shape[1] != 4 or valid.shape != values.shape or valid.dtype != torch.bool:
        raise ValueError("Report concepts require values and boolean masks with shape [B,4]")
    if "s1_concept_mask" not in output or output["s1_concept_mask"].shape != (len(values),):
        raise ValueError("Report concepts require an explicit factual S1 mask")
    if "pcr_mask" not in output or output["pcr_mask"].shape != (len(values),):
        raise ValueError("Response report concepts require factual neoadjuvant applicability")
    mask = valid & output["s1_concept_mask"].bool()[:, None]
    mask[:, :2] &= output["pcr_mask"].bool()[:, None]
    if not torch.isfinite(values[mask]).all():
        raise ValueError("Eligible report concept targets must be finite")
    return mask


def report_training_mask(batch):
    """Match the factual S1 applicability without running a trainable encoder."""
    n = len(batch["s1_concepts"])
    factual = torch.zeros(n, dtype=torch.bool, device=batch["s1_concepts"].device)
    nac = factual.clone()
    if batch["event_mask"].shape[1]:
        factual = (batch["event_mask"][:, 0].bool() & (batch["phase"][:, 0] == 1)
                   & (batch["event_order"][:, 0] == 1) & (batch["operation"][:, 0] == 3)
                   & ((batch["role"][:, 0] == 0) | (batch["role"][:, 0] == 2)))
        columns = [0, 2, 3, 4, 5]
        nac = factual & (batch["modality_value"][:, 0, columns].bool()
                         & batch["modality_known"][:, 0, columns].bool()
                         & batch["modality_applicable"][:, 0, columns].bool()).any(1)
    return report_concept_mask({"objective": "terminal_state_v1", "s1_concept_mask": factual,
                                "pcr_mask": nac}, batch)


def patient_report_concept_loss(output, batch, targets):
    mask = report_concept_mask(output, batch)
    predictions = output["s1_concept_logits"]
    if predictions.shape != mask.shape:
        raise ValueError("Report concept predictions must have shape [B,4]")
    safe = torch.where(mask, predictions, torch.zeros_like(predictions))
    target = targets.report_targets(batch["s1_concepts"], mask)
    losses = torch.cat((F.binary_cross_entropy_with_logits(safe[:, :1], target[:, :1], reduction="none"),
                        F.smooth_l1_loss(safe[:, 1:], target[:, 1:], reduction="none")), dim=1)
    per_patient = (losses * mask).sum(1) / mask.sum(1).clamp_min(1)
    patients = mask.any(1)
    return per_patient[patients].mean() if patients.any() else safe.sum() * 0


def timeline_loss(output, batch, targets, config):
    query_mask = eligible_query_mask(output, batch)
    query = patient_query_bce(output["logits"], batch["binary"], query_mask, batch["binary_valid"])
    zero = torch.where(query_mask, output["logits"], torch.zeros_like(output["logits"])).sum() * 0
    set_loss = moment_loss = zero
    valid = scan_valid(batch, output)
    if config.forecast_weight and valid.any():
        predicted = output["forecast"][valid].float()
        observed = batch["ct1"][valid].float()
        p, t = targets.normalize(predicted), targets.normalize(observed)
        distances = torch.cdist(p, t).square() / p.shape[-1]
        set_loss = 0.5 * (distances.min(1).values.mean() + distances.min(2).values.mean())
        moment_loss = F.mse_loss(targets(predicted), targets(observed))
    pcr = zero
    pcr_valid = pcr_label_mask(output, batch) if config.pcr_weight else torch.zeros_like(batch["pcr_valid"])
    if config.pcr_weight and pcr_valid.any():
        pcr = F.binary_cross_entropy_with_logits(output["pcr_logits"][pcr_valid], batch["pcr"][pcr_valid].float())
    alignment = zero
    alignment_weight = getattr(config, "alignment_weight", 0.0)
    if alignment_weight:
        if output.get("objective") != "terminal_state_v1":
            raise ValueError("S1 alignment is only defined for the terminal_state_v1 objective")
        if valid.any():
            target = targets.anchor(batch["ct1"][valid])
            predicted = output["s1_anchor"][valid].float()
            if predicted.shape != target.shape:
                raise ValueError("S1 anchor and fixed target coordinates must have matching shapes")
            alignment = F.mse_loss(predicted, target)
    drift = output.get("drift_regularization", zero)
    if not isinstance(drift, torch.Tensor):
        drift = zero + float(drift)
    report_weight = getattr(config, "report_concept_weight", 0.0)
    report_loss = patient_report_concept_loss(output, batch, targets) if report_weight else zero
    total = (query + config.forecast_weight * (set_loss + moment_loss)
             + config.pcr_weight * pcr + alignment_weight * alignment + config.drift_weight * drift
             + report_weight * report_loss)
    pcr_name = "pcr_s1" if output.get("objective") == "terminal_state_v1" else "pcr_baseline"
    terms = {"total": total, "query": query, "forecast_set": set_loss,
             "forecast_moments": moment_loss, pcr_name: pcr, "alignment": alignment, "drift": drift}
    if "s1_concept_logits" in output:
        terms["report_concepts"] = report_loss
    return terms
