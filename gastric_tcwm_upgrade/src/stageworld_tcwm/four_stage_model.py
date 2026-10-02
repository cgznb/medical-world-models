"""Small four-stage *weak latent* control, not a measured-concept model.

The shared medical transition is an explicit transfer assumption. CT1 embeddings
and pCR provide auxiliary targets at S1, but do not identify biological meanings
for latent coordinates. No observed S2/S3 measurements or causal effects are
claimed. The only recurrence output is at a complete factual S3 boundary.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .clinical import ClinicalAnchor
from .modality_schema import OPERATION_ID, status_ids
from .research_neural import terminal_eligible as _base_terminal_eligible
from .terminal_readiness import pcr_s1_mask


SEMANTICS = "weak_latent"
RISK_INCREMENT_BOUND = .25


def terminal_eligible(batch):
    """Require the audited three-slot factual schema, never a partial path."""
    eligible = _base_terminal_eligible(batch)
    expected_operation = torch.tensor(
        [OPERATION_ID["treatment_summary"], OPERATION_ID["procedure"],
         OPERATION_ID["treatment_summary"]], device=eligible.device)
    applicable = batch["modality_applicable"]
    if applicable.shape != (len(eligible), 3, 7):
        raise ValueError("Four-stage inputs require exactly three seven-modality slots")
    expected_applicable = torch.zeros((3, 7), dtype=torch.bool, device=eligible.device)
    expected_applicable[0, [0, 2, 3, 4, 5]] = True
    expected_applicable[1, 6] = True
    expected_applicable[2, 0] = True
    # Validate forbidden value/known/applicable combinations as well.
    status_ids(batch["modality_value"], batch["modality_known"], applicable)
    return (eligible & (batch["operation"] == expected_operation).all(1)
            & (applicable == expected_applicable).all((1, 2)))


class LowRankResidual(nn.Module):
    def __init__(self, hidden_dim, condition_dim, rank):
        super().__init__()
        self.down = nn.Linear(hidden_dim + condition_dim, rank)
        self.up = nn.Linear(rank, hidden_dim)

    def forward(self, state, condition):
        return .1 * torch.tanh(self.up(torch.tanh(self.down(torch.cat((state, condition), -1)))))


class LowRankStateAdapter(nn.Module):
    """Bounded correction to a frozen baseline state, initially exactly zero."""

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank)
        self.up = nn.Linear(rank, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, state):
        return .1 * torch.tanh(self.up(torch.tanh(self.down(state))))


class FourStageModel(nn.Module):
    """S0 -> shared medical update -> S1 -> surgery -> S2 -> medical -> S3.

    ``semantics='strict_concepts'`` deliberately rejects this model. Genuine
    measurement names, units, validity and stage targets require a separate
    clinically verified interface; renaming this latent vector is not sufficient.
    """

    def __init__(self, image_dim=768, hidden_dim=8, rank=2, dropout=.15,
                 semantics=SEMANTICS, medical_condition_encoding="masked_inapplicable",
                 state_adapter_rank=0):
        super().__init__()
        if semantics != SEMANTICS:
            raise ValueError("Strict measured-concept training requires real stage annotations and a validated concept interface; this model is weak_latent only")
        if not (image_dim >= hidden_dim >= 2 and rank >= 1 and 0 <= dropout < 1):
            raise ValueError("Require image_dim >= hidden_dim >= 2, rank >= 1 and dropout in [0,1)")
        if medical_condition_encoding not in ("masked_inapplicable", "legacy_one_hot"):
            raise ValueError("Unknown medical condition encoding")
        if (isinstance(state_adapter_rank, bool)
                or not isinstance(state_adapter_rank, int) or state_adapter_rank < 0):
            raise ValueError("state_adapter_rank must be a nonnegative integer")
        self.config = dict(image_dim=image_dim, hidden_dim=hidden_dim, rank=rank,
                           dropout=dropout, semantics=semantics,
                           medical_condition_encoding=medical_condition_encoding,
                           state_adapter_rank=state_adapter_rank)
        self.register_buffer("image_mean", torch.zeros(image_dim))
        self.register_buffer("image_scale", torch.ones(image_dim))
        self.register_buffer("target_projection", torch.zeros(image_dim, hidden_dim))
        self.register_buffer("target_center", torch.zeros(hidden_dim))
        self.register_buffer("target_scale", torch.ones(hidden_dim))
        self.register_buffer("statistics_fitted", torch.tensor(False))
        self.anchor = ClinicalAnchor()
        self.image_encoder = nn.Sequential(nn.Linear(image_dim, 4, bias=False),
                                           nn.Linear(4, hidden_dim), nn.Tanh())
        self.clinical_encoder = nn.Linear(32, hidden_dim)
        self.representation_dropout = nn.Dropout(dropout)
        self.medical_transition = LowRankResidual(hidden_dim, 28, rank)
        self.surgery_transition = LowRankResidual(hidden_dim, 4, rank)
        # Separate parameters avoid weight-decay updates to a nominally frozen
        # row of a common stage-embedding matrix. No untrained random post row.
        self.nac_gate = nn.Parameter(torch.zeros(()))
        self.post_gate = nn.Parameter(torch.zeros(()))
        self.pcr_head = nn.Linear(hidden_dim, 1)
        self.ct1_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)
        # Initialize after every original parameter so paired seeds retain
        # identical backbone/transitions and rank0 preserves the old state_dict.
        self.state_adapter = (LowRankStateAdapter(hidden_dim, state_adapter_rank)
                              if state_adapter_rank else None)
        self._representation_frozen = False
        self._terminal_adaptation = False

    @torch.no_grad()
    def fit_statistics(self, train_batch):
        """Caller supplies training rows only; never inspect held-out tensors."""
        if bool(self.statistics_fitted):
            raise RuntimeError("Training statistics are immutable once fitted")
        observed = train_batch["image_valid"][:, 0].bool()
        valid = train_batch["binary_valid"].bool() & terminal_eligible(train_batch)
        if int(observed.sum()) < self.config["hidden_dim"] or not valid.any():
            raise ValueError("Require enough observed training CT0 rows and factual terminal labels")
        tokens = train_batch["ct0"][observed].detach().double().cpu()
        if not torch.isfinite(tokens).all():
            raise ValueError("Observed training CT0 must be finite")
        mean = tokens.mean((0, 1))
        scale = tokens.std((0, 1), unbiased=False).clamp_min(1e-3)
        patient_mean = ((tokens - mean) / scale).mean(1)
        centered = patient_mean - patient_mean.mean(0)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:self.config["hidden_dim"]].T.contiguous()
        # Fix each SVD vector's arbitrary sign to make the target reproducible.
        pivots = basis.abs().argmax(0)
        signs = basis[pivots, torch.arange(basis.shape[1])].sign()
        basis *= torch.where(signs == 0, torch.ones_like(signs), signs)
        projected = patient_mean @ basis
        self.image_mean.copy_(mean)
        self.image_scale.copy_(scale)
        self.target_projection.copy_(basis)
        self.target_center.copy_(projected.mean(0))
        self.target_scale.copy_(projected.std(0, unbiased=False).clamp_min(1e-3))
        self.anchor.fit(train_batch["clinical"], train_batch["binary"], valid)
        mask = self.pcr_mask(train_batch)
        prevalence = float(train_batch["pcr"][mask].float().mean()) if mask.any() else .5
        prevalence = min(max(prevalence, 1e-4), 1 - 1e-4)
        self.pcr_head.weight.zero_()
        self.pcr_head.bias.fill_(math.log(prevalence / (1 - prevalence)))
        self.statistics_fitted.fill_(True)
        return {"kind": "train_only_CT0_standardization_PCA_and_clinical_anchor",
                "training_patients": len(observed), "ct0_training_patients": int(observed.sum()),
                "terminal_training_patients": int(valid.sum()), "pcr_s1_training_patients": int(mask.sum()),
                "pcr_s1_prevalence": prevalence, "target_rank": self.config["hidden_dim"],
                "target_source": "train_CT0_patient_mean_PCA_whitened",
                "ct1_used_to_fit_statistics": False, "validated_concept_dynamics": False,
                "semantics": SEMANTICS, "risk_increment_logit_bound": RISK_INCREMENT_BOUND}

    def initial_state(self, batch):
        """Read only baseline CT0, its observed flag, and baseline clinical data."""
        if not bool(self.statistics_fitted):
            raise RuntimeError("Fit training statistics before computing states")
        observed = batch["image_valid"][:, 0].bool()
        tokens = (batch["ct0"].float() - self.image_mean) / self.image_scale
        tokens = torch.where(observed[:, None, None], tokens, torch.zeros_like(tokens))
        image = self.image_encoder(tokens).mean(1)
        image = torch.where(observed[:, None], image, torch.zeros_like(image))
        state = torch.tanh(image + self.clinical_encoder(batch["clinical"].float()))
        if self.state_adapter is not None:
            state = state + self.state_adapter(state)
        return self.representation_dropout(state)

    def _event_status(self, batch, event_index):
        value = batch["modality_value"][:, event_index]
        known = batch["modality_known"][:, event_index]
        applicable = batch["modality_applicable"][:, event_index]
        status = status_ids(value, known, applicable)
        return torch.where(applicable, status + 1, 0)

    def advance(self, state, batch, event_index):
        """One latent transition; no endpoint risk or future observation access.

        This low-level method permits hypothetical action tensors for engineering
        checks, not clinical treatment ranking. Event slots are semantic stages,
        not elapsed time. An absent surgery is the identity operation.
        """
        if event_index not in (0, 1, 2):
            raise ValueError("event_index must be 0 (NAC), 1 (surgery), or 2 (postoperative)")
        status = self._event_status(batch, event_index)
        if event_index == 1:
            condition = F.one_hot(status[:, 6], 4).float()
            occurred = status[:, 6] == 3
            update = self.surgery_transition(state, condition) * occurred[:, None]
        else:
            condition = F.one_hot(status, 4).float()
            if self.config["medical_condition_encoding"] == "masked_inapplicable":
                # Inapplicable slots contribute zero, not a separate learned
                # column. Some postoperative inapplicability columns were never
                # activated in NAC pretraining and otherwise remain random
                # after the shared core is frozen. Unknown/absent/present stay
                # distinct. Transfer to postoperative use is still unvalidated.
                condition = condition * batch["modality_applicable"][:, event_index, :, None]
            condition = condition.flatten(1)
            gate = self.nac_gate if event_index == 0 else self.post_gate
            update = self.medical_transition(state, condition) * (1 + .1 * gate.tanh())
        updated = state + update
        return torch.where(batch["event_mask"][:, event_index, None].bool(), updated, state)

    def event_states(self, batch):
        """Return all four same-dimensional *latent*, unvalidated states."""
        state = self.initial_state(batch)
        states = [state]
        for event_index in range(3):
            state = self.advance(state, batch, event_index)
            states.append(state)
        return torch.stack(states, 1)

    @staticmethod
    def pcr_mask(batch):
        return pcr_s1_mask(batch) & (batch["scan_event_index"] == 1)

    def ct1_target(self, batch):
        """Target-only access: frozen CT0-derived projection of CT1 mean."""
        if not bool(self.statistics_fitted):
            raise RuntimeError("Fit training statistics before computing targets")
        observed = batch["image_valid"][:, 1].bool()
        ct1 = torch.where(observed[:, None, None], batch["ct1"].float(),
                          torch.zeros_like(batch["ct1"], dtype=torch.float))
        projected = ((ct1 - self.image_mean) / self.image_scale).mean(1) @ self.target_projection
        target = (projected - self.target_center) / self.target_scale
        return torch.where(observed[:, None], target, torch.zeros_like(target)).detach()

    def auxiliary_targets(self, batch):
        factual_s1 = (((batch["role"][:, 0] == 0) | (batch["role"][:, 0] == 2))
                      & batch["event_mask"][:, 0].bool() & (batch["phase"][:, 0] == 1)
                      & (batch["scan_event_index"] == 1))
        return {"pcr": batch["pcr"].float(), "pcr_mask": self.pcr_mask(batch),
                "ct1_target": self.ct1_target(batch),
                "ct1_mask": batch["image_valid"][:, 1].bool() & factual_s1}

    def forward(self, batch):
        if not terminal_eligible(batch).all():
            raise ValueError("FourStageModel only exposes complete factual S3 risk with the audited operation/applicability schema")
        states = self.event_states(batch)
        increment = RISK_INCREMENT_BOUND * self.risk_head(states[:, 3]).squeeze(-1).tanh()
        surgery = ((self._event_status(batch, 1)[:, 6] == 3)
                   & batch["event_mask"][:, 1].bool())
        surgery_occurred = torch.stack((torch.zeros_like(surgery), torch.zeros_like(surgery),
                                       surgery, surgery), 1)
        return {"logits": self.anchor(batch["clinical"]) + increment,
                "states": states, "pcr_logits": self.pcr_head(states[:, 1]).squeeze(-1),
                "ct1_forecast": self.ct1_decoder(states[:, 1]),
                "surgery_occurred": surgery_occurred}

    def freeze_representation(self):
        """Preserve S0/S1 interface; terminal learns surgery, post gate, risk.

        The frozen medical function remains differentiable with respect to its
        input S2, so recurrence gradients can still reach the surgery module.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for module in (self.surgery_transition, self.risk_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.post_gate.requires_grad_(True)
        self._representation_frozen = True
        self._terminal_adaptation = False
        self.train(self.training)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def configure_terminal_adaptation(self):
        """Adapt states with S3 recurrence plus S1 auxiliary supervision.

        Image/clinical encoders and auxiliary heads stay fixed. Frozen heads
        remain differentiable with respect to S1, so their losses can constrain
        the small state adapter and shared medical transition. This is weak
        latent adaptation, not clinically validated concept supervision.
        """
        if self.state_adapter is None:
            raise ValueError("Terminal adaptation requires state_adapter_rank > 0")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for module in (self.state_adapter, self.medical_transition,
                       self.surgery_transition, self.risk_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.nac_gate.requires_grad_(True)
        self.post_gate.requires_grad_(True)
        self._representation_frozen = False
        self._terminal_adaptation = True
        self.train(self.training)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        if self._representation_frozen or self._terminal_adaptation:
            for module in (self.image_encoder, self.clinical_encoder,
                           self.representation_dropout,
                           self.pcr_head, self.ct1_decoder):
                module.eval()
        if self._representation_frozen:
            self.medical_transition.eval()
            if self.state_adapter is not None:
                self.state_adapter.eval()
        return self

    def claims(self):
        return {"semantics": SEMANTICS, "states": ["S0", "S1", "S2", "S3"],
                "validated_concept_dynamics": False, "measured_S2_S3_targets": False,
                "treatment_strategy_interface_enabled": False,
                "intermediate_risk_interface_enabled": False,
                "medical_transition_shared": True,
                "medical_condition_encoding": self.config["medical_condition_encoding"],
                "state_adapter_rank": self.config["state_adapter_rank"],
                "state_adapter_coordinate_bound": .1 if self.state_adapter is not None else 0.,
                "postoperative_medical_transfer_is_unvalidated": True,
                "fixed_clinical_anchor_is_explicit_bypass": True,
                "risk_increment_logit_bound": RISK_INCREMENT_BOUND}
