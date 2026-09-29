"""Frozen patient-level CT projection with directly supervised future prediction."""
import math

import torch
from torch import nn

from .clinical import ClinicalAnchor
from .belief import validate_epsilon


class PredictiveCTWorld(nn.Module):
    measurement_sigma = .05

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.validate()
        if cfg.architecture != "predictive_ct":
            raise ValueError("PredictiveCTWorld requires the predictive_ct architecture")
        state_dim = 2 * cfg.ct_rank
        self.state_dim = state_dim
        self.register_buffer("image_mean", torch.zeros(cfg.image_dim))
        self.register_buffer("image_scale", torch.ones(cfg.image_dim))
        self.register_buffer("pca_basis", torch.zeros(cfg.image_dim, cfg.ct_rank))
        self.register_buffer("latent_mean", torch.zeros(state_dim))
        self.register_buffer("latent_scale", torch.ones(state_dim))
        self.register_buffer("tab_mean", torch.zeros(361))
        self.register_buffer("tab_scale", torch.ones(361))
        self.register_buffer("initial_logvar", torch.zeros(state_dim))
        self.register_buffer("statistics_fitted", torch.tensor(False))
        if cfg.predictive_transition:
            self.condition = nn.Sequential(nn.Linear(361, 32), nn.GELU())
            self.transition = nn.Sequential(nn.Linear(state_dim + 32, cfg.hidden), nn.GELU(),
                                            nn.Dropout(cfg.dropout),
                                            nn.Linear(cfg.hidden, 2 * state_dim))
            nn.init.zeros_(self.transition[-1].weight)
            nn.init.zeros_(self.transition[-1].bias)
        self.surgery = nn.Linear(state_dim, state_dim)
        nn.init.zeros_(self.surgery.weight)
        nn.init.zeros_(self.surgery.bias)
        self.outcome = nn.Linear(2 * state_dim, 1)
        self.pcr_output = nn.Linear(2 * state_dim, 1)
        for output in (self.outcome, self.pcr_output):
            nn.init.normal_(output.weight, std=.02)
            nn.init.zeros_(output.bias)
        if cfg.clinical_anchor:
            self.recurrence_anchor = ClinicalAnchor()
            self.pcr_anchor = ClinicalAnchor()
            nn.init.zeros_(self.outcome.weight)
            nn.init.zeros_(self.pcr_output.weight)

    @staticmethod
    def raw_condition(batch):
        return torch.cat((batch["clinical"].float(), batch["treatment"].float().flatten(1),
                          torch.log1p(batch["interval_days"][:, None].float() / 30)), 1)

    @torch.no_grad()
    def fit_statistics(self, training_batch):
        # Patient means select patient variation rather than within-scan anatomy.
        cpu = {name: value.detach().cpu() for name, value in training_batch.items()}
        observations = torch.cat((cpu["ct0"][cpu["image_valid"][:, 0]],
                                  cpu["ct1"][cpu["image_valid"][:, 1]]), 0).float()
        if not len(observations):
            raise ValueError("The CT projection requires observed training images")
        means = observations.mean(1)
        image_mean = means.mean(0)
        image_scale = means.std(0, unbiased=False).clamp_min(.05)
        standardized = (means - image_mean) / image_scale
        covariance = standardized.T @ standardized / max(1, len(standardized))
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -self.cfg.ct_rank:].flip(1)
        pivots = basis.abs().argmax(0)
        signs = basis[pivots, torch.arange(basis.shape[1])].sign()
        basis = basis * signs
        projected = ((observations - image_mean) / image_scale) @ basis
        raw_latent = torch.cat((projected.mean(1), projected.std(1, unbiased=False)), 1)
        self.image_mean.copy_(image_mean)
        self.image_scale.copy_(image_scale)
        self.pca_basis.copy_(basis)
        self.latent_mean.copy_(raw_latent.mean(0))
        self.latent_scale.copy_(raw_latent.std(0, unbiased=False).clamp_min(.05))
        raw = self.raw_condition(cpu)
        self.tab_mean.copy_(raw.mean(0))
        self.tab_scale.copy_(raw.std(0, unbiased=False).clamp_min(.05))
        self.statistics_fitted.fill_(True)
        paired = training_batch["image_valid"].all(1)
        if paired.any():
            before = self.encode_image(training_batch["ct0"][paired])
            after = self.encode_image(training_batch["ct1"][paired])
            variance = (after-before).var(0, unbiased=False).clamp_min(self.measurement_sigma**2)
            self.initial_logvar.copy_(variance.log().clamp(-6., 3.))
        else:
            self.initial_logvar.fill_(math.log(self.measurement_sigma**2))

    def encode_image(self, values):
        if not bool(self.statistics_fitted):
            raise RuntimeError("The CT projection must be fitted on training patients before use")
        with torch.autocast(values.device.type, enabled=False):
            tokens = ((values.float()-self.image_mean)/self.image_scale) @ self.pca_basis
            latent = torch.cat((tokens.mean(1), tokens.std(1, unbiased=False)), 1)
            return (latent-self.latent_mean)/self.latent_scale

    @torch.no_grad()
    def fit_clinical_anchors(self, training_batch):
        if self.cfg.clinical_anchor:
            clinical = training_batch["clinical"]
            self.recurrence_anchor.fit(clinical, training_batch["binary"], training_batch["binary_valid"])
            self.pcr_anchor.fit(clinical, training_batch["pcr"], training_batch["pcr_valid"])

    def prior_parameters(self, baseline, batch):
        if not self.cfg.predictive_transition:
            return baseline, torch.full_like(baseline, math.log(self.measurement_sigma**2))
        condition = self.condition((self.raw_condition(batch)-self.tab_mean)/self.tab_scale)
        delta, logvar_delta = self.transition(torch.cat((baseline, condition), 1)).chunk(2, -1)
        return baseline+delta.float(), (self.initial_logvar+logvar_delta.float()).clamp(-6., 3.)

    def reference_transition(self, state, surgery):
        present = (surgery == 1)[:, None, None]
        return state+torch.where(present, self.surgery(state), 0.)

    @staticmethod
    def readout_features(baseline, state):
        baseline = baseline[:, None].expand_as(state)
        return torch.cat((baseline, state-baseline), -1)

    def forward(self, batch, samples=4, seed=None, max_stage=2, compute_aux=True, force_gaussian=False,
                epsilon=None, return_diagnostics=False):
        if samples < 1 or max_stage not in (0, 1, 2):
            raise ValueError("Invalid Monte Carlo samples or stage")
        baseline = self.encode_image(batch["ct0"])
        baseline = torch.where(batch["image_valid"][:, 0, None], baseline, 0.)
        pmean, plogvar = self.prior_parameters(baseline, batch)
        generator = None if seed is None else torch.Generator(device=baseline.device).manual_seed(seed)
        shape = (len(baseline), samples, self.state_dim)
        epsilon = (torch.randn(shape, device=baseline.device, generator=generator) if epsilon is None
                   else validate_epsilon(epsilon, shape, baseline.device))
        generated = pmean[:, None]+(.5*plogvar).exp()[:, None]*epsilon
        states = [generated]
        qmean, qlogvar = pmean, plogvar
        target = None
        if max_stage >= 1 or compute_aux:
            target = self.encode_image(batch["ct1"])
            observed = target[:, None].expand(-1, samples, -1)
            valid_ct1 = batch["image_valid"][:, 1]
            qmean = torch.where(valid_ct1[:, None], target, pmean)
            qlogvar = torch.where(valid_ct1[:, None], torch.full_like(plogvar, math.log(self.measurement_sigma**2)), plogvar)
            for stage in range(1, max_stage+1):
                legal = valid_ct1 & (batch["ct1_available_stage"] <= stage)
                states.append(torch.where(legal[:, None, None], observed, generated))
        predictions = []
        references = []
        for state in states:
            reference = self.reference_transition(state, batch["surgery"])
            references.append(reference)
            predictions.append(self.outcome(self.readout_features(baseline, reference)).squeeze(-1).float())
        result = {"predictions": torch.stack(predictions, 1), "pmean": pmean, "plogvar": plogvar,
                  "qmean": qmean, "qlogvar": qlogvar, "baseline_ct_latent": baseline,
                  "predicted_ct_latent": pmean}
        if target is not None:
            result["target_ct_latent"] = target
        neural_residual = result["predictions"]
        anchor_logit = baseline.new_zeros(len(baseline))
        if self.cfg.clinical_anchor:
            anchor_logit = self.recurrence_anchor(batch["clinical"])
            neural_residual = self.cfg.residual_scale*neural_residual
            result["predictions"] = anchor_logit[:, None, None]+neural_residual
        if return_diagnostics:
            result["diagnostics"] = {"deterministic":pmean,"injected":generated,
                "updated":states[min(1,len(states)-1)],"reference":torch.stack(references,1),
                "neural_residual":neural_residual,"anchor_logit":anchor_logit,
                "readout":result["predictions"]}
        if compute_aux:
            result["features"] = batch["ct0"]
            result["pcr_logits"] = self.pcr_output(self.readout_features(baseline, generated)).squeeze(-1).float()
            if self.cfg.clinical_anchor:
                result["pcr_logits"] = (self.pcr_anchor(batch["clinical"])[:, None]
                                        + self.cfg.residual_scale*result["pcr_logits"])
            if self.cfg.predictive_transition:
                nll = .5*(math.log(2*math.pi)+plogvar+(target-pmean).square()*(-plogvar).exp()).mean(1)
                result["prior_nll"] = torch.where(batch["image_valid"].all(1), nll, 0.)
        return result
