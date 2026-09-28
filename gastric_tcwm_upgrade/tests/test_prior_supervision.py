from dataclasses import replace

import pytest
import torch

from stageworld_tcwm.config import TrainConfig
from stageworld_tcwm.losses import total_loss
from stageworld_tcwm.model import TreatmentBeliefWorld


def test_prior_reconstruction_trains_forecast_without_posterior(config, cohort):
    model = TreatmentBeliefWorld(config)
    batch = cohort.batch(torch.arange(8))
    model.fit_statistics(batch)
    output = model(batch, samples=2, seed=11)
    tc = TrainConfig(epochs=3, warmup_epochs=1, ct_weight=0, kl_weight=0,
                     pcr_weight=0, flow_weight=0, prior_ct_weight=1)
    loss, _ = total_loss(output, batch, model, tc, epoch=0)
    loss.backward()
    gradients = [p.grad for p in model.prior_params.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in model.posterior.parameters())
    assert all(p.grad is None for p in model.outcome.parameters())


def test_forecast_features_cannot_read_ct1(config, cohort):
    model = TreatmentBeliefWorld(config).eval()
    batch = cohort.batch(torch.arange(4))
    before = model(batch, samples=2, seed=11)['prior_features']
    batch['ct1'] = torch.randn_like(batch['ct1'])*100
    after = model(batch, samples=2, seed=11)['prior_features']
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_readout_penalty_respects_disabled_warmup_heads(config, cohort):
    model = TreatmentBeliefWorld(config)
    batch = cohort.batch(torch.arange(4))
    loss, metrics = total_loss(model(batch), batch, model,
                              TrainConfig(epochs=3, warmup_epochs=1, readout_l2=1), epoch=0)
    loss.backward()
    assert not metrics['readout_l2_active']
    assert all(p.grad is None for p in model.outcome.parameters())
    assert all(p.grad is None for p in model.pcr_output.parameters())
