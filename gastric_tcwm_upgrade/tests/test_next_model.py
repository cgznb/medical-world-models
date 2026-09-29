from dataclasses import replace

import pytest
import torch

from stageworld_tcwm.losses import feature_set_loss
from stageworld_tcwm.model import TreatmentBeliefWorld


def test_updated_reconstruction_trains_observation_only_at_legal_stage(config, cohort):
    model = TreatmentBeliefWorld(replace(config, observation_update="residual")).eval()
    batch = cohort.batch(torch.arange(4))
    model.fit_statistics(batch)
    epsilon = torch.randn(4, 3, config.latent_dim)
    output = model(batch, samples=3, epsilon=epsilon, return_diagnostics=True)
    loss = feature_set_loss(output["updated_features"], batch["ct1"],
                            batch["image_valid"].all(1), model.image_scale)
    loss.backward()
    for parameter in (model.observation_gate, model.observation_attention.in_proj_weight):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    altered = {key: value.clone() for key, value in batch.items()}
    altered["ct1"] = altered["ct1"].roll(1, 0)
    altered["binary"] = 1-altered["binary"]
    altered["pcr"] = 1-altered["pcr"]
    altered["future_actual_treatment"] = torch.randn_like(batch["treatment"])
    after = model(altered, samples=3, epsilon=epsilon)
    torch.testing.assert_close(output["predictions"][:, 0], after["predictions"][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(output["pcr_logits"], after["pcr_logits"], rtol=0, atol=0)
    batch["ct1_available_stage"].fill_(2)
    unavailable = model(batch, samples=3, epsilon=epsilon)
    torch.testing.assert_close(unavailable["updated_features"], unavailable["prior_features"], rtol=0, atol=0)


@pytest.mark.parametrize("bad", [torch.zeros(2, 3, 8), torch.zeros(4, 3, 8, dtype=torch.long),
                                  torch.full((4, 3, 8), float("nan"))])
def test_explicit_noise_rejects_invalid_inputs(config, cohort, bad):
    model = TreatmentBeliefWorld(config).eval()
    with pytest.raises(ValueError, match="epsilon"):
        model(cohort.batch(torch.arange(4)), samples=3, epsilon=bad)


def test_diagnostics_do_not_change_predictions_or_parameters(config, cohort):
    model = TreatmentBeliefWorld(replace(config, observation_update="residual")).eval()
    before = set(model.state_dict())
    batch = cohort.batch(torch.arange(4))
    plain = model(batch, samples=3, seed=17)
    diagnostic = model(batch, samples=3, seed=17, return_diagnostics=True)
    torch.testing.assert_close(plain["predictions"], diagnostic["predictions"], rtol=0, atol=0)
    assert set(model.state_dict()) == before
    states = diagnostic["diagnostics"]
    torch.testing.assert_close(states["readout"], diagnostic["predictions"], rtol=0, atol=0)
    assert states["reference"].shape[:3] == (4, 3, 3)
