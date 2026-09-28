from dataclasses import asdict, replace

import pytest
import torch

from stageworld_tcwm.config import ModelConfig
from stageworld_tcwm.model import TreatmentBeliefWorld


def predict(model, batch, stage=2, aux=False):
    return model(batch, samples=3, seed=19, max_stage=stage, compute_aux=aux)


def residual_model(config):
    return TreatmentBeliefWorld(replace(config, observation_update="residual")).eval()


def test_residual_observation_updates_only_legal_stages(config, cohort):
    model = residual_model(config)
    batch = cohort.batch(torch.arange(3))
    model.fit_statistics(batch)
    before = predict(model, batch, aux=True)
    batch["ct1"] = batch["ct1"].roll(1, 0)
    after = predict(model, batch, aux=True)
    torch.testing.assert_close(before["predictions"][:, 0], after["predictions"][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(before["pcr_logits"], after["pcr_logits"], rtol=0, atol=0)
    assert not torch.allclose(before["predictions"][:, 1], after["predictions"][:, 1], rtol=0, atol=1e-7)
    torch.testing.assert_close(after["predictions"][:, 1], after["predictions"][:, 2], rtol=0, atol=0)


@pytest.mark.parametrize("availability", [2, 3])
def test_residual_future_observations_fall_back_to_prior(config, cohort, availability):
    model = residual_model(config)
    batch = cohort.batch(torch.arange(3))
    batch["ct1_available_stage"].fill_(availability)
    before = predict(model, batch)["predictions"]
    torch.testing.assert_close(before[:, 0], before[:, 1], rtol=0, atol=0)
    batch["ct1"] = torch.randn_like(batch["ct1"]) * 10
    after = predict(model, batch)["predictions"]
    torch.testing.assert_close(before[:, :availability], after[:, :availability], rtol=0, atol=0)
    if availability == 3:
        torch.testing.assert_close(after[:, 0], after[:, 2], rtol=0, atol=0)
    else:
        assert not torch.allclose(before[:, 2], after[:, 2], rtol=0, atol=1e-7)


def test_residual_missing_observations_fall_back_per_patient(config, cohort):
    model = residual_model(config)
    batch = cohort.batch(torch.arange(3))
    batch["image_valid"][0, 1] = False
    before = predict(model, batch)["predictions"]
    torch.testing.assert_close(before[0, 0], before[0, 1], rtol=0, atol=0)
    torch.testing.assert_close(before[0, 0], before[0, 2], rtol=0, atol=0)
    batch["ct1"][0] = torch.randn_like(batch["ct1"][0]) * 10
    after = predict(model, batch)["predictions"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_residual_s0_does_not_read_future_or_labels(config, cohort):
    model = residual_model(config)
    batch = cohort.batch(torch.arange(3))
    before = predict(model, batch, stage=0)["predictions"]
    for key in ("ct1", "binary", "binary_valid", "pcr", "pcr_valid"):
        batch.pop(key)
    after = predict(model, batch, stage=0)["predictions"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_residual_update_does_not_require_token_correspondence(config):
    model = residual_model(config)
    predicted = torch.randn(3, 27, config.hidden)
    observed = torch.randn_like(predicted)
    before = model.observation_innovation(predicted, observed)
    after = model.observation_innovation(predicted, observed[:, torch.randperm(27)])
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-7)


def test_residual_update_receives_endpoint_gradient(config, cohort):
    model = residual_model(config)
    batch = cohort.batch(torch.arange(3))
    output = predict(model, batch)["predictions"]
    output[:, 1].square().mean().backward()
    for parameter in (model.observation_gate, model.observation_attention.in_proj_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["latent", "residual"])
def test_observation_update_checkpoint_roundtrip(config, cohort, tmp_path, mode):
    cfg = replace(config, observation_update=mode)
    model = TreatmentBeliefWorld(cfg).eval()
    batch = cohort.batch(torch.arange(3))
    expected = predict(model, batch)["predictions"]
    model_config = asdict(cfg)
    if mode == "latent":
        # Bundles written before the new option have no observation mode or weights.
        model_config.pop("observation_update")
        assert not any(name.startswith("observation_") for name in model.state_dict())
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_config": model_config, "model_state": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = TreatmentBeliefWorld(ModelConfig(**saved["model_config"])).eval()
    restored.load_state_dict(saved["model_state"], strict=True)
    actual = predict(restored, batch)["predictions"]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
