from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from stageworld_tcwm.config import ModelConfig, TrainConfig
from stageworld_tcwm.data import make_split, split_indices
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.predictive import PredictiveCTWorld
from stageworld_tcwm.synthetic import synthetic_cohort


def fitted(config, cohort, **options):
    cfg = replace(config, architecture="predictive_ct", ct_rank=4, **options)
    model = PredictiveCTWorld(cfg).eval()
    train = cohort.batch(torch.arange(24))
    model.fit_statistics(train)
    model.fit_clinical_anchors(train)
    return model, cohort.batch(torch.arange(24, 28))


def predict(model, batch, **kwargs):
    return model(batch, samples=3, seed=17, **kwargs)


def test_projection_fits_only_supplied_training_images(config, cohort):
    model, _ = fitted(config, cohort)
    expected = {name: value.clone() for name, value in model.named_buffers()}
    cohort.tensors["ct0"][24:] += 100
    cohort.tensors["ct1"][24:] *= 100
    cohort.tensors["binary"] = 1-cohort.tensors["binary"]
    other, _ = fitted(config, cohort)
    for name, value in expected.items():
        torch.testing.assert_close(value, dict(other.named_buffers())[name], rtol=0, atol=0)
    assert not model.pca_basis.requires_grad


def test_projection_selects_patient_mean_covariance(config, cohort):
    model, _ = fitted(config, cohort)
    train = cohort.batch(torch.arange(24))
    images = torch.cat((train["ct0"], train["ct1"]))
    means = images.mean(1)
    torch.testing.assert_close(model.image_scale, means.std(0, unbiased=False).clamp_min(.05))
    x = (means-model.image_mean)/model.image_scale
    _, vectors = torch.linalg.eigh(x.T @ x / len(x))
    expected = vectors[:, -model.cfg.ct_rank:]
    torch.testing.assert_close(model.pca_basis @ model.pca_basis.T, expected @ expected.T, atol=1e-5, rtol=1e-5)
    before = model.encode_image(train["ct0"])
    after = model.encode_image(train["ct0"][:, torch.randperm(27)])
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("transition", [False, True])
def test_predictive_stage_and_pcr_boundaries(config, cohort, transition):
    model, batch = fitted(config, cohort, predictive_transition=transition)
    before = predict(model, batch)
    batch["ct1"] = batch["ct1"].roll(1, 0)
    batch["binary"] = 1-batch["binary"]
    batch["pcr"] = 1-batch["pcr"]
    after = predict(model, batch)
    torch.testing.assert_close(before["predictions"][:, 0], after["predictions"][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(before["pcr_logits"], after["pcr_logits"], rtol=0, atol=0)
    assert not torch.allclose(before["predictions"][:, 1], after["predictions"][:, 1])
    torch.testing.assert_close(after["predictions"][:, 1], after["predictions"][:, 2], rtol=0, atol=0)
    with torch.no_grad():
        model.surgery.weight.fill_(.1)
    before = predict(model, batch)
    batch["surgery"].zero_()
    after = predict(model, batch)
    torch.testing.assert_close(before["pcr_logits"], after["pcr_logits"], rtol=0, atol=0)


def test_predictive_s0_does_not_read_future_or_labels(config, cohort):
    model, batch = fitted(config, cohort)
    before = predict(model, batch, max_stage=0, compute_aux=False)
    for name in ("ct1", "ct1_available_stage", "binary", "binary_valid", "pcr", "pcr_valid"):
        batch.pop(name)
    after = predict(model, batch, max_stage=0, compute_aux=False)
    torch.testing.assert_close(before["predictions"], after["predictions"], rtol=0, atol=0)
    assert "target_ct_latent" not in after


@pytest.mark.parametrize("availability", [2, 3])
def test_predictive_missing_and_future_ct1_fallback(config, cohort, availability):
    model, batch = fitted(config, cohort)
    batch["ct1_available_stage"].fill_(availability)
    batch["image_valid"][0, 1] = False
    output = predict(model, batch)["predictions"]
    torch.testing.assert_close(output[:, 0], output[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(output[0, 0], output[0, 2], rtol=0, atol=0)
    if availability == 3:
        torch.testing.assert_close(output[:, 0], output[:, 2], rtol=0, atol=0)


def test_prior_forecast_has_direct_gradient_and_correct_nll(config, cohort):
    model, batch = fitted(config, cohort)
    output = predict(model, batch)
    target, mean, logvar = output["target_ct_latent"], output["pmean"], output["plogvar"]
    expected = -torch.distributions.Normal(mean, (.5*logvar).exp()).log_prob(target).mean(1)
    torch.testing.assert_close(output["prior_nll"], expected)
    output["prior_nll"].mean().backward()
    assert model.transition[-1].weight.grad.abs().sum() > 0
    assert model.outcome.weight.grad is None
    assert model.pcr_output.weight.grad is None
    torch.testing.assert_close(mean, output["baseline_ct_latent"], rtol=0, atol=0)


def test_direct_ct_ablation_has_no_learned_transition(config, cohort):
    model, batch = fitted(config, cohort, predictive_transition=False)
    assert not hasattr(model, "transition") and not hasattr(model, "condition")
    output = predict(model, batch)
    torch.testing.assert_close(output["predicted_ct_latent"], output["baseline_ct_latent"], rtol=0, atol=0)
    assert "prior_nll" not in output


def test_predictive_anchor_and_checkpoint_roundtrip(config, cohort, tmp_path):
    model, batch = fitted(config, cohort, clinical_anchor=True)
    expected = predict(model, batch)
    anchor = model.recurrence_anchor(batch["clinical"])
    torch.testing.assert_close(expected["predictions"], anchor[:, None, None].expand(-1, 3, 3))
    path = tmp_path / "model.pt"
    torch.save({"model_config": asdict(model.cfg), "model_state": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = PredictiveCTWorld(ModelConfig(**saved["model_config"])).eval()
    restored.load_state_dict(saved["model_state"], strict=True)
    actual = predict(restored, batch)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


def test_predictive_projection_requires_observed_training_images(config, cohort):
    cfg = replace(config, architecture="predictive_ct", ct_rank=4)
    model = PredictiveCTWorld(cfg)
    batch = cohort.batch(torch.arange(4))
    with pytest.raises(RuntimeError, match="training patients"):
        predict(model, batch)
    batch["image_valid"].fill_(False)
    with pytest.raises(ValueError, match="observed training images"):
        model.fit_statistics(batch)


@pytest.mark.parametrize("transition", [False, True])
def test_predictive_train_export_inference_and_exact_resume(config, tmp_path, monkeypatch, transition):
    from stageworld_tcwm import training

    cfg = replace(config, architecture="predictive_ct", ct_rank=4, clinical_anchor=True,
                  predictive_transition=transition, dropout=.2)
    tc = TrainConfig(epochs=3, warmup_epochs=0, max_optimizer_steps=5, batch_size=16,
                     samples_train=2, samples_eval=3, observation_dropout=.25,
                     ct_weight=0, kl_weight=0, flow_weight=0,
                     prior_weight=.1 if transition else 0, readout_l2=.01,
                     readout_learning_rate=.001, patience=3)
    cohort = synthetic_cohort(n=40, image_dim=cfg.image_dim, seed=13)
    path = tmp_path / "cohort.pt"
    cohort.save(path)
    split = make_split(cohort.ids, cohort.tensors["binary"].numpy(), 17)
    full = training.train(path, split, cfg, tc, tmp_path / "full")
    save = training.atomic_save

    def interrupt(payload, destination):
        save(payload, destination)
        if Path(destination).name == "last.pt" and payload["epoch"] == 0:
            raise InterruptedError("after a complete epoch checkpoint")

    monkeypatch.setattr(training, "atomic_save", interrupt)
    with pytest.raises(InterruptedError):
        training.train(path, split, cfg, tc, tmp_path / "resume")
    monkeypatch.setattr(training, "atomic_save", save)
    resumed = training.train(path, split, cfg, tc, tmp_path / "resume", resume=True)
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert full["test_evaluated"] is resumed["test_evaluated"] is False
    assert a["history"] == b["history"]
    assert a["optimizer_steps"] == b["optimizer_steps"] == 5
    for name in a["model_state"]:
        torch.testing.assert_close(a["model_state"][name], b["model_state"][name], rtol=0, atol=0)
    predictor = Predictor(tmp_path / "resume/inference.pt")
    assert isinstance(predictor.model, PredictiveCTWorld)
    rows = split_indices(cohort, split)["validation"][:2]
    fields = {"clinical", "treatment", "interval_days", "surgery", "ct0", "ct1", "image_valid", "ct1_available_stage"}
    query = {"schema": "tcwm-query-v1", "plan_source": "hypothetical", "interval_source": "specified_query",
             "tensors": {name: value for name, value in cohort.batch(rows).items() if name in fields}}
    for stage in range(3):
        probability = predictor.predict(query, stage, samples=3, allow_extrapolation=True)["recorded_status_probability"]
        assert torch.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
