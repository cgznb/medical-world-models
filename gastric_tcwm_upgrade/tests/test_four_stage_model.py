"""Synthetic engineering boundaries; no study outcomes are used here."""
import pytest
import torch
from torch.nn import functional as F

from modality_fixtures import modality_batch
from stageworld_tcwm.four_stage_model import (
    FourStageModel, RISK_INCREMENT_BOUND, terminal_eligible,
)


def batch(n=12):
    result = modality_batch(n=n, image_dim=8)
    result["modality_value"] = result["modality_value"].bool()
    return result


def fitted(data, dropout=0.):
    model = FourStageModel(image_dim=8, hidden_dim=4, rank=2, dropout=dropout)
    model.fit_statistics(data)
    return model


def test_four_states_terminal_initialization_bound_and_small_size():
    data = batch()
    model = fitted(data).eval()
    result = model(data)
    assert result["states"].shape == (12, 4, 4)
    assert result["ct1_forecast"].shape == (12, 4)
    assert result["surgery_occurred"][0].tolist() == [False, False, True, True]
    torch.testing.assert_close(result["logits"], model.anchor(data["clinical"]), rtol=0, atol=0)
    with torch.no_grad():
        model.risk_head.weight.fill_(1e6)
    increment = model(data)["logits"] - model.anchor(data["clinical"])
    assert increment.abs().max() <= RISK_INCREMENT_BOUND + 1e-6
    assert sum(parameter.numel() for parameter in FourStageModel().parameters()) < 8000
    assert model.claims()["fixed_clinical_anchor_is_explicit_bypass"]
    assert not model.claims()["validated_concept_dynamics"]


def test_all_future_observation_and_target_values_are_excluded_from_forward():
    data = batch()
    model = fitted(data).eval()
    with torch.no_grad():
        model.risk_head.weight.fill_(.3)
    original = model(data)
    changed = {key: value.clone() for key, value in data.items()}
    changed["ct1"].fill_(float("nan"))
    changed["image_valid"][:, 1] = False
    changed["pcr"] = 1 - changed["pcr"]
    changed["pcr_valid"].logical_not_()
    changed["binary"] = 1 - changed["binary"]
    changed["binary_valid"].logical_not_()
    changed["report_concept_value"] = torch.randn(12, 4)
    for key, value in original.items():
        torch.testing.assert_close(value, model(changed)[key], rtol=0, atol=0)


def test_prefix_and_resumed_rollout_agree_future_action_does_not_change_s1():
    data = batch()
    model = fitted(data).eval()
    states = model.event_states(data)
    s1 = model.advance(model.initial_state(data), data, 0)
    torch.testing.assert_close(s1, states[:, 1], rtol=0, atol=0)
    s2 = model.advance(s1, data, 1)
    s3 = model.advance(s2, data, 2)
    torch.testing.assert_close(s3, states[:, 3], rtol=0, atol=0)
    changed = {key: value.clone() for key, value in data.items()}
    changed["modality_value"][:, 1, 6] = False
    changed["modality_value"][:, 2, 0] = False
    revised = model.event_states(changed)
    torch.testing.assert_close(states[:, :2], revised[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(revised[:, 1], revised[:, 2], rtol=0, atol=0)
    assert not torch.allclose(states[:, 3], revised[:, 3])
    changed["event_mask"][:, 1:] = False
    partial = model.event_states(changed)
    torch.testing.assert_close(partial[:, 1], partial[:, 3], rtol=0, atol=0)


def test_risk_rejects_incomplete_hypothetical_wrong_operations_and_applicability():
    data = batch()
    model = fitted(data).eval()
    data["event_mask"][0, 2] = False
    data["role"][1, 1] = 3
    data["operation"][2, 1] = 3
    data["event_order"][3, 1] = 3
    data["event_id"][4, 2] = data["event_id"][4, 0]
    data["modality_value"][5, 1, 6] = False
    data["modality_known"][5, 1, 6] = False
    data["modality_applicable"][5, 1, 6] = False
    assert terminal_eligible(data).tolist() == [False] * 6 + [True] * 6
    with pytest.raises(ValueError, match="complete factual S3"):
        model(data)


def test_pcr_requires_actual_nac_and_ct1_targets_have_separate_masks():
    data = batch()
    model = fitted(data)
    data["modality_value"][0, 0] = False
    data["role"][1, 0] = 3
    data["scan_event_index"][2] = 0
    data["pcr_valid"][3] = False
    data["image_valid"][4, 1] = False
    data["ct1"][4] = float("nan")
    targets = model.auxiliary_targets(data)
    assert targets["pcr_mask"].tolist() == [False] * 4 + [True] * 8
    assert targets["ct1_mask"].tolist() == [True, False, False, True, False] + [True] * 7
    assert torch.isfinite(targets["ct1_target"]).all()
    assert not targets["ct1_target"].requires_grad


def test_shared_medical_module_and_terminal_freeze_preserve_s1_and_gradients():
    torch.manual_seed(1)
    data = batch()
    model = fitted(data, dropout=.5)
    calls = []
    hook = model.medical_transition.register_forward_hook(lambda *args: calls.append(1))
    model.eval()(data)
    hook.remove()
    assert len(calls) == 2
    before = model.event_states(data)[:, 1].detach().clone()
    trainable = model.freeze_representation()
    model.train()
    assert not model.representation_dropout.training
    assert not model.medical_transition.training
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert all(name.startswith(("surgery_transition.", "risk_head.")) or name == "post_gate" for name in names)
    optimizer = torch.optim.AdamW(trainable, lr=.1, weight_decay=.1)
    for _ in range(2):
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(model(data)["logits"], data["binary"]).backward()
        optimizer.step()
    assert model.surgery_transition.down.weight.grad.abs().sum() > 0
    assert model.post_gate.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.medical_transition.parameters())
    assert model.nac_gate.grad is None
    after = model.event_states(data)[:, 1]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_training_ct0_only_statistics_and_standardized_projection():
    data = batch()
    model = fitted(data)
    altered = {key: value.clone() for key, value in data.items()}
    altered["ct1"].fill_(1e9)
    altered["image_valid"][:, 1] = False
    other = fitted(altered)
    for name in ("image_mean", "image_scale", "target_projection", "target_center", "target_scale"):
        torch.testing.assert_close(getattr(model, name), getattr(other, name), rtol=0, atol=0)
    data["ct1"] = data["ct0"].clone()
    targets = model.ct1_target(data)
    torch.testing.assert_close(targets.mean(0), torch.zeros(4), atol=2e-6, rtol=0)
    torch.testing.assert_close(targets.std(0, unbiased=False), torch.ones(4), atol=2e-6, rtol=0)
    with pytest.raises(RuntimeError, match="immutable"):
        model.fit_statistics(data)


def test_strict_concepts_are_not_silently_replaced_with_latent_coordinates():
    with pytest.raises(ValueError, match="real stage annotations"):
        FourStageModel(semantics="strict_concepts")
