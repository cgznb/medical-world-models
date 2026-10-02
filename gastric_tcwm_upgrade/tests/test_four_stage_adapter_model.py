"""Synthetic checks for a bounded trainable adapter with frozen measurements."""
import pytest
import torch
from torch.nn import functional as F

from modality_fixtures import modality_batch
from stageworld_tcwm.four_stage_model import FourStageModel


def batch():
    data = modality_batch(n=12, image_dim=8)
    data["modality_value"] = data["modality_value"].bool()
    return data


def fitted(data, adapter_rank=2, dropout=.5):
    model = FourStageModel(image_dim=8, hidden_dim=4, rank=2,
                           dropout=dropout, state_adapter_rank=adapter_rank)
    model.fit_statistics(data)
    return model


def test_zero_adapter_preserves_seeded_backbone_and_all_outputs_exactly():
    data = batch()
    torch.manual_seed(17)
    original = fitted(data, adapter_rank=0).eval()
    torch.manual_seed(17)
    adapted = fitted(data).eval()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)
    assert not any("state_adapter" in name for name in original.state_dict())
    assert adapted.config["state_adapter_rank"] == 2
    for name, value in original(data).items():
        torch.testing.assert_close(value, adapted(data)[name], rtol=0, atol=0)
    # Historical config files contain no adapter key and retain strict loading.
    legacy_config = {key: value for key, value in original.config.items()
                     if key != "state_adapter_rank"}
    restored = FourStageModel(**legacy_config).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    for name, value in original(data).items():
        torch.testing.assert_close(value, restored(data)[name], rtol=0, atol=0)


def test_adapter_is_bounded_and_applied_to_baseline_before_dropout():
    data = batch()
    model = fitted(data).eval()
    initial = model.initial_state(data).detach()
    with torch.no_grad():
        model.state_adapter.up.weight.zero_()
        model.state_adapter.up.bias.copy_(torch.tensor([1e6, -1e6, .5, 0.]))
    delta = model.initial_state(data) - initial
    expected = .1 * torch.tensor([1e6, -1e6, .5, 0.]).tanh()
    torch.testing.assert_close(delta, expected.expand_as(delta), rtol=0, atol=6e-8)
    assert delta.abs().max() <= .1 + 1e-7


def test_terminal_mode_only_trains_small_adapter_and_transitions():
    data = batch()
    model = fitted(data)
    trainable = model.configure_terminal_adaptation()
    expected_prefixes = ("state_adapter.", "medical_transition.",
                         "surgery_transition.", "risk_head.")
    for name, parameter in model.named_parameters():
        expected = name.startswith(expected_prefixes) or name in ("nac_gate", "post_gate")
        assert parameter.requires_grad == expected, name
    assert {id(parameter) for parameter in trainable} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    default = FourStageModel(state_adapter_rank=2)
    assert sum(parameter.numel() for parameter in default.configure_terminal_adaptation()) == 201
    for mode in (False, True):
        model.train(mode)
        for module in (model.image_encoder, model.clinical_encoder,
                       model.representation_dropout, model.pcr_head, model.ct1_decoder):
            assert not module.training
        assert model.state_adapter.training == mode
        assert model.medical_transition.training == mode
    with pytest.raises(ValueError, match="state_adapter_rank > 0"):
        fitted(data, adapter_rank=0).configure_terminal_adaptation()


def test_s3_gradients_update_all_states_while_backbone_and_statistics_stay_fixed():
    torch.manual_seed(71)
    data = batch()
    model = fitted(data)
    trainable = model.configure_terminal_adaptation()
    # After the initially zero risk head learns, S3 gradients must reach S0.
    with torch.no_grad():
        model.risk_head.weight.fill_(.2)
    fixed = {name: value.detach().clone() for name, value in model.state_dict().items()
             if name not in {name for name, parameter in model.named_parameters()
                             if parameter.requires_grad}}
    before = model.event_states(data).detach().clone()
    optimizer = torch.optim.SGD(trainable, lr=.2)
    for step in range(2):
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(model(data)["logits"], data["binary"]).backward()
        assert model.state_adapter.up.weight.grad.abs().sum() > 0
        for module in (model.medical_transition, model.surgery_transition, model.risk_head):
            assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
                       for parameter in module.parameters())
        assert model.nac_gate.grad.abs() > 0
        assert model.post_gate.grad.abs() > 0
        if step == 0:
            # Zero up weights preserve exact identity and delay down gradients
            # for one adapter update, without blocking the trainable up branch.
            assert model.state_adapter.down.weight.grad.abs().sum() == 0
        else:
            assert model.state_adapter.down.weight.grad.abs().sum() > 0
        optimizer.step()
    after = model.event_states(data).detach()
    assert all(not torch.equal(before[:, stage], after[:, stage]) for stage in range(4))
    for name, value in fixed.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    for module in (model.image_encoder, model.clinical_encoder, model.pcr_head, model.ct1_decoder):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_frozen_auxiliary_heads_keep_s1_gradient_path_and_recurrence_uses_only_s3():
    data = batch()
    model = fitted(data)
    with torch.no_grad():
        model.pcr_head.weight.fill_(.2)  # Stand in for an auxiliary-trained head.
        model.risk_head.weight.fill_(.2)
    model.configure_terminal_adaptation()
    result = model(data)
    pcr_loss = F.binary_cross_entropy_with_logits(result["pcr_logits"], data["pcr"])
    pcr_gradient = torch.autograd.grad(pcr_loss, result["states"], retain_graph=True)[0]
    assert pcr_gradient[:, 1].abs().sum() > 0
    assert pcr_gradient[:, [0, 2, 3]].abs().sum() == 0
    recurrence = F.binary_cross_entropy_with_logits(result["logits"], data["binary"])
    recurrence_gradient = torch.autograd.grad(recurrence, result["states"], retain_graph=True)[0]
    assert recurrence_gradient[:, 3].abs().sum() > 0
    assert recurrence_gradient[:, :3].abs().sum() == 0
    auxiliary_loss = pcr_loss + F.smooth_l1_loss(result["ct1_forecast"], model.ct1_target(data))
    auxiliary_loss.backward()
    assert model.state_adapter.up.weight.grad.abs().sum() > 0
    assert model.medical_transition.down.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.pcr_head.parameters())
    assert all(parameter.grad is None for parameter in model.ct1_decoder.parameters())
    assert all(parameter.grad is None or parameter.grad.abs().sum() == 0
               for parameter in model.surgery_transition.parameters())
    assert model.post_gate.grad is None or model.post_gate.grad.abs() == 0


def test_adapter_does_not_expose_future_measurements_or_actions_to_earlier_states():
    data = batch()
    model = fitted(data).eval()
    with torch.no_grad():
        model.state_adapter.up.weight.fill_(.3)
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
    for name, value in original.items():
        torch.testing.assert_close(value, model(changed)[name], rtol=0, atol=0)
    changed["modality_value"][:, 1, 6] = False
    changed["modality_value"][:, 2, 0] = False
    revised = model(changed)
    torch.testing.assert_close(original["states"][:, :2], revised["states"][:, :2], rtol=0, atol=0)
    torch.testing.assert_close(original["pcr_logits"], revised["pcr_logits"], rtol=0, atol=0)
    torch.testing.assert_close(original["ct1_forecast"], revised["ct1_forecast"], rtol=0, atol=0)


def test_old_freeze_mode_also_freezes_adapter_and_preserves_s1():
    data = batch()
    model = fitted(data).eval()
    model.configure_terminal_adaptation()
    model.train()
    before = model.event_states(data)[:, :2].detach().clone()
    trainable = model.freeze_representation()
    model.train()
    assert not model.state_adapter.training
    assert not model.medical_transition.training
    assert not model.representation_dropout.training
    assert all(not parameter.requires_grad for parameter in model.state_adapter.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=.1)
    for _ in range(2):
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(model(data)["logits"], data["binary"]).backward()
        optimizer.step()
    torch.testing.assert_close(before, model.event_states(data)[:, :2], rtol=0, atol=0)


@pytest.mark.parametrize("rank", [-1, 1.5, True])
def test_adapter_rank_rejects_invalid_values(rank):
    with pytest.raises(ValueError, match="nonnegative integer"):
        FourStageModel(state_adapter_rank=rank)
