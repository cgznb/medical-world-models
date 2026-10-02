"""Condition-support regressions using synthetic patients only."""
import copy

import pytest
import torch
from torch.nn import functional as F

from modality_fixtures import modality_batch
from stageworld_tcwm.four_stage_model import FourStageModel
from stageworld_tcwm.four_stage_training import EXPORT_SCHEMA, load_four_stage_export


def fitted(encoding="masked_inapplicable"):
    data = modality_batch(n=12, image_dim=8)
    data["modality_value"] = data["modality_value"].bool()
    model = FourStageModel(image_dim=8, hidden_dim=4, rank=2, dropout=0.,
                           medical_condition_encoding=encoding)
    model.fit_statistics(data)
    # A zero endpoint head would hide an encoding regression in the logits.
    with torch.no_grad():
        model.risk_head.weight.fill_(.2)
        model.risk_head.bias.fill_(.05)
    return model.eval(), data


@pytest.mark.parametrize("encoding", ["masked_inapplicable", "legacy_one_hot"])
def test_unlearned_nac_columns_cannot_change_masked_postoperative_states(encoding):
    model, data = fitted(encoding)
    # These four modalities are applicable during NAC, and inapplicable after
    # surgery. Their inapplicable one-hot columns have no NAC training signal.
    unsupported = torch.tensor([8, 12, 16, 20])
    status0 = model._event_status(data, 0)
    status2 = model._event_status(data, 2)
    nac = F.one_hot(status0, 4).flatten(1)
    postoperative = F.one_hot(status2, 4).flatten(1)
    assert not nac[:, unsupported].any()
    assert postoperative[:, unsupported].all()

    original = model(data)
    loss = (original["ct1_forecast"].square().mean()
            + F.binary_cross_entropy_with_logits(original["pcr_logits"], data["pcr"]))
    loss.backward()
    columns = model.config["hidden_dim"] + unsupported
    assert not model.medical_transition.down.weight.grad[:, columns].any()

    with torch.no_grad():
        model.medical_transition.down.weight[:, columns] += 1000.
    changed = model(data)
    torch.testing.assert_close(original["states"][:, :3], changed["states"][:, :3],
                               rtol=0, atol=0)
    if encoding == "masked_inapplicable":
        for name in original:
            torch.testing.assert_close(original[name], changed[name], rtol=0, atol=0)
    else:
        assert not torch.allclose(original["states"][:, 3], changed["states"][:, 3])
        assert not torch.allclose(original["logits"], changed["logits"])


def test_masking_preserves_unknown_absent_and_present_actions():
    model, data = fitted()
    data["modality_known"][0, 0, 0] = False
    data["modality_value"][0, 0, 0] = False
    data["modality_value"][1, 0, 0] = False
    # Make this controlled transition depend only on the chemotherapy status;
    # identical incoming states must still distinguish all three statuses.
    with torch.no_grad():
        model.medical_transition.down.weight.zero_()
        model.medical_transition.down.bias.zero_()
        start = model.config["hidden_dim"]
        model.medical_transition.down.weight[:, start + 1] = -.75
        model.medical_transition.down.weight[:, start + 3] = .75
        model.medical_transition.up.weight.fill_(.3)
        model.medical_transition.up.bias.zero_()
    captured = []
    hook = model.medical_transition.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[1].detach().clone()))
    try:
        updated = model.advance(torch.zeros(12, 4), data, 0)
    finally:
        hook.remove()
    condition = captured[0].reshape(12, 7, 4)
    assert not condition[~data["modality_applicable"][:, 0]].any()
    torch.testing.assert_close(condition[:3, 0], torch.tensor([
        [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), rtol=0, atol=0)
    assert (updated[0] < updated[1]).all()
    assert (updated[1] < updated[2]).all()


def export(model, path, *, historical=False):
    config = copy.deepcopy(model.config)
    if historical:
        config.pop("medical_condition_encoding")
    torch.save({"schema": EXPORT_SCHEMA, "semantics": "weak_latent",
                "concept_validated": False, "config": config,
                "model": model.state_dict()}, path)


def test_historical_export_without_encoding_reproduces_legacy_states_and_logits(tmp_path):
    model, data = fitted("legacy_one_hot")
    path = tmp_path / "historical.pt"
    export(model, path, historical=True)
    loaded, _ = load_four_stage_export(path)
    assert loaded.config["medical_condition_encoding"] == "legacy_one_hot"
    expected, actual = model(data), loaded(data)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_new_export_preserves_masked_encoding(tmp_path):
    model, data = fitted()
    path = tmp_path / "masked.pt"
    export(model, path)
    loaded, payload = load_four_stage_export(path)
    assert payload["config"]["medical_condition_encoding"] == "masked_inapplicable"
    assert loaded.config["medical_condition_encoding"] == "masked_inapplicable"
    expected, actual = model(data), loaded(data)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="encoding"):
        FourStageModel(medical_condition_encoding="unknown")
