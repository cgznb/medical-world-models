from copy import deepcopy

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import EVENT_FIELDS, TimelineModel


def terminal_model(batch):
    model = TimelineModel(TimelineConfig(image_dim=16, hidden=64, dropout=0., objective="terminal_state_v1"))
    model.fit_statistics(batch)
    return model.eval()


@pytest.mark.parametrize("settings", [{"ablation": "static"}, {"time_basis": "calendar_days"},
                                      {"assimilate_ct1": True}])
def test_terminal_protocol_rejects_untrained_modes(settings):
    assert TimelineConfig().objective == "legacy_multistage"
    with pytest.raises(ValueError, match="requires dynamic ordinal"):
        TimelineConfig(objective="terminal_state_v1", **settings).validate()


def test_only_complete_factual_s3_can_use_terminal_readout():
    batch = modality_batch()
    model = terminal_model(batch)
    output = model(batch)
    assert output["objective"] == "terminal_state_v1"
    assert torch.equal(output["query_mask"], output["terminal_mask"])
    assert output["terminal_mask"].tolist() == [[False, False, False, True]] * 2
    assert (output["logits"][:, :3] == 0).all()
    torch.testing.assert_close(output["logits"][:, 3], model.outcome(output["checkpoint_states"][3]))
    with pytest.raises(ValueError, match="raw intermediate"):
        model.query_many(output["checkpoint_states"], torch.ones(2, 1, dtype=torch.long))
    missing = deepcopy(batch)
    missing["event_mask"][:, 2] = False
    missing["query_mask"][:, 3] = False
    assert not model(missing)["terminal_mask"].any()


@pytest.mark.parametrize("prefix", [0, 1, 2])
def test_a_truncated_prefix_is_never_relabelled_terminal(prefix):
    batch = modality_batch()
    model = terminal_model(batch)
    for name in EVENT_FIELDS:
        batch[name] = batch[name][:, :prefix]
    batch["query_order"] = torch.arange(prefix + 1).repeat(2, 1)
    batch["query_mask"] = torch.ones(2, prefix + 1, dtype=torch.bool)
    result = model(batch)
    assert not result["query_mask"].any()
    assert (result["logits"] == 0).all()
    assert len(result["checkpoint_states"]) == prefix + 1
    if prefix == 0:
        assert not result["pcr_mask"].any()
        assert not result["forecast_mask"].any()


def test_s1_response_and_anchor_ignore_ct1_and_future_actions():
    batch = modality_batch()
    model = terminal_model(batch)
    original = model(batch)
    changed = deepcopy(batch)
    changed["ct1"].fill_(float("nan"))
    changed["modality_value"][:, 1:, :] = 0
    result = model(changed)
    assert original["pcr_cutoff"] == "post_neoadjuvant_s1"
    torch.testing.assert_close(original["s1_anchor"], original["checkpoint_states"][1].z.mean(1))
    for key in ("s1_anchor", "pcr_logits", "forecast"):
        torch.testing.assert_close(original[key], result[key], rtol=0, atol=0)
    no_nac = deepcopy(batch)
    no_nac["modality_value"][:, 0, :] = 0
    assert not torch.equal(original["pcr_logits"], model(no_nac)["pcr_logits"])
    original["pcr_logits"].sum().backward()
    gradients = [p.grad for p in model.event_jump.parameters() if p.grad is not None]
    assert gradients and any(g.abs().sum() > 0 for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients)


def test_pcr_requires_factual_present_neoadjuvant_treatment():
    batch = modality_batch(n=5)
    model = terminal_model(batch)
    batch["modality_value"][1, 0] = 0
    batch["modality_known"][2, 0] = False
    batch["role"][3, 0] = 3
    batch["role"][4, 0] = 1
    result = model(batch)
    assert result["pcr_mask"].tolist() == [True, False, False, False, False]
    assert result["forecast_mask"].tolist() == [True, True, True, False, False]
    assert result["terminal_mask"][:, 3].tolist() == [True, True, True, False, False]
    batch["scan_event_index"][0] = 2
    with pytest.raises(ValueError, match="scan_event_index == 1"):
        model(batch)


def test_provenance_does_not_change_the_action_transition():
    batch = modality_batch()
    model = terminal_model(batch)
    report = model(batch)
    for role in (0, 3):
        changed = deepcopy(batch)
        changed["role"].fill_(role)
        result = model(changed)
        for before, after in zip(report["checkpoint_states"], result["checkpoint_states"]):
            for field in ("z", "memory", "history", "active_value", "active_known"):
                torch.testing.assert_close(getattr(before, field), getattr(after, field), rtol=0, atol=0)
        torch.testing.assert_close(result["pcr_logits"], report["pcr_logits"], rtol=0, atol=0)
        if role == 3:
            assert result["checkpoint_states"][-1].hypothetical.all()
            assert not result["terminal_mask"].any()


def test_terminal_boundary_requires_the_actual_stage_semantics():
    batch = modality_batch()
    model = terminal_model(batch)
    batch["phase"][:, 2] = 1
    with pytest.raises(ValueError, match="stage semantics"):
        model(batch)
    batch["phase"][:, 2] = 3
    batch["modality_applicable"][:, 0, 6] = True
    with pytest.raises(ValueError, match="stage scope"):
        model(batch)
