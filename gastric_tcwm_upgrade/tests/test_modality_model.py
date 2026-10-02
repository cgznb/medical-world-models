from copy import deepcopy

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import TimelineModel


def fitted_model(batch, **kwargs):
    model = TimelineModel(TimelineConfig(image_dim=16, hidden=64, dropout=0., **kwargs))
    model.fit_statistics(batch)
    return model.eval()


def event_at(batch, index):
    names = ("modality_value", "modality_known", "modality_applicable", "event_mask",
             "phase", "operation", "role", "event_order", "time_features",
             "occurred_at", "available_at", "event_id")
    return {key: batch[key][:, index] for key in names}


@pytest.mark.parametrize("ablation", ["dynamic", "static"])
def test_future_events_and_ct1_cannot_change_early_predictions(ablation):
    batch = modality_batch()
    model = fitted_model(batch, ablation=ablation)
    original = model(batch)
    altered = deepcopy(batch)
    altered["ct1"] = altered["ct1"] * 50 + 20
    altered["modality_value"][:, 2, 0] = 0
    changed = model(altered)
    torch.testing.assert_close(original["logits"][:, :3], changed["logits"][:, :3], rtol=0, atol=0)
    torch.testing.assert_close(original["pcr_logits"], changed["pcr_logits"], rtol=0, atol=0)
    torch.testing.assert_close(original["forecast"], changed["forecast"], rtol=0, atol=0)


def test_query_list_is_read_only_and_order_independent():
    batch = modality_batch()
    model = fitted_model(batch)
    output = model(batch)
    states = output["checkpoint_states"]
    before = [(state.z.clone(), state.memory.clone(), state.active_value.clone()) for state in states]
    one = model.query_many(states, torch.full((2, 1), 2, dtype=torch.long))
    many = model.query_many(states, torch.tensor([[3, 2, 0], [3, 2, 0]]))
    torch.testing.assert_close(one[:, 0], many[:, 1], rtol=0, atol=0)
    for state, saved in zip(states, before):
        for actual, expected in zip((state.z, state.memory, state.active_value), saved):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_plans_summaries_and_surgery_have_distinct_ledger_semantics():
    batch = modality_batch()
    model = fitted_model(batch)
    state = model.initialize(batch)
    summary = model.apply_event(state, event_at(batch, 0))
    assert not summary.active_known.any()
    planned = event_at(batch, 2)
    planned["role"] = torch.ones(2, dtype=torch.long)
    planned["operation"] = torch.full((2,), 4, dtype=torch.long)
    plan_state = model.apply_event(state, planned)
    torch.testing.assert_close(plan_state.z, state.z, rtol=0, atol=0)
    assert not plan_state.active_known.any()
    assert plan_state.planned_known[:, 0].all()
    surgery = model.apply_event(summary, event_at(batch, 1))
    assert not surgery.active_known[:, 6].any()
    again = model.apply_event(surgery, event_at(batch, 1))
    torch.testing.assert_close(again.z, surgery.z, rtol=0, atol=0)
    torch.testing.assert_close(again.memory, surgery.memory, rtol=0, atol=0)


def test_verified_start_stop_only_controls_active_modality():
    batch = modality_batch()
    model = fitted_model(batch)
    state = model.initialize(batch)
    event = event_at(batch, 2)
    event["role"] = torch.zeros(2, dtype=torch.long)
    event["operation"] = torch.ones(2, dtype=torch.long)
    started = model.apply_event(state, event)
    assert started.active_known[:, 0].all()
    assert started.active_value[:, 0].bool().all()
    event["event_id"] = torch.full((2,), 10, dtype=torch.long)
    event["operation"] = torch.full((2,), 2, dtype=torch.long)
    stopped = model.apply_event(started, event)
    assert stopped.active_known[:, 0].all()
    assert not stopped.active_value[:, 0].any()


def test_padding_does_not_change_predictions():
    batch = modality_batch()
    model = fitted_model(batch)
    original = model(batch)["logits"]
    padded = deepcopy(batch)
    for key, value in event_at(batch, 0).items():
        source = padded[key]
        padded[key] = torch.cat((source, torch.zeros_like(source[:, :1])), dim=1)
    padded["event_mask"][:, -1] = False
    changed = model(padded)["logits"]
    torch.testing.assert_close(original, changed, rtol=0, atol=1e-7)


def test_ordinal_cannot_be_relabelled_as_elapsed_days():
    batch = modality_batch()
    model = fitted_model(batch)
    state = model.initialize(batch)
    unchanged = model.advance(state, 0)
    torch.testing.assert_close(unchanged.z, state.z, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model.advance(state, 30)
    with pytest.raises(ValueError):
        fitted_model(batch, time_basis="calendar_days")(batch)


def test_calendar_no_event_drift_converges_and_has_gradients():
    batch = modality_batch(n=1)
    batch["baseline_time"] = torch.zeros(1)
    model = fitted_model(batch, time_basis="calendar_days")
    initial = model.initialize(batch)
    zero = model.advance(initial, torch.zeros(1))
    torch.testing.assert_close(initial.z, zero.z, rtol=0, atol=0)
    states = []
    for step in (.5, 1., 2.):
        model.drift.step_days = step
        states.append(model.advance(initial, torch.full((1,), 2.)).z)
    assert not torch.equal(initial.z, states[1])
    torch.testing.assert_close(states[0], states[1], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(states[1], states[2], rtol=2e-4, atol=2e-5)
    states[1].square().mean().backward()
    gradients = [parameter.grad for parameter in model.drift.parameters() if parameter.grad is not None]
    assert gradients and any(gradient.abs().sum() > 0 for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_shared_risk_and_forecast_losses_reach_event_network():
    batch = modality_batch()
    model = fitted_model(batch)
    result = model(batch)
    (result["logits"].square().mean() + result["forecast"].square().mean()).backward()
    event_gradients = [parameter.grad for name, parameter in model.named_parameters()
                       if ("event" in name or "jump" in name) and parameter.grad is not None]
    assert event_gradients and any(gradient.abs().sum() > 0 for gradient in event_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in event_gradients)
    assert not any("treatment.weight" in name or "tab_mean" in name for name in model.state_dict())


def test_old_or_unsupported_configuration_is_rejected():
    with pytest.raises(ValueError):
        TimelineConfig.from_dict({"schema": "tcwm-cohort-v1"})
    with pytest.raises(ValueError):
        TimelineConfig.from_dict({"schema_enabled": [True] * 7})
    with pytest.raises(ValueError):
        TimelineConfig.from_dict({"treatment_dim": 82})


def test_missing_postoperative_record_cannot_establish_s3():
    batch = modality_batch()
    model = fitted_model(batch)
    batch["event_mask"][:, 2] = False
    with pytest.raises(ValueError, match="missing event"):
        model(batch)
    batch["query_mask"][:, 3] = False
    result = model(batch)
    assert not result["query_mask"][:, 3].any()


def test_baseline_only_zero_event_timeline():
    batch = modality_batch()
    model = fitted_model(batch)
    for key in event_at(batch, 0):
        batch[key] = batch[key][:, :0]
    batch["query_order"] = torch.zeros(2, 1, dtype=torch.long)
    batch["query_mask"] = torch.ones(2, 1, dtype=torch.bool)
    batch["scan_event_index"] = torch.zeros(2, dtype=torch.long)
    batch["image_valid"][:, 1] = False
    output = model(batch)
    assert output["logits"].shape == (2, 1)
    assert torch.isfinite(output["logits"]).all()
    assert len(output["checkpoint_states"]) == 1
