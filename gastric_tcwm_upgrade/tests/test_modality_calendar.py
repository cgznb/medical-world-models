from copy import deepcopy

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import TimelineModel, EVENT_FIELDS


def calendar_case():
    batch = modality_batch(n=1)
    batch["time_basis"] = "calendar_days"
    batch["baseline_time"] = torch.zeros(1)
    batch["occurred_at"] = torch.tensor([[.25, 1., 1.5]])
    batch["available_at"] = torch.tensor([[.5, 1., 1.5]])
    batch["scan_time"] = torch.tensor([.75])
    model = TimelineModel(TimelineConfig(image_dim=16, hidden=64, dropout=0., time_basis="calendar_days"))
    model.fit_statistics(batch)
    return batch, model.eval()


def test_delayed_record_updates_availability_and_encodes_delay():
    batch, model = calendar_case()
    event = {key: batch[key][:, 0] for key in EVENT_FIELDS}
    updated = model.apply_event(model.initialize(batch), event)
    torch.testing.assert_close(updated.time, batch["available_at"][:, 0])
    changed = deepcopy(batch)
    changed["occurred_at"][:, 0] = .5
    with torch.no_grad():
        original = model(batch)
        current = model(changed)
    torch.testing.assert_close(original["logits"][:, 0], current["logits"][:, 0], rtol=0, atol=0)
    assert not torch.equal(original["checkpoint_states"][1].memory, current["checkpoint_states"][1].memory)


def test_future_occurrence_cannot_be_delivered_or_retrospective():
    batch, model = calendar_case()
    batch["occurred_at"][:, 0] = 2.
    with pytest.raises(ValueError, match="before it occurred"):
        model(batch)
    event = {key: batch[key][:, 0] for key in EVENT_FIELDS}
    with pytest.raises(ValueError, match="before it occurred"):
        model.apply_event(model.initialize(batch), event)


def test_query_cannot_skip_equal_time_event_or_event_after_padding():
    batch, model = calendar_case()
    batch["occurred_at"][:, 1] = .5
    batch["available_at"][:, 1] = .5
    batch["scan_event_index"][:] = 2
    with torch.no_grad():
        result = model(batch)
        with pytest.raises(ValueError, match="skipped an event"):
            model.query_many(result["checkpoint_states"], torch.tensor([[1]]), torch.tensor([[.5]]))
    batch, model = calendar_case()
    batch["event_mask"][:, 1] = False
    batch["query_mask"][:, 2] = False
    with torch.no_grad():
        result = model(batch)
        with pytest.raises(ValueError, match="skipped an event"):
            model.query_many(result["checkpoint_states"], torch.tensor([[1]]), torch.tensor([[1.5]]))


def test_scan_cannot_skip_available_event_and_missing_query_is_ignored():
    batch, model = calendar_case()
    batch["scan_time"][:] = 1.25
    with pytest.raises(ValueError, match="inconsistent information prefixes"):
        model(batch)
    batch, model = calendar_case()
    batch["query_mask"][:, -1] = False
    batch["query_time"] = torch.tensor([[0., .75, 1.25, float("nan")]])
    with torch.no_grad():
        assert torch.isfinite(model(batch)["logits"]).all()


def test_assimilation_cannot_put_future_ct1_into_baseline():
    batch = modality_batch(n=1)
    batch["scan_event_index"][:] = 0
    model = TimelineModel(TimelineConfig(image_dim=16, hidden=64, dropout=0., assimilate_ct1=True))
    model.fit_statistics(batch)
    with pytest.raises(ValueError, match="post-baseline"):
        model(batch)


def test_calendar_is_invariant_to_origin_of_verified_dates():
    batch, model = calendar_case()
    shifted = deepcopy(batch)
    for key in ("baseline_time", "occurred_at", "available_at", "scan_time"):
        shifted[key] = shifted[key] + 1000.
    with torch.no_grad():
        first, second = model(batch), model(shifted)
    torch.testing.assert_close(first["logits"], second["logits"], rtol=1e-5, atol=1e-6)
