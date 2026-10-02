from copy import deepcopy

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modality_fixtures import modality_batch
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import EVENT_FIELDS, TimelineModel


def compact_config(**changes):
    settings = dict(image_dim=16, hidden=64, objective="terminal_state_v1", capacity_profile="compact_v1",
                    init_blocks=1, field_blocks=1, history_blocks=1, event_blocks=1, drift_blocks=1,
                    observation_blocks=1, readout_blocks=1, dropout=0.)
    return TimelineConfig(**dict(settings, **changes))


def fitted(batch, **settings):
    model = TimelineModel(compact_config(**settings)).eval()
    model.fit_statistics(batch)
    metadata = model.fit_outcome_priors(batch)
    return model, metadata


def test_compact_depths_reduce_capacity_and_legacy_defaults_preserve_initialization():
    torch.manual_seed(77)
    default = TimelineModel(TimelineConfig(image_dim=16, hidden=64))
    torch.manual_seed(77)
    explicit = TimelineModel(TimelineConfig(image_dim=16, hidden=64, capacity_profile="protocol_v1",
                                           clinical_normalization="standard", terminal_clinical_anchor=False))
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
    assert not hasattr(default, "outcome_priors_fitted")
    assert default.fit_outcome_priors({}) == {"initialized": False}
    compact = TimelineModel(compact_config(s1_report_concepts=True))
    for blocks in (compact.clinical_encoder.blocks, compact.clinical_encoder.pool.blocks,
                   compact.initial_memory.blocks, compact.event_jump.memory_blocks,
                   compact.pcr_pool.blocks, compact.outcome.pool.blocks, compact.s1_report_pool.blocks,
                   compact.init_blocks, compact.event_encoder.field_blocks, compact.event_encoder.history_blocks,
                   compact.event_jump.blocks):
        assert len(blocks) == 1
    assert len(default.clinical_encoder.blocks) == len(default.event_jump.memory_blocks) == 2
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in default.parameters())


@pytest.mark.parametrize("changes", [{"hidden": 128}, {"init_blocks": 2}, {"event_blocks": True},
                                    {"capacity_profile": "unknown"}, {"terminal_clinical_anchor": 1},
                                    {"terminal_residual_scale": 0}, {"terminal_residual_scale": float("nan")},
                                    {"terminal_residual_scale": True}, {"clinical_normalization": "all_binary"},
                                    {"terminal_clinical_anchor": True, "objective": "legacy_multistage"}])
def test_new_options_require_explicit_valid_semantics(changes):
    with pytest.raises(ValueError):
        compact_config(**changes).validate()


def test_continuous_only_statistics_preserve_binary_constants_and_missing_fields():
    batch = modality_batch(n=17)
    batch["clinical"][:, 0] = torch.arange(17).remainder(2)
    batch["clinical"][:, 1] = .1
    batch["clinical"][:, 2] = float("nan")
    batch["clinical"][:, 3] = torch.arange(17) + 40
    model = TimelineModel(compact_config(clinical_normalization="continuous_only"))
    model.fit_statistics(batch)
    torch.testing.assert_close(model.clinical_mean[:3], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(model.clinical_scale[:3], torch.ones(3), rtol=0, atol=0)
    torch.testing.assert_close(model.clinical_mean[3], batch["clinical"][:, 3].mean())
    torch.testing.assert_close(model.clinical_scale[3], batch["clinical"][:, 3].std(unbiased=False))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    held = deepcopy(batch)
    held["clinical"][:, 0] = 1
    held["clinical"][:, 3] = 1000
    assert torch.isfinite(model.initialize(held).z).all()
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_anchor_fits_only_valid_training_labels_and_starts_at_logistic_baseline():
    torch.manual_seed(321)
    train = modality_batch(n=24)
    train["clinical"][:, 0] = torch.arange(24).remainder(2)
    train["binary_valid"][[2, 3]] = False
    train["binary"][[2, 3]] = float("nan")
    model, metadata = fitted(train, terminal_clinical_anchor=True, clinical_normalization="continuous_only")
    valid = train["binary_valid"]
    reference = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., solver="lbfgs", max_iter=2000, tol=1e-8, random_state=17))
    reference.fit(train["clinical"][valid].double().numpy(), train["binary"][valid].numpy())
    held = modality_batch(n=6)
    output = model(held)
    np.testing.assert_allclose(output["logits"][:, 3].detach().numpy(),
                               reference.decision_function(held["clinical"].double().numpy()), atol=3e-6)
    assert metadata["terminal_prior_patients"] == 22
    assert metadata["terminal_clinical_anchor_fitted"] is True
    assert not list(model.outcome.clinical_anchor.parameters())
    assert not output["query_mask"][:, :3].any()
    assert (output["logits"][:, :3] == 0).all()
    assert model.clinical_mean[0] == 0
    assert model.outcome.clinical_anchor.mean[0] != 0
    with pytest.raises(ValueError, match="already fitted"):
        model.fit_outcome_priors(held)
    restored = TimelineModel(model.cfg).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(held)["logits"], output["logits"], rtol=0, atol=0)


def test_priors_use_factual_terminal_and_applicable_s1_labels_only():
    batch = modality_batch(n=8)
    batch["modality_value"][0, 0] = 0
    batch["role"][1, 0] = 1
    batch["role"][2, 0] = 3
    batch["modality_known"][4, 0] = False
    batch["pcr"] = torch.tensor([1., 1., 1., 0., 1., 1., float("nan"), 1.])
    batch["pcr_valid"][6] = False
    batch["binary_valid"][7] = False
    batch["binary"][7] = float("nan")
    model, metadata = fitted(batch)
    result = model(batch)
    assert metadata["pcr_prior_patients"] == 3
    assert metadata["pcr_prior_probability"] == pytest.approx(2 / 3)
    assert metadata["terminal_prior_patients"] == 5
    assert metadata["terminal_prior_probability"] == pytest.approx(2 / 5)
    torch.testing.assert_close(result["pcr_logits"].sigmoid(), torch.full((8,), 2 / 3))
    torch.testing.assert_close(result["logits"][result["terminal_mask"]].sigmoid(), torch.full((6,), 2 / 5))
    prefix = deepcopy(batch)
    for key in EVENT_FIELDS:
        prefix[key] = prefix[key][:, :0]
    prefix["query_order"] = torch.zeros(8, 1, dtype=torch.long)
    prefix["query_mask"] = torch.ones(8, 1, dtype=torch.bool)
    assert not model(prefix)["pcr_mask"].any()
    assert not model(prefix)["terminal_mask"].any()


def test_anchor_imputes_missing_clinical_from_valid_training_only():
    batch = modality_batch(n=12)
    batch["clinical"][0, 0] = float("nan")
    batch["clinical"][:, 1] = float("nan")
    batch["binary_valid"][-1] = False
    batch["clinical"][-1, 0] = 1e6
    model, _ = fitted(batch, terminal_clinical_anchor=True)
    expected = batch["clinical"][1:-1, 0].mean()
    torch.testing.assert_close(model.outcome.clinical_anchor.mean[0], expected)
    assert model.outcome.clinical_anchor.mean[1] == 0
    assert torch.isfinite(model(batch)["logits"]).all()


def test_anchored_incremental_state_matches_full_forward_and_keeps_baseline_snapshot():
    batch = modality_batch(n=8)
    model, _ = fitted(batch, terminal_clinical_anchor=True)
    with torch.no_grad():
        model.outcome.output[-1].weight.normal_(std=.1)
    output = model(batch)
    state = model.initialize(batch)
    for index in range(3):
        state = model.apply_event(state, {key: batch[key][:, index] for key in EVENT_FIELDS})
    torch.testing.assert_close(model.outcome(state), output["logits"][:, 3], rtol=1e-5, atol=1e-6)
    chosen = model._select_state(output["checkpoint_states"], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    torch.testing.assert_close(chosen.raw_clinical, batch["clinical"], rtol=0, atol=0)
    baseline = state.raw_clinical.clone()
    batch["clinical"].add_(1000)
    torch.testing.assert_close(state.raw_clinical, baseline, rtol=0, atol=0)
    snapshot = state.snapshot()
    snapshot.raw_clinical.add_(10)
    torch.testing.assert_close(state.raw_clinical, baseline, rtol=0, atol=0)


def test_compact_future_target_isolation_and_frozen_anchor_gradients():
    batch = modality_batch(n=8)
    batch["s1_concepts"] = torch.randn(8, 4)
    model, _ = fitted(batch, terminal_clinical_anchor=True, s1_report_concepts=True)
    with torch.no_grad():
        model.outcome.output[-1].weight.normal_(std=.1)
        model.pcr_output[-1].weight.normal_(std=.1)
    original = model(batch)
    changed = deepcopy(batch)
    changed["ct1"].fill_(float("nan"))
    changed["binary"].fill_(float("nan"))
    changed["pcr"].fill_(float("nan"))
    changed["s1_concepts"].fill_(float("nan"))
    result = model(changed)
    for key in ("logits", "pcr_logits", "s1_concept_logits", "forecast"):
        torch.testing.assert_close(result[key], original[key], rtol=0, atol=0)
    changed["modality_value"][:, 1:] = 0
    result = model(changed)
    for key in ("pcr_logits", "s1_concept_logits", "forecast"):
        torch.testing.assert_close(result[key], original[key], rtol=0, atol=0)
    original["logits"].sum().backward()
    assert model.outcome.output[-1].weight.grad.abs().sum() > 0
    assert all(value.grad is None for value in model.outcome.clinical_anchor.buffers())
