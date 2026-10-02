from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_data import TimelineCohort
from stageworld_tcwm.timeline_losses import FixedCTMoments, timeline_loss
from stageworld_tcwm.timeline_training import (
    TimelineTrainConfig, query_consistency, train_timeline,
)
from test_timeline_training import TinyTimeline, _small_cohort


def test_terminal_loss_masks_intermediate_hypothetical_and_inapplicable_pcr():
    logits = torch.tensor([[float("nan"), 0.], [100., 2.], [100., 100.]], requires_grad=True)
    pcr_logits = torch.tensor([0., float("nan"), 100.], requires_grad=True)
    mask = torch.ones(3, 2, dtype=torch.bool)
    output = {"objective": "terminal_state_v1", "logits": logits, "query_mask": mask,
              "terminal_mask": torch.tensor([[0, 1]] * 3, dtype=torch.bool),
              "hypothetical": torch.tensor([[0, 0], [0, 0], [0, 1]], dtype=torch.bool),
              "pcr_logits": pcr_logits, "pcr_cutoff": "post_neoadjuvant_s1",
              "pcr_mask": torch.tensor([1, 0, 1], dtype=torch.bool)}
    batch = {"binary": torch.tensor([0., 1., 1.]), "binary_valid": torch.ones(3, dtype=torch.bool),
             "pcr": torch.tensor([1., float("nan"), float("nan")]),
             "pcr_valid": torch.tensor([1, 1, 0], dtype=torch.bool),
             "query_mask": mask, "image_valid": torch.zeros(3, 2, dtype=torch.bool)}
    losses = timeline_loss(output, batch, None, TimelineTrainConfig(forecast_weight=0, pcr_weight=.1))
    expected = (F.softplus(torch.tensor(0.)) + F.softplus(torch.tensor(-2.))) / 2
    torch.testing.assert_close(losses["query"], expected)
    torch.testing.assert_close(losses["pcr_s1"], F.softplus(torch.tensor(0.)))
    losses["total"].backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(pcr_logits.grad).all()
    assert logits.grad[:, 0].count_nonzero() == 0 and logits.grad[2].count_nonzero() == 0
    assert pcr_logits.grad[0] != 0 and pcr_logits.grad[1:].count_nonzero() == 0
    with pytest.raises(ValueError, match="terminal mask"):
        timeline_loss({k: v for k, v in output.items() if k != "terminal_mask"}, batch, None,
                      TimelineTrainConfig(forecast_weight=0))
    with pytest.raises(ValueError, match="applicability"):
        timeline_loss({k: v for k, v in output.items() if k != "pcr_mask"}, batch, None,
                      TimelineTrainConfig(forecast_weight=0))


def test_fixed_anchor_is_train_only_detached_reproducible_and_permutation_invariant():
    generator = torch.Generator().manual_seed(32)
    tokens = torch.randn(7, 27, 8, generator=generator, requires_grad=True)
    valid = torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    rng = torch.get_rng_state().clone()
    targets = FixedCTMoments(8, anchor_dim=64).fit(tokens, valid)
    assert torch.equal(rng, torch.get_rng_state())
    reference = FixedCTMoments(8, anchor_dim=64).fit(tokens[:4], torch.ones(4, dtype=torch.bool))
    for key, value in targets.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    anchors = targets.anchor(tokens)
    assert not anchors.requires_grad and not list(targets.parameters())
    assert anchors[:4].std(0, unbiased=False).mean() > .9
    torch.testing.assert_close(anchors, targets.anchor(tokens[:, torch.randperm(27, generator=generator)]),
                               atol=2e-6, rtol=2e-5)
    before = {key: value.clone() for key, value in targets.state_dict().items()}
    targets.anchor(tokens[4:] * 100 + 50)
    for key, value in targets.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    restored = FixedCTMoments(8, anchor_dim=64)
    restored.load_state_dict(targets.state_dict(), strict=True)
    torch.testing.assert_close(restored.anchor(tokens), anchors, rtol=0, atol=0)
    legacy = FixedCTMoments(8).fit(tokens, valid)
    assert set(legacy.state_dict()) == {"image_mean", "image_scale", "moment_mean", "moment_scale", "fitted"}
    FixedCTMoments(8).load_state_dict(legacy.state_dict(), strict=True)


def test_alignment_is_s1_only_and_uses_observed_factual_scan_mask():
    tokens = torch.randn(3, 27, 8)
    targets = FixedCTMoments(8, anchor_dim=64).fit(tokens, torch.ones(3, dtype=torch.bool))
    anchor = torch.randn(3, 64, requires_grad=True)
    output = {"objective": "terminal_state_v1", "logits": torch.zeros(3, 1, requires_grad=True),
              "query_mask": torch.ones(3, 1, dtype=torch.bool),
              "terminal_mask": torch.ones(3, 1, dtype=torch.bool), "s1_anchor": anchor,
              "forecast_mask": torch.tensor([1, 0, 1], dtype=torch.bool)}
    batch = {"ct1": tokens, "image_valid": torch.ones(3, 2, dtype=torch.bool),
             "query_mask": output["query_mask"], "binary": torch.zeros(3),
             "binary_valid": torch.ones(3, dtype=torch.bool), "pcr_valid": torch.zeros(3, dtype=torch.bool),
             "scan_event_index": torch.ones(3, dtype=torch.long),
             "role": torch.tensor([[2], [2], [3]]), "event_mask": torch.ones(3, 1, dtype=torch.bool)}
    config = TimelineTrainConfig(forecast_weight=0, pcr_weight=0, alignment_weight=.1)
    loss = timeline_loss(output, batch, targets, config)
    torch.testing.assert_close(loss["alignment"], F.mse_loss(anchor[:1], targets.anchor(tokens[:1])))
    loss["total"].backward()
    assert anchor.grad[0].abs().sum() > 0 and anchor.grad[1:].count_nonzero() == 0


@pytest.mark.parametrize("weight", [-.1, float("nan"), float("inf"), True])
def test_alignment_config_rejects_invalid_weights(weight):
    with pytest.raises(ValueError):
        replace(TimelineTrainConfig(), alignment_weight=weight).validate()


class TinyTerminal(TinyTimeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.anchor = nn.Linear(cfg.image_dim, cfg.hidden)

    def forward(self, batch):
        output = super().forward(batch)
        terminal = batch["query_mask"] & (batch["query_order"] == 3)
        output.update(objective="terminal_state_v1", query_mask=terminal, terminal_mask=terminal,
                      pcr_cutoff="post_neoadjuvant_s1", pcr_mask=torch.ones(len(terminal), dtype=torch.bool),
                      s1_anchor=self.anchor(self.image(batch["ct0"]).mean(1)))
        return output


def test_terminal_query_consistency_requires_real_comparisons_and_checks_subset_masks():
    batch = {"query_order": torch.tensor([[0, 1, 3], [0, 3, 2]]),
             "query_mask": torch.ones(2, 3, dtype=torch.bool)}

    class QueryModel(nn.Module):
        def forward(self, values):
            mask = values["query_mask"] & (values["query_order"] == 3)
            return {"query_mask": mask, "logits": values["query_order"].float() * .1}

    result = query_consistency(QueryModel(), batch)
    assert result["compared_patient_queries"] == 2 and result["subset_requests"] == 2
    with pytest.raises(ValueError, match="all-masked"):
        query_consistency(QueryModel(), dict(batch, query_mask=torch.zeros(2, 3, dtype=torch.bool)))

    class BrokenSubset(QueryModel):
        def forward(self, values):
            output = super().forward(values)
            if values["query_order"].shape[1] == 1:
                output["query_mask"].fill_(False)
            return output

    with pytest.raises(AssertionError, match="eligibility"):
        query_consistency(BrokenSubset(), batch)


def test_terminal_training_recovers_two_steps_and_exports_fixed_training_targets(tmp_path, monkeypatch):
    import stageworld_tcwm.timeline_model as model_module
    import stageworld_tcwm.timeline_training as training_module

    _small_cohort(tmp_path, "test")
    cohort = TimelineCohort.load(tmp_path / "cohort.pt")
    cohort.tensors["ct1"][4:] = cohort.tensors["ct1"][4:] * 20 + 100
    cohort.save(tmp_path / "cohort.pt")
    monkeypatch.setattr(model_module, "TimelineModel", TinyTerminal)
    config = TimelineTrainConfig(device="cpu", batch_size=2, accumulation_steps=1, alignment_weight=.1,
                                 max_optimizer_steps=2, minimum_optimizer_steps=1,
                                 validation_interval=1, checkpoint_interval=1)
    model_config = TimelineConfig(hidden=64, image_dim=8, objective="terminal_state_v1")
    source = tmp_path / "source"
    source.mkdir()
    original_collect = training_module.collect_predictions
    test_calls = []

    def checked_collect(model, actual_cohort, indices, *args, **kwargs):
        if set(indices.tolist()) == {6, 7}:
            test_calls.append(True)
        return original_collect(model, actual_cohort, indices, *args, **kwargs)

    monkeypatch.setattr(training_module, "collect_predictions", checked_collect)

    def run(name, resume=False):
        return train_timeline(tmp_path / "cohort.pt", tmp_path / "split.json", model_config, config,
                              tmp_path / name, source, resume=resume)

    continuous = run("continuous")
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupt(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("injected interruption after saved terminal update")
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run("interrupted")
    assert len(test_calls) == 1
    assert not (tmp_path / "interrupted/test_metrics.json").exists()
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    resumed = run("interrupted", resume=True)
    assert len(test_calls) == 2 and resumed["test"] == continuous["test"]
    assert resumed["selection_rule"] == "minimum_validation_terminal_BCE"
    assert resumed["terminal_boundary"] == "last_treatment_summary_not_fixed_followup_horizon"
    assert set(resumed["blocked_stages"]) == {"B_measured_concepts", "C_causal_strategies"}
    assert "blocked_arms" not in resumed
    assert set(resumed["validation"]["per_query"]) == {"terminal"}
    assert resumed["validation"]["pcr"]["cutoff"] == "post_neoadjuvant_s1"
    assert resumed["validation"]["s1_anchor_patients"] == 2
    expected = FixedCTMoments(8, anchor_dim=64).fit(cohort.tensors["ct1"][:4], torch.ones(4, dtype=torch.bool))
    first = torch.load(tmp_path / "continuous/last.pt", weights_only=True)
    second = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)
    exported = torch.load(tmp_path / "interrupted/inference.pt", weights_only=True)
    for key, value in first["model_state"].items():
        torch.testing.assert_close(value, second["model_state"][key], rtol=0, atol=0)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, exported["target_statistics"][key], rtol=0, atol=0)
    assert exported["target_space"]["fit_patients"] == 4
    assert exported["objective"] == "terminal_state_v1"
    assert exported["pcr_cutoff"] == "post_neoadjuvant_s1"
    assert exported["metadata"]["scored_query_orders"] == [3]
    assert exported["metadata"]["intermediate_state_risk_supported"] is False
    rows = json.loads((tmp_path / "interrupted/test_predictions.json").read_text())
    assert len(rows) == 2 and all(row["query_name"] == "terminal" and row["query_index"] == 3 for row in rows)


def test_actual_terminal_model_two_step_diagnostic_resumes_exactly(tmp_path, monkeypatch):
    _small_cohort(tmp_path, "test")
    source = tmp_path / "source"
    source.mkdir()
    config = TimelineTrainConfig(device="cpu", batch_size=2, accumulation_steps=1, alignment_weight=.1,
                                 max_optimizer_steps=2, minimum_optimizer_steps=1,
                                 validation_interval=1, checkpoint_interval=1)
    model_config = TimelineConfig(hidden=64, image_dim=8, objective="terminal_state_v1")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def run(name, resume=False):
        return train_timeline(tmp_path / "cohort.pt", tmp_path / "split.json", model_config, config,
                              tmp_path / name, source, resume=resume, diagnostic=True)

    try:
        first = run("actual_continuous")
        original_step = torch.optim.AdamW.step
        calls = 0

        def interrupt(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("injected actual-model recovery check")
            return original_step(optimizer, *args, **kwargs)

        monkeypatch.setattr(torch.optim.AdamW, "step", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run("actual_resumed")
        monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
        second = run("actual_resumed", resume=True)
        assert first["validation"] == second["validation"]
        assert second["query_consistency"]["compared_patient_queries"] == 2
        assert second["query_consistency"]["subset_requests"] == 1
        assert second["validation"]["pcr"]["patients"] == 2
        assert "test" not in second and not (tmp_path / "actual_resumed/test_metrics.json").exists()
        a = torch.load(tmp_path / "actual_continuous/last.pt", weights_only=True)
        b = torch.load(tmp_path / "actual_resumed/last.pt", weights_only=True)
        for key, value in a["model_state"].items():
            torch.testing.assert_close(value, b["model_state"][key], rtol=0, atol=0)
        assert b["history"][-1]["training"]["pcr_s1"] > 0
        assert b["history"][-1]["training"]["alignment"] > 0
    finally:
        torch.set_num_threads(old_threads)
