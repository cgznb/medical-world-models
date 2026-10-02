from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stageworld_tcwm.modality_data import ordinal_event_tensors
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_data import TimelineCohort
from stageworld_tcwm.timeline_losses import FixedCTMoments, patient_query_bce, scan_valid, timeline_loss
from stageworld_tcwm.timeline_training import (
    StatefulPatientSampler, TimelineTrainConfig, capture_rng, restore_rng, train_timeline,
)


def test_patient_average_and_unobserved_query_nan_are_masked():
    logits = torch.tensor([[0., 2., float("nan")], [4., float("nan"), float("nan")],
                           [float("nan"), float("nan"), float("nan")]], requires_grad=True)
    labels = torch.tensor([0., 1., float("nan")])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.bool)
    observed = torch.tensor([1, 1, 0], dtype=torch.bool)
    loss = patient_query_bce(logits, labels, mask, observed)
    expected = (F.softplus(torch.tensor([0., 2.])).mean() + F.softplus(torch.tensor(-4.))) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[~(mask & observed[:, None])]) == 0


def test_scan_target_is_fixed_differentiable_and_permutation_invariant():
    generator = torch.Generator().manual_seed(1)
    observed = torch.randn(3, 27, 8, generator=generator)
    target = FixedCTMoments(8).fit(observed, torch.tensor([1, 1, 0], dtype=torch.bool))
    before = {name: value.clone() for name, value in target.state_dict().items()}
    predicted = torch.randn(3, 27, 8, generator=generator, requires_grad=True)
    output = {"logits": torch.zeros(3, 2, requires_grad=True), "query_mask": torch.ones(3, 2, dtype=torch.bool),
              "forecast": predicted, "forecast_mask": torch.tensor([1, 0, 1], dtype=torch.bool),
              "pcr_logits": torch.zeros(3, requires_grad=True)}
    batch = {"binary": torch.tensor([0., 1., 0.]), "binary_valid": torch.ones(3, dtype=torch.bool),
             "pcr": torch.zeros(3), "pcr_valid": torch.ones(3, dtype=torch.bool),
             "image_valid": torch.ones(3, 2, dtype=torch.bool), "ct1": observed,
             "query_mask": output["query_mask"], "scan_mask": torch.tensor([1, 1, 0], dtype=torch.bool)}
    assert scan_valid(batch, output).tolist() == [True, False, False]
    config = TimelineTrainConfig()
    loss = timeline_loss(output, batch, target, config)
    permuted = dict(batch, ct1=observed[:, torch.randperm(27, generator=generator)])
    second = timeline_loss(output, permuted, target, config)
    torch.testing.assert_close(loss["forecast_set"], second["forecast_set"])
    torch.testing.assert_close(loss["forecast_moments"], second["forecast_moments"])
    loss["total"].backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad[0].abs().sum() > 0
    assert predicted.grad[1:].count_nonzero() == 0
    assert not list(target.parameters())
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_hypothetical_suffix_has_no_factual_endpoint_or_scan_supervision():
    logits = torch.tensor([[.2, 100.]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    output = {"logits": logits, "query_mask": mask, "hypothetical": torch.tensor([[False, True]])}
    batch = {"binary": torch.ones(1), "binary_valid": torch.ones(1, dtype=torch.bool),
             "pcr": torch.zeros(1), "pcr_valid": torch.zeros(1, dtype=torch.bool),
             "query_mask": mask, "image_valid": torch.ones(1, 2, dtype=torch.bool),
             "scan_event_index": torch.tensor([2]), "role": torch.tensor([[0, 3, 0]]),
             "event_mask": torch.ones(1, 3, dtype=torch.bool)}
    assert not scan_valid(batch).any()
    earlier_scan = dict(batch, scan_event_index=torch.tensor([1]))
    assert scan_valid(earlier_scan).all()
    config = TimelineTrainConfig(forecast_weight=0, pcr_weight=0)
    loss = timeline_loss(output, batch, None, config)["total"]
    torch.testing.assert_close(loss, F.softplus(torch.tensor(-.2)))
    loss.backward()
    assert logits.grad[0, 0] != 0 and logits.grad[0, 1] == 0


def test_sampler_and_rng_resume_exactly():
    sampler = StatefulPatientSampler(torch.arange(7), 19)
    assert len(sampler.next(11)) == 11
    checkpoint = sampler.state_dict()
    expected = sampler.next(20)
    restored = StatefulPatientSampler(torch.arange(7), 0)
    restored.load_state_dict(checkpoint)
    assert torch.equal(expected, restored.next(20))
    rng = capture_rng()
    first = (torch.rand(3), np.random.random(3))
    restore_rng(rng)
    assert torch.equal(first[0], torch.rand(3))
    assert np.array_equal(first[1], np.random.random(3))


@pytest.mark.parametrize("field,value", [("learning_rate", float("nan")), ("pcr_weight", float("inf")),
                                         ("batch_size", 1.5), ("patience", True)])
def test_training_config_rejects_nonfinite_or_nonintegral(field, value):
    with pytest.raises(ValueError):
        replace(TimelineTrainConfig(), **{field: value}).validate()


class TinyTimeline(nn.Module):
    """Small stochastic model to exercise the actual restartable training loop."""

    def __init__(self, cfg):
        super().__init__()
        self.image = nn.Linear(cfg.image_dim, cfg.image_dim)
        self.outcome = nn.Linear(cfg.image_dim, 1)
        self.decoder = nn.Linear(cfg.image_dim, cfg.image_dim)
        self.dropout = nn.Dropout(.2)
        self.register_buffer("mean", torch.zeros(cfg.image_dim))

    def fit_statistics(self, batch):
        self.mean.copy_(batch["ct0"].mean((0, 1)))

    def forward(self, batch):
        spatial = self.dropout(self.image(batch["ct0"] - self.mean))
        baseline = self.outcome(spatial.mean(1)).squeeze(-1)
        return {"logits": baseline[:, None].expand_as(batch["query_order"]),
                "query_mask": batch["query_mask"], "forecast": self.decoder(spatial),
                "pcr_logits": baseline, "forecast_mask": batch["image_valid"][:, 1]}


def _small_cohort(tmp_path, evaluation_role="outer_evaluation"):
    ids = [f"P{i}" for i in range(8)]
    rows = [{"patient_id": patient, "methods": {"chemotherapy": 1, "immunotherapy": 0,
             "targeted": 0, "interventional": 0, "hipec": 0}} for patient in ids]
    tensors = ordinal_event_tensors(rows, torch.ones(8, 3, dtype=torch.long))
    generator = torch.Generator().manual_seed(12)
    tensors.update(ct0=torch.randn(8, 27, 8, generator=generator),
                   ct1=torch.randn(8, 27, 8, generator=generator),
                   clinical=torch.randn(8, 32, generator=generator),
                   image_valid=torch.ones(8, 2, dtype=torch.bool), binary=torch.arange(8).float() % 2,
                   pcr=torch.arange(8).float() % 2, binary_valid=torch.ones(8, dtype=torch.bool),
                   pcr_valid=torch.ones(8, dtype=torch.bool))
    split = {"train": ids[:4], "validation": ids[4:6], evaluation_role: ids[6:]}
    cohort = TimelineCohort({"schema": "modality-event-v2", "ids": ids, "tensors": tensors,
                            "metadata": {"time_basis": "ordinal_stage", "source_mode": "synthetic"},
                            "encoders": {"fit_ids": split["train"], "clinical": {}}})
    cohort.save(tmp_path / "cohort.pt")
    (tmp_path / "split.json").write_text(json.dumps(split))


@pytest.mark.parametrize("evaluation_role", ["outer_evaluation", "test"])
def test_interrupted_training_restores_optimizer_rng_and_sampler(tmp_path, monkeypatch, evaluation_role):
    import stageworld_tcwm.timeline_model as model_module
    monkeypatch.setattr(model_module, "TimelineModel", TinyTimeline)
    _small_cohort(tmp_path, evaluation_role)
    config = TimelineTrainConfig(device="cpu", batch_size=2, accumulation_steps=2,
                                 max_optimizer_steps=4, minimum_optimizer_steps=1,
                                 validation_interval=2, checkpoint_interval=1, patience=3)
    model_config = TimelineConfig(hidden=64, image_dim=8)
    source_root = tmp_path / "source"
    source_root.mkdir()
    def run(directory, resume=False):
        return train_timeline(tmp_path / "cohort.pt", tmp_path / "split.json", model_config, config,
                              tmp_path / directory, source_root, resume=resume)
    uninterrupted = run("continuous")
    original_step = torch.optim.AdamW.step
    calls = 0
    def interrupt(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("injected interruption after two saved updates")
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run("interrupted")
    assert json.loads((tmp_path / "interrupted/status.json").read_text())["status"] == "fail"
    evaluation_prefix = "outer" if evaluation_role == "outer_evaluation" else "test"
    assert not (tmp_path / f"interrupted/{evaluation_prefix}_metrics.json").exists()
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    resumed = run("interrupted", resume=True)
    assert uninterrupted["selected_step"] == resumed["selected_step"]
    assert uninterrupted[evaluation_role] == resumed[evaluation_role]
    a = torch.load(tmp_path / "continuous/last.pt", weights_only=True)
    b = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)
    assert a["optimizer_steps"] == b["optimizer_steps"] == 4
    for name in a["model_state"]:
        torch.testing.assert_close(a["model_state"][name], b["model_state"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identical"):
        train_timeline(tmp_path / "cohort.pt", tmp_path / "split.json", model_config,
                       replace(config, learning_rate=2e-4), tmp_path / "interrupted", source_root, resume=True)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_fixed_test_is_isolated_until_selection_and_skipped_by_diagnostic(tmp_path, monkeypatch, diagnostic):
    import stageworld_tcwm.modality_support as support_module
    import stageworld_tcwm.timeline_model as model_module
    import stageworld_tcwm.timeline_training as training_module

    _small_cohort(tmp_path, "test")
    cohort = TimelineCohort.load(tmp_path / "cohort.pt")
    split = json.loads((tmp_path / "split.json").read_text())
    training_indices = torch.tensor([cohort.ids.index(patient) for patient in split["train"]])
    test_indices = torch.tensor([cohort.ids.index(patient) for patient in split["test"]])
    output_dir = tmp_path / "run"
    test_evaluations, support_fits = [], []

    class CheckedTimeline(TinyTimeline):
        def fit_statistics(self, batch):
            torch.testing.assert_close(batch["ct0"], cohort.tensors["ct0"][training_indices])
            super().fit_statistics(batch)

    monkeypatch.setattr(model_module, "TimelineModel", CheckedTimeline)
    original_fit_support = support_module.fit_modality_support

    def checked_support(batch):
        torch.testing.assert_close(batch["ct0"], cohort.tensors["ct0"][training_indices])
        support_fits.append(len(batch["ct0"]))
        return original_fit_support(batch)

    monkeypatch.setattr(support_module, "fit_modality_support", checked_support)
    original_collect = training_module.collect_predictions

    def checked_collect(model, actual_cohort, indices, *args, **kwargs):
        if torch.equal(indices, test_indices):
            assert not diagnostic
            selected = torch.load(output_dir / "best.pt", weights_only=True)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, selected["model_state"][name], rtol=0, atol=0)
            history = json.loads((output_dir / "history.json").read_text())
            assert history[-1]["optimizer_steps"] == 2
            test_evaluations.append(selected["selected_step"])
        else:
            assert {actual_cohort.ids[index] for index in indices.tolist()} == set(split["validation"])
        return original_collect(model, actual_cohort, indices, *args, **kwargs)

    monkeypatch.setattr(training_module, "collect_predictions", checked_collect)
    original_batch = TimelineCohort.batch

    def checked_batch(actual_cohort, indices, *args, **kwargs):
        if diagnostic:
            assert not set(torch.as_tensor(indices).tolist()) & set(test_indices.tolist())
        return original_batch(actual_cohort, indices, *args, **kwargs)

    monkeypatch.setattr(TimelineCohort, "batch", checked_batch)
    config = TimelineTrainConfig(device="cpu", batch_size=2, accumulation_steps=1,
                                 max_optimizer_steps=2, minimum_optimizer_steps=1,
                                 validation_interval=1, checkpoint_interval=1)
    source_root = tmp_path / "source"
    source_root.mkdir()
    args = (tmp_path / "cohort.pt", tmp_path / "split.json", TimelineConfig(hidden=64, image_dim=8),
            config, output_dir, source_root)
    result = train_timeline(*args, diagnostic=diagnostic)
    assert support_fits == [4]
    assert result["selection_rule"] == "minimum_validation_patient_mean_query_BCE"
    assert result["test_used_for_selection"] is False
    assert result["historical_holdouts_repartitioned"] is True
    assert result["original_holdout65_accessed"] is True
    contract = json.loads((output_dir / "contract.json").read_text())["contract"]
    assert contract["evaluation_role"] == "test"
    assert contract["holdout65_accessed"] is True
    assert contract["split_patients"] == {"train": 4, "validation": 2, "test": 2}
    assert not (output_dir / "outer_metrics.json").exists()
    if diagnostic:
        assert test_evaluations == []
        assert "test" not in result
        assert not (output_dir / "test_metrics.json").exists()
        assert not (output_dir / "test_predictions.json").exists()
    else:
        assert test_evaluations == [result["selected_step"]]
        rows = json.loads((output_dir / "test_predictions.json").read_text())
        assert {row["patient_id"] for row in rows} == set(split["test"])
        saved = json.loads((output_dir / "test_metrics.json").read_text())
        assert saved["used_for_selection"] is False
        assert saved["metrics"] == result["test"]
    assert train_timeline(*args, diagnostic=diagnostic, resume=True) == result
    assert len(test_evaluations) == (0 if diagnostic else 1)
    assert support_fits == [4]
