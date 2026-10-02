from dataclasses import replace
import json

import pytest
import torch
from torch.nn import functional as F

from stageworld_tcwm.data import fingerprint
from stageworld_tcwm.report_concept_data import S1_CONCEPT_NAMES, attach_report_concepts
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_data import TimelineCohort
from stageworld_tcwm.timeline_losses import (
    FixedCTMoments, patient_report_concept_loss, report_concept_mask, report_training_mask, timeline_loss,
)
from stageworld_tcwm.timeline_training import (
    TimelineTrainConfig, report_concept_metrics, train_timeline, validate_report_concept_contract,
)
from test_timeline_training import _small_cohort


def report_targets():
    values = torch.tensor([[0., .1, .5, .7], [1., .3, 1.5, 1.7],
                           [0., .2, 1., 1.2], [1., .4, 2., 2.2]])
    return FixedCTMoments(8, report_concepts=True).fit_report_concepts(values, torch.ones_like(values, dtype=torch.bool))


def test_report_statistics_use_only_valid_training_labels_and_are_detached():
    values = torch.tensor([[0., .2, 1., 2.], [1., float("nan"), 3., 4.],
                           [1., .8, 100., 200.]], requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    targets = FixedCTMoments(8, report_concepts=True).fit_report_concepts(values, valid)
    torch.testing.assert_close(targets.report_mean, torch.tensor([.5, .2, 2., 3.]))
    torch.testing.assert_close(targets.report_scale, torch.tensor([1., .05, 1., 1.]))
    assert targets.report_counts.tolist() == [2, 1, 2, 2]
    standardized = targets.report_targets(values, valid)
    assert standardized.isfinite().all() and not standardized.requires_grad
    before = {name: value.clone() for name, value in targets.state_dict().items()}
    targets.report_targets(torch.full((5, 4), 50.), torch.ones(5, 4, dtype=torch.bool))
    for name, value in targets.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    restored = FixedCTMoments(8, report_concepts=True)
    restored.load_state_dict(targets.state_dict(), strict=True)
    torch.testing.assert_close(restored.report_targets(values, valid), standardized, rtol=0, atol=0)
    decoded = targets.decode_report(torch.zeros(2, 4))
    torch.testing.assert_close(decoded, targets.report_mean.expand(2, -1))
    assert not any(name.startswith("report_") for name in FixedCTMoments(8, anchor_dim=64).state_dict())
    with pytest.raises(ValueError, match="training example"):
        targets.fit_report_concepts(values[:1], torch.tensor([[1, 0, 1, 1]], dtype=torch.bool))


def test_report_loss_balances_patients_and_masks_nan_and_non_nac_response():
    targets = report_targets()
    values = torch.tensor([[1., float("nan"), float("nan"), float("nan")],
                           [0., .25, 1.25, 1.45],
                           [float("nan")] * 4,
                           [1., .25, 1.25, 1.45]], requires_grad=True)
    valid = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    predictions = torch.tensor([[0., float("nan"), float("nan"), float("nan")],
                                 [0., 1., -1., 0.], [float("nan")] * 4,
                                 [float("nan"), float("nan"), 1., 0.]], requires_grad=True)
    batch = {"s1_concepts": values, "s1_concept_valid": valid}
    output = {"objective": "terminal_state_v1", "s1_concept_logits": predictions,
              "s1_concept_mask": torch.tensor([1, 1, 0, 1], dtype=torch.bool),
              "pcr_mask": torch.tensor([1, 1, 0, 0], dtype=torch.bool)}
    mask = report_concept_mask(output, batch)
    assert mask.tolist() == [[True, False, False, False], [True] * 4, [False] * 4,
                            [False, False, True, True]]
    loss = patient_report_concept_loss(output, batch, targets)
    log2 = F.softplus(torch.tensor(0.))
    expected = (log2 + (log2 + .5 + .5) / 4 + .5 / 2) / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(predictions.grad).all() and predictions.grad[~mask].count_nonzero() == 0
    assert values.grad is None
    assert predictions.grad[0, 0].abs() > predictions.grad[1, 0].abs()
    output["s1_concept_mask"].fill_(False)
    assert patient_report_concept_loss(output, batch, targets).item() == 0


def test_report_zero_weight_has_no_auxiliary_gradient_and_preserves_loss_keys_without_head():
    predictions = torch.zeros(2, 4, requires_grad=True)
    output = {"objective": "terminal_state_v1", "logits": torch.zeros(2, 1, requires_grad=True),
              "query_mask": torch.ones(2, 1, dtype=torch.bool), "terminal_mask": torch.ones(2, 1, dtype=torch.bool),
              "s1_concept_logits": predictions}
    batch = {"query_mask": output["query_mask"], "binary": torch.zeros(2),
             "binary_valid": torch.ones(2, dtype=torch.bool), "pcr_valid": torch.zeros(2, dtype=torch.bool),
             "image_valid": torch.zeros(2, 2, dtype=torch.bool)}
    config = TimelineTrainConfig(forecast_weight=0, pcr_weight=0, report_concept_weight=0)
    loss = timeline_loss(output, batch, None, config)
    loss["total"].backward()
    assert predictions.grad is None and loss["report_concepts"].item() == 0
    output.pop("s1_concept_logits")
    assert "report_concepts" not in timeline_loss(output, batch, None, config)


@pytest.mark.parametrize("weight", [-.1, float("nan"), float("inf"), True])
def test_report_weight_is_finite_nonnegative(weight):
    with pytest.raises(ValueError):
        replace(TimelineTrainConfig(), report_concept_weight=weight).validate()


def test_report_metrics_include_training_constant_comparators_and_untrained_flag():
    targets = report_targets()
    labels = [[0., 1.], [.1, .4], [.5, 2.], [.7, 2.2]]
    predictions = [[.5, .5], [.25, .25], [1.25, 1.25], [1.45, 1.45]]
    result = report_concept_metrics(labels, predictions, targets, trained=False)
    assert result["head_role"] == "untrained_auxiliary_control"
    assert result["report_concept_head_trained"] is False
    assert result["clinically_adjudicated"] is False and result["full_longitudinal_concepts"] is False
    for row in result["per_target"].values():
        assert row["patients"] == 2 and row["training_patients"] == 4
        for metric, value in row["model"].items():
            other = row["training_constant_baseline"][metric]
            if value is not None:
                assert value == pytest.approx(other, abs=1e-6)


def make_report_cohort(tmp_path):
    _small_cohort(tmp_path, "test")
    cohort = TimelineCohort.load(tmp_path / "cohort.pt")
    split = json.loads((tmp_path / "split.json").read_text())
    cohort.metadata["split_sha256"] = fingerprint(split)
    values = torch.tensor([[0., .1, .5, .7], [1., .3, 1.5, 1.7],
                           [0., .2, 1., 1.2], [1., .4, 2., 2.2],
                           [0., .8, 20., 30.], [1., .9, 30., 40.],
                           [0., .6, 40., 50.], [1., .7, 50., 60.]])
    observed = torch.ones(8, 4, dtype=torch.bool)
    observed[1, 1] = False
    values[1, 1] = float("nan")
    result = attach_report_concepts(cohort, split, {"s1_concepts": values, "s1_concept_valid": observed},
                                    observed, {"synthetic_test": True}, {"kind": "synthetic_test"})
    result.save(tmp_path / "cohort.pt")
    return result


def test_report_fitting_mask_excludes_missing_prefix_hypothetical_and_response_without_nac(tmp_path):
    cohort = make_report_cohort(tmp_path)
    batch = cohort.batch(torch.arange(4))
    batch["modality_value"][0, 0] = False
    batch["role"][1, 0] = 3
    batch["role"][2, 0] = 1
    batch["event_mask"][3, 0] = False
    mask = report_training_mask(batch)
    assert mask.tolist() == [[False, False, True, True], [False] * 4, [False] * 4, [False] * 4]
    batch["event_mask"] = batch["event_mask"][:, :0]
    assert not report_training_mask(batch).any()
    contract = cohort.metadata["s1_report_concepts"]
    assert validate_report_concept_contract(cohort.metadata) == contract
    with pytest.raises(ValueError, match="verified"):
        validate_report_concept_contract({"s1_report_concepts": dict(contract, validated_source=False)})


@pytest.mark.parametrize("weight", [0., .1])
def test_actual_report_head_training_recovers_and_exports_train_only_statistics(tmp_path, monkeypatch, weight):
    cohort = make_report_cohort(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    cfg = TimelineConfig(hidden=64, image_dim=8, objective="terminal_state_v1", s1_report_concepts=True)
    training = TimelineTrainConfig(device="cpu", batch_size=2, accumulation_steps=1, alignment_weight=.1,
                                   report_concept_weight=weight, max_optimizer_steps=2, minimum_optimizer_steps=1,
                                   validation_interval=1, checkpoint_interval=1)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def run(name, resume=False):
        return train_timeline(tmp_path / "cohort.pt", tmp_path / "split.json", cfg, training,
                              tmp_path / name, source, resume=resume, diagnostic=True)

    try:
        first = run("continuous")
        original_step = torch.optim.AdamW.step
        calls = 0

        def interrupt(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("report target checkpoint recovery test")
            return original_step(optimizer, *args, **kwargs)

        monkeypatch.setattr(torch.optim.AdamW, "step", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run("interrupted")
        monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
        second = run("interrupted", resume=True)
        assert first["validation"] == second["validation"]
        assert first["selection_rule"] == "minimum_validation_terminal_BCE"
        assert first["report_concept_head_trained"] is (weight > 0)
        assert first["validation"]["s1_report_concepts"]["report_concept_head_trained"] is (weight > 0)
        assert "test" not in second and not (tmp_path / "interrupted/test_metrics.json").exists()
        a = torch.load(tmp_path / "continuous/last.pt", weights_only=True)
        b = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)
        exported = torch.load(tmp_path / "interrupted/inference.pt", weights_only=True)
        for key, value in a["model_state"].items():
            torch.testing.assert_close(value, b["model_state"][key], rtol=0, atol=0)
        expected = FixedCTMoments(8, report_concepts=True).fit_report_concepts(
            cohort.tensors["s1_concepts"][:4], cohort.tensors["s1_concept_valid"][:4])
        for key in ("report_mean", "report_scale", "report_counts", "report_fitted"):
            torch.testing.assert_close(exported["target_statistics"][key], expected.state_dict()[key], rtol=0, atol=0)
        assert exported["metadata"]["report_concept_head_trained"] is (weight > 0)
        assert exported["metadata"]["full_longitudinal_concepts"] is False
        assert b["history"][-1]["training"]["report_concepts"] > 0 if weight else (
            b["history"][-1]["training"]["report_concepts"] == 0)
        counts = first["validation"]["s1_report_concepts"]["per_target"]
        assert all(counts[name]["patients"] == 2 for name in S1_CONCEPT_NAMES)
        assert counts[S1_CONCEPT_NAMES[1]]["training_patients"] == 3
    finally:
        torch.set_num_threads(old_threads)
