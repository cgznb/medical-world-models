"""Synthetic checks of staged training and its held-out information boundary."""
import json
from types import SimpleNamespace

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm import four_stage_training
from stageworld_tcwm.four_stage_training import (
    EpochEarlyStopping, evaluate_four_stage, load_four_stage_export,
    train_four_stage, training_schedule,
)
from stageworld_tcwm.research_neural import INPUT_FIELDS


def cohort_fixture():
    torch.manual_seed(105)
    tensors = modality_batch(n=20, image_dim=8)
    tensors["modality_value"] = tensors["modality_value"].bool()
    # Validation has a deliberately different image distribution. A successful
    # fit is insufficient: verify the fitted buffers against training only.
    tensors["ct0"][12:16] += 1000
    tensors["ct1"][:12] += 30
    # One training patient has no confirmed NAC: its pCR must be excluded.
    tensors["modality_value"][0, 0] = False
    # Poison every floating prediction/target input in reserved test rows.
    # Training should never move or validate these rows as a whole cohort.
    for name in ("ct0", "ct1", "clinical", "pcr", "binary"):
        tensors[name][16:] = float("nan")
    ids = [f"anonymous-synthetic-{index}" for index in range(20)]
    return SimpleNamespace(ids=ids, tensors=tensors,
                           encoders={"fit_ids": ids[:12]})


@pytest.fixture(scope="module")
def diagnostic_run(tmp_path_factory):
    cohort = cohort_fixture()
    out = tmp_path_factory.mktemp("four_stage") / "diagnostic"
    cohort.sampled_minibatches = []
    original_subset = four_stage_training._subset

    def record_minibatch(batch, rows):
        # Evaluation reads all 12 training rows together; only optimization
        # reads a subset of at most eight. Record actual sampled patient rows.
        if len(batch["clinical"]) == 12 and len(rows) <= 8:
            cohort.sampled_minibatches.append(rows.cpu().tolist())
        return original_subset(batch, rows)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(four_stage_training, "_subset", record_minibatch)
        result = train_four_stage(cohort, list(range(12)), list(range(12, 16)),
                                  out, seed=29, device="cpu", diagnostic=True)
    return cohort, out, result


@pytest.mark.parametrize("phase", ["auxiliary", "terminal"])
def test_formal_schedule_uses_one_full_epoch_between_validations(phase):
    schedule = training_schedule(456, phase=phase)
    assert schedule["steps_per_epoch"] == 57
    assert schedule["validation_interval_steps"] == 57
    assert schedule["maximum_epochs"] == 100
    assert schedule["maximum_steps"] == 5700
    # A partial final minibatch remains one optimizer update in the epoch.
    partial = training_schedule(457, phase=phase)
    assert partial["steps_per_epoch"] == 58
    assert partial["maximum_steps"] == 5800


@pytest.mark.parametrize("phase", ["auxiliary", "terminal"])
def test_diagnostic_schedule_retains_its_two_update_budget(phase):
    schedule = training_schedule(456, phase=phase, diagnostic=True)
    assert schedule["steps_per_epoch"] == 57
    assert schedule["validation_interval_steps"] == 1
    assert schedule["maximum_steps"] == 2
    assert schedule["maximum_epochs"] == pytest.approx(2 / 57)


def test_early_stopping_starts_patience_after_thirty_complete_epochs():
    stopping = EpochEarlyStopping(warmup_epochs=30, patience_epochs=20, min_delta=1e-5)
    for epoch in range(31):
        improved, should_stop = stopping.update(epoch, .5)
        assert improved is (epoch == 0)
        assert not should_stop
        assert stopping.stale == 0
    assert stopping.best == .5
    assert stopping.best_epoch == 0
    for epoch in range(31, 51):
        improved, should_stop = stopping.update(epoch, .5)
        assert not improved
        assert stopping.stale == epoch - 30
        assert should_stop is (epoch == 50)


def test_warmup_best_is_preserved_and_later_improvement_resets_patience():
    stopping = EpochEarlyStopping(warmup_epochs=30, patience_epochs=20, min_delta=1e-5)
    assert stopping.update(0, .5) == (True, False)
    for epoch in range(1, 10):
        assert stopping.update(epoch, .6) == (False, False)
    assert stopping.update(10, .4) == (True, False)
    for epoch in range(11, 35):
        assert stopping.update(epoch, .45) == (False, False)
    assert stopping.best_epoch == 10
    assert stopping.best == .4
    assert stopping.stale == 4
    assert stopping.update(35, .3) == (True, False)
    assert stopping.best_epoch == 35
    assert stopping.stale == 0
    for epoch in range(36, 56):
        # Changes below min_delta must not indefinitely postpone stopping.
        improved, should_stop = stopping.update(epoch, .3 - 5e-6)
        assert not improved
        assert stopping.best == .3
        assert stopping.stale == epoch - 35
        assert should_stop is (epoch == 55)


def test_each_phase_samples_a_full_epoch_without_dropping_last_batch(diagnostic_run):
    cohort, _, result = diagnostic_run
    sampled = cohort.sampled_minibatches
    assert [len(rows) for rows in sampled] == [8, 4, 8, 4]
    assert sorted(sampled[0] + sampled[1]) == list(range(12))
    assert sorted(sampled[2] + sampled[3]) == list(range(12))
    for phase_result in (result, result["auxiliary"]):
        assert phase_result["steps_per_epoch"] == 2
        assert phase_result["completed_epochs"] == 1
        assert phase_result["selected_epoch"] == phase_result["selected_step"] / 2


def test_two_stage_diagnostic_keeps_test_sealed_and_fits_training_only(diagnostic_run):
    cohort, out, result = diagnostic_run
    model, payload = load_four_stage_export(out / "inference.pt")
    assert result["completed_steps"] == result["auxiliary"]["completed_steps"] == 2
    assert result["partition_counts"] == {"train": 12, "validation": 4}
    assert result["train"]["n"] == 12 and result["validation"]["n"] == 4
    assert result["test_evaluated"] is False
    assert result["diagnostic"] is True
    assert result["semantics"] == "weak_latent"
    assert result["concept_validated"] is False
    assert result["causal_strategy_validated"] is False
    assert result["intermediate_risk_interface_enabled"] is False
    assert result["s2_s3_measurements_available"] is False
    assert result["parameter_count"] < 8000
    initialization = result["initialization"]
    assert initialization["training_patients"] == 12
    assert initialization["ct0_training_patients"] == 12
    assert initialization["terminal_training_patients"] == 12
    assert initialization["pcr_s1_training_patients"] == 11
    assert initialization["pcr_s1_prevalence"] == pytest.approx(6 / 11)
    assert initialization["ct1_used_to_fit_statistics"] is False
    images = cohort.tensors["ct0"][:12].double()
    torch.testing.assert_close(model.image_mean, images.mean((0, 1)).float(), rtol=0, atol=0)
    torch.testing.assert_close(model.image_scale,
                               images.std((0, 1), unbiased=False).clamp_min(1e-3).float(),
                               rtol=0, atol=0)
    torch.testing.assert_close(model.anchor.mean,
                               cohort.tensors["clinical"][:12].double().mean(0).float(),
                               rtol=0, atol=0)
    assert torch.isnan(cohort.tensors["ct0"][16:]).all()
    assert torch.isnan(cohort.tensors["binary"][16:]).all()
    assert payload["metrics"] == result
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["completed_steps"] == 2
    for name in ("aux_last.pt", "last.pt"):
        checkpoint = torch.load(out / name, weights_only=True)
        assert checkpoint["step"] == 2
        assert checkpoint["optimizer"]["state"]
        assert len(checkpoint["sampler_rng"]) and checkpoint["sampler_cursor"] > 0


def test_export_roundtrip_evaluates_without_future_targets(diagnostic_run):
    cohort, out, result = diagnostic_run
    exported, payload = load_four_stage_export(out / "inference.pt")
    selected, _ = load_four_stage_export(out / "best.pt")
    full = {key: value[12:16] for key, value in cohort.tensors.items()}
    permitted = {key: full[key] for key in INPUT_FIELDS + ("binary", "binary_valid")}
    assert not {"ct1", "pcr", "pcr_valid", "scan_event_index"} & permitted.keys()
    full_metrics, full_predictions = evaluate_four_stage(selected, full)
    minimal_metrics, minimal_predictions = evaluate_four_stage(exported, permitted)
    assert full_metrics == minimal_metrics == result["validation"]
    for key in ("probability", "labels", "logits"):
        torch.testing.assert_close(full_predictions[key], minimal_predictions[key], rtol=0, atol=0)
    assert payload["selected_step"] == result["selected_step"]
    expected = ("clinical_baseline_at_terminal_step0" if result["selected_step"] == 0
                else "trained_weak_four_stage")
    assert payload["selected_kind"] == expected


def test_terminal_training_preserves_selected_auxiliary_representation(diagnostic_run):
    cohort, out, _ = diagnostic_run
    auxiliary, _ = load_four_stage_export(out / "aux_best.pt")
    terminal, _ = load_four_stage_export(out / "inference.pt")
    terminal.freeze_representation()
    frozen = {name for name, value in terminal.named_parameters() if not value.requires_grad}
    assert frozen
    auxiliary_state, terminal_state = auxiliary.state_dict(), terminal.state_dict()
    for name in frozen:
        torch.testing.assert_close(auxiliary_state[name], terminal_state[name], rtol=0, atol=0)
    # Fitted target projection and clinical anchor buffers are immutable too.
    for name, _ in terminal.named_buffers():
        torch.testing.assert_close(auxiliary_state[name], terminal_state[name], rtol=0, atol=0)
    validation = {key: value[12:16] for key, value in cohort.tensors.items()}
    torch.testing.assert_close(auxiliary.event_states(validation)[:, :2],
                               terminal.event_states(validation)[:, :2], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["overlap", "encoder_fit", "formal_size"])
def test_invalid_partition_contract_is_rejected_before_training(tmp_path, invalid):
    cohort = cohort_fixture()
    train, validation, diagnostic = list(range(12)), list(range(12, 16)), True
    if invalid == "overlap":
        validation = [11, 12, 13, 14]
        message = "overlap"
    elif invalid == "encoder_fit":
        cohort.encoders["fit_ids"] = cohort.ids[:11] + [cohort.ids[12]]
        message = "exactly the training patients"
    else:
        diagnostic = False
        message = "fixed 651/456/65"
    with pytest.raises(ValueError, match=message):
        train_four_stage(cohort, train, validation, tmp_path / invalid,
                         device="cpu", diagnostic=diagnostic)
    assert not (tmp_path / invalid).exists()


def test_completed_attempt_is_not_overwritten(diagnostic_run):
    cohort, out, _ = diagnostic_run
    original = (out / "inference.pt").read_bytes()
    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        train_four_stage(cohort, list(range(12)), list(range(12, 16)),
                         out, device="cpu", diagnostic=True)
    assert (out / "inference.pt").read_bytes() == original
