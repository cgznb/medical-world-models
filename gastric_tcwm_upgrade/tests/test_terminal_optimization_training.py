from dataclasses import asdict, replace
import json

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.modality_inference import ModalityPredictor
from stageworld_tcwm.modality_support import fit_modality_support
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_data import TimelineCohort
from stageworld_tcwm.timeline_model import EVENT_FIELDS, TimelineModel
from stageworld_tcwm.timeline_training import (TimelineTrainConfig, checkpoint_head_status,
                                               collect_predictions, optimizer_groups,
                                               set_head_training_status, train_timeline)
from test_terminal_training import TinyTerminal
from test_timeline_training import _small_cohort


@pytest.mark.parametrize("name", ["evaluate_test", "validate_initial", "freeze_image_projection", "collect_full_train_metrics"])
def test_optimization_flags_require_booleans(name):
    with pytest.raises(ValueError, match="boolean"):
        replace(TimelineTrainConfig(), **{name: 1}).validate()


def test_frozen_projection_is_explicit_and_all_other_parameters_are_optimized():
    model = TinyTerminal(TimelineConfig(image_dim=8, hidden=64))
    model.image.requires_grad_(False)
    with pytest.raises(ValueError, match="image projection"):
        optimizer_groups(model, TimelineTrainConfig())
    groups, manifest = optimizer_groups(model, TimelineTrainConfig(freeze_image_projection=True))
    assert [row["name"] for row in manifest] == ["event_state_readout"]
    assert {id(p) for group in groups for p in group["params"]} == {id(p) for p in model.parameters() if p.requires_grad}


def test_initial_checkpoint_can_win_resume_and_never_score_test(tmp_path, monkeypatch):
    import stageworld_tcwm.timeline_model as model_module
    import stageworld_tcwm.timeline_training as training_module

    _small_cohort(tmp_path, "test")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(model_module, "TimelineModel", TinyTerminal)
    original_collect = training_module.collect_predictions
    observed_roles = []

    def checked_collect(model, cohort, indices, *args, **kwargs):
        members = set(indices.tolist())
        assert not members.intersection({6, 7}), "Validation-only training must never forward test patients"
        observed_roles.append(members)
        rows, metrics = original_collect(model, cohort, indices, *args, **kwargs)
        # Make the selected step deterministic while still exercising real evaluation.
        metrics["selection_nll"] = 0.0 if model.prediction_head_status["optimizer_steps"] == 0 else 1.0
        return rows, metrics

    monkeypatch.setattr(training_module, "collect_predictions", checked_collect)
    config = TimelineTrainConfig(device="cpu", batch_size=2, accumulation_steps=1,
                                 max_optimizer_steps=2, minimum_optimizer_steps=1,
                                 validation_interval=1, checkpoint_interval=1, evaluate_test=False,
                                 validate_initial=True, collect_full_train_metrics=True,
                                 freeze_image_projection=True)
    cfg = TimelineConfig(image_dim=8, hidden=64, objective="terminal_state_v1")

    def run(name, resume=False):
        return train_timeline(tmp_path/"cohort.pt", tmp_path/"split.json", cfg, config,
                              tmp_path/name, source, resume=resume)

    uninterrupted = run("continuous")
    original_step = torch.optim.AdamW.step

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("Synthetic interruption with a saved step-zero recovery point")

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run("resumed")
    checkpoint = torch.load(tmp_path/"resumed/last.pt", weights_only=True)
    assert checkpoint["optimizer_steps"] == checkpoint["selected_step"] == 0
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    resumed = run("resumed", resume=True)
    assert resumed["selected_step"] == 0 and resumed["optimizer_steps"] == 2
    assert resumed["test"] is None and resumed["test_evaluation_skipped"]
    assert resumed["validation"] == uninterrupted["validation"]
    assert resumed["train"]["patients"] == 4
    assert resumed["initial_validation"]["head_training_status"]["world_model_trained"] is False
    assert resumed["validation"]["pcr"]["available"] is False
    assert resumed["validation"]["forecast_moment_mse"] is None
    assert not (tmp_path/"resumed/test_predictions.json").exists()
    assert not (tmp_path/"resumed/test_metrics.json").exists()
    selected = torch.load(tmp_path/"resumed/inference.pt", weights_only=True)
    assert selected["optimizer_steps"] == 0 and selected["total_optimizer_steps"] == 2
    assert selected["head_training_status"]["world_model_trained"] is False
    assert selected["metadata"]["head_training_status"] == selected["head_training_status"]
    for key, value in selected["model_state"].items():
        torch.testing.assert_close(value, checkpoint["model_state"][key], rtol=0, atol=0)


def test_step_zero_predictor_masks_auxiliary_outputs_and_rejects_strategy(tmp_path):
    batch = modality_batch(3, image_dim=8)
    cfg = TimelineConfig(hidden=64, image_dim=8, objective="terminal_state_v1", dropout=0.)
    model = TimelineModel(cfg)
    model.fit_statistics(batch)
    status = checkpoint_head_status(cfg, TimelineTrainConfig(), 0)
    predictor = ModalityPredictor(model, fit_modality_support(batch), {"head_training_status": status})
    result = predictor.predict(batch)
    assert result["risk"][:, 3].isfinite().all()
    assert result["pcr_probability"].isnan().all() and not result["pcr_valid"].any()
    assert result["forecast"].isnan().all() and not result["forecast_valid"].any()
    history = {**batch, **{name: batch[name][:, :1] for name in EVENT_FIELDS}}
    future = {name: batch[name][:, 1:] for name in EVENT_FIELDS}
    with pytest.raises(ValueError, match="untrained world model"):
        predictor.predict_strategy(history, future, strategy_name="untrained")
    path = tmp_path/"inference.pt"
    torch.save({"schema": "modality-event-v2", "model_config": asdict(cfg), "model_state": model.state_dict(),
                "fit_ids": ["synthetic"], "encoders": {"fit_ids": ["synthetic"]},
                "support": fit_modality_support(batch), "metadata": {"head_training_status": status},
                "head_training_status": status, "optimizer_steps": 1}, path)
    with pytest.raises(ValueError, match="step and prediction-head status"):
        ModalityPredictor.load(path)


@pytest.mark.parametrize("corruption", [None, "report_metadata", "step_zero_trained", "zero_weight_pcr"])
def test_export_head_flags_match_selected_step_and_training_contract(tmp_path, corruption):
    batch = modality_batch(3, image_dim=8)
    cfg = TimelineConfig(hidden=64, image_dim=8, objective="terminal_state_v1", dropout=0.)
    model = TimelineModel(cfg)
    model.fit_statistics(batch)
    step = 1 if corruption == "zero_weight_pcr" else 0
    training = TimelineTrainConfig(pcr_weight=0.)
    status = checkpoint_head_status(cfg, training, step)
    if corruption == "step_zero_trained":
        status["world_model_trained"] = True
    if corruption == "zero_weight_pcr":
        status["pcr_head_trained"] = True
    metadata = {"head_training_status": status,
                "report_concept_head_trained": corruption == "report_metadata"}
    path = tmp_path / "inference.pt"
    torch.save({"schema": "modality-event-v2", "model_config": asdict(cfg), "model_state": model.state_dict(),
                "fit_ids": ["synthetic"], "encoders": {"fit_ids": ["synthetic"]},
                "support": fit_modality_support(batch), "metadata": metadata,
                "head_training_status": status, "optimizer_steps": step,
                "contract": {"training": asdict(training)}}, path)
    if corruption is None:
        assert ModalityPredictor.load(path).predict(batch)["pcr_probability"].isnan().all()
    else:
        with pytest.raises(ValueError, match="Prediction-head training flags disagree"):
            ModalityPredictor.load(path)


def test_zero_step_anchor_reference_and_residual_metrics_are_honest(tmp_path):
    _small_cohort(tmp_path, "test")
    cohort = TimelineCohort.load(tmp_path/"cohort.pt")
    cfg = TimelineConfig(hidden=64, image_dim=8, objective="terminal_state_v1", dropout=0.,
                         capacity_profile="compact_v1", terminal_clinical_anchor=True,
                         clinical_normalization="continuous_only", init_blocks=1, field_blocks=1,
                         history_blocks=1, event_blocks=1, drift_blocks=1,
                         observation_blocks=1, readout_blocks=1)
    model = TimelineModel(cfg)
    fitting = cohort.batch(torch.arange(4))
    model.fit_statistics(fitting)
    result = model.fit_outcome_priors(fitting)
    assert result["initialized"]
    model.training_recurrence_probability = .5
    set_head_training_status(model, checkpoint_head_status(cfg, TimelineTrainConfig(), 0))
    _, metrics = collect_predictions(model, cohort, torch.tensor([4, 5]), 2, "cpu")
    assert metrics["clinical_anchor_reference"] == metrics["per_query"]["terminal"]
    assert metrics["terminal_residual_logit_rms"] == metrics["terminal_residual_logit_std"] == 0
    assert metrics["training_constant_baseline"]["patients"] == 2
