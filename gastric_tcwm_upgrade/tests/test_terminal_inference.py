from dataclasses import asdict

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.modality_inference import ModalityPredictor
from stageworld_tcwm.modality_support import fit_modality_support
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import EVENT_FIELDS, TimelineModel


def setup_predictor(n=6):
    batch = modality_batch(n=n)
    model = TimelineModel(TimelineConfig(image_dim=16, hidden=64, dropout=0., objective="terminal_state_v1"))
    model.fit_statistics(batch)
    model.eval()
    return batch, ModalityPredictor(model, fit_modality_support(batch), {"source_mode": "synthetic"})


def explicit_path(batch, prefix):
    history = {**batch, **{name: batch[name][:, :prefix].clone() for name in EVENT_FIELDS}}
    future = {name: batch[name][:, prefix:].clone() for name in EVENT_FIELDS}
    return history, future


@pytest.mark.parametrize("prefix", [0, 1, 2])
def test_explicit_strategy_reaches_same_terminal_state_without_observed_suffix(prefix):
    batch, predictor = setup_predictor()
    history, future = explicit_path(batch, prefix)
    history["ct1"] = torch.full_like(history["ct1"], float("nan"))
    result = predictor.predict_strategy(history, future, strategy_name="explicit_reference_path")
    expected = predictor.predict(batch)
    torch.testing.assert_close(result["risk"][:, 0], expected["risk"][:, 3], rtol=1e-5, atol=1e-6)
    assert result["query_mask"].all()
    assert result["support"]["support_audited"]
    assert result["metadata"]["future_event_source"] == "explicit_argument"
    assert result["metadata"]["causal_effects_identified"] is False
    assert result["metadata"]["contains_hypothetical_history"] == [True] * len(batch["ct0"])
    assert (future["role"] == 2).all()


def test_factual_predict_masks_raw_early_risks():
    batch, predictor = setup_predictor()
    output = predictor.predict(batch)
    assert output["risk"][:, :3].isnan().all()
    assert output["risk"][:, 3].isfinite().all()
    assert output["metadata"]["readout"] == "factual_complete_terminal_only"


def test_strategy_rejects_implicit_or_incomplete_future():
    batch, predictor = setup_predictor()
    history, future = explicit_path(batch, 1)
    with pytest.raises(ValueError, match="explicitly"):
        predictor.predict_strategy(history, None, strategy_name="missing")
    with pytest.raises(ValueError, match="physically truncated"):
        predictor.predict_strategy(batch, future, strategy_name="future_leak")
    short = {name: value[:, :1] for name, value in future.items()}
    with pytest.raises(ValueError, match="every future event"):
        predictor.predict_strategy(history, short, strategy_name="unfinished")
    with pytest.raises(ValueError, match="third modeled event"):
        predictor.predict_strategy(history, future, strategy_name="wrong_boundary", final_event_order=2)
    future["phase"][:, -1] = 1
    with pytest.raises(ValueError, match="stage semantics"):
        predictor.predict_strategy(history, future, strategy_name="wrong_stage")


@pytest.mark.parametrize("stage,modality", [(1, 6), (2, 0)])
def test_unsupported_absence_is_not_given_a_supported_risk(stage, modality):
    batch, predictor = setup_predictor()
    history, future = explicit_path(batch, 0)
    future["modality_value"][:, stage, modality] = 0
    result = predictor.predict_strategy(history, future, strategy_name="unsupported_absence")
    assert result["risk"].isnan().all()
    assert not result["query_mask"].any()
    assert result["metadata"]["unsupported_future_action"] == [True] * len(batch["ct0"])


def test_sparse_or_missing_support_cannot_authorize_strategy_risks():
    batch, predictor = setup_predictor(n=2)
    history, future = explicit_path(batch, 1)
    result = predictor.predict_strategy(history, future, strategy_name="sparse")
    assert result["risk"].isnan().all()
    assert result["metadata"]["sparse_future_action"] == [True, True]
    predictor.support.pop("phase_action_patients")
    result = predictor.predict_strategy(history, future, strategy_name="unaudited")
    assert not result["support"]["support_audited"]
    assert result["risk"].isnan().all()


def test_terminal_export_and_legacy_checkpoint_defaults_roundtrip(tmp_path):
    batch, predictor = setup_predictor()
    for objective in ("terminal_state_v1", "legacy_multistage"):
        model = predictor.model if objective == "terminal_state_v1" else TimelineModel(
            TimelineConfig(image_dim=16, hidden=64, dropout=0.))
        if objective == "legacy_multistage":
            model.fit_statistics(batch)
        model.eval()
        config = asdict(model.cfg)
        if objective == "legacy_multistage":
            config.pop("objective")
        path = tmp_path / f"{objective}.pt"
        torch.save({"schema": "modality-event-v2", "model_config": config,
                    "model_state": model.state_dict(), "support": predictor.support,
                    "fit_ids": ["SYN-A"], "encoders": {"fit_ids": ["SYN-A"]},
                    "metadata": {"source_mode": "synthetic"}}, path)
        loaded = ModalityPredictor.load(path)
        assert loaded.model.cfg.objective == objective
        expected = ModalityPredictor(model, predictor.support, {}).predict(batch)
        torch.testing.assert_close(loaded.predict(batch)["risk"], expected["risk"], rtol=0, atol=0, equal_nan=True)
        if objective == "legacy_multistage":
            assert loaded.predict(batch)["query_mask"].all()
            history, future = explicit_path(batch, 1)
            with pytest.raises(ValueError, match="requires terminal_state_v1"):
                loaded.predict_strategy(history, future, strategy_name="invalid_legacy")
