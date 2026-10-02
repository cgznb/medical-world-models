import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("terminal_selection_runner", ROOT / "scripts" / "run_terminal.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def selected_study(tmp_path, monkeypatch):
    from stageworld_tcwm import timeline_config, timeline_data, timeline_losses, timeline_model, timeline_training

    data = tmp_path / "data"
    data.mkdir()
    (data / "cohort.pt").write_bytes(b"mock cohort; no patient data")
    (data / "split.json").write_text(json.dumps({"train": [], "validation": [], "test": []}))
    base = json.loads((ROOT / "configs/terminal/optimization_v1.json").read_text())
    config = {
        "model": dict(base["model"], terminal_clinical_anchor=True, s1_report_concepts=True),
        "training": dict(base["training"], device="cpu", report_concept_weight=.01),
    }
    identity = {
        "diagnostic": False,
        "suite": "optimization-v1",
        "arms": ["compact_control", "anchored_report"],
        "source_sha256": {},
        "data_root": str(data),
        "data_sha256": {name: runner.file_sha256(data / name) for name in ("cohort.pt", "split.json")},
        "configurations": {name: copy.deepcopy(config) for name in ("compact_control", "anchored_report")},
        "threads": 1,
    }
    records = {
        "compact_control": {"validation": {"selection_nll": .6}, "selected_step": 10},
        "anchored_report": {"validation": {"selection_nll": .48}, "selected_step": 0},
    }
    (tmp_path / "results.json").write_text(json.dumps(records))
    (tmp_path / "study_protocol.json").write_text(json.dumps(identity))
    status = {
        "optimizer_steps": 0, "world_model_trained": False, "terminal_head_trained": False,
        "clinical_anchor_fitted": True, "pcr_head_trained": False, "forecast_head_trained": False,
        "s1_alignment_trained": False, "report_concept_head_trained": False,
    }
    export = {
        "contract_id": "mock-contract",
        "contract": {
            "source_sha256": {}, "cohort_sha256": identity["data_sha256"]["cohort.pt"],
            "split_sha256": runner.FROZEN_SPLIT_SHA256, "training": config["training"],
        },
        "model_config": config["model"],
        "model_state": {"mock_parameter": torch.tensor(1.)},
        "selected_step": 0,
        "optimizer_steps": 0,
        "target_statistics": {},
        "head_training_status": status,
        "metadata": {
            "report_concept_head_trained": False,
            "head_training_status": status,
            "training_recurrence_probability": .25,
        },
    }
    run = tmp_path / "anchored_report"
    run.mkdir()
    torch.save(export, run / "inference.pt")
    torch.save(export, run / "best.pt")
    selection = {
        "selected_arm": "anchored_report", "selected_step": 0,
        "selected_inference_sha256": runner.file_sha256(run / "inference.pt"),
        "validation_scores": {name: row["validation"]["selection_nll"] for name, row in records.items()},
        "test_used_for_selection": False,
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    calls = []

    class MockModel:
        def __init__(self, cfg):
            self.cfg = cfg

        def to(self, device):
            return self

        def load_state_dict(self, state):
            assert torch.equal(state["mock_parameter"], torch.tensor(1.))

    class MockTargets:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def load_state_dict(self, state):
            assert state == {}

    def load_cohort(path):
        assert json.loads((tmp_path / "selection.json").read_text()) == selection
        calls.append("cohort_loaded_after_selection")
        return SimpleNamespace()

    def collect(model, cohort, rows, batch_size, device, targets, **kwargs):
        assert len(rows) == 130
        assert model.prediction_head_status == status
        assert model.training_recurrence_probability == .25
        assert kwargs["report_concept_head_trained"] is False
        calls.append("selected_test_scored")
        return [{"query_order": 3} for _ in range(130)], {"patients": 130, "selection_nll": .51}

    monkeypatch.setattr(runner, "verify_source", lambda root, source: None)
    monkeypatch.setattr(timeline_config.TimelineConfig, "from_dict", staticmethod(lambda value: SimpleNamespace(**value)))
    monkeypatch.setattr(timeline_model, "TimelineModel", MockModel)
    monkeypatch.setattr(timeline_losses, "FixedCTMoments", MockTargets)
    monkeypatch.setattr(timeline_data.TimelineCohort, "load", staticmethod(load_cohort))
    monkeypatch.setattr(timeline_data, "split_indices", lambda cohort, split: {
        role: torch.arange(count) for role, count in runner.FIXED_COUNTS.items()})
    monkeypatch.setattr(timeline_training, "collect_predictions", collect)
    return SimpleNamespace(out=tmp_path, identity=identity, selection=selection,
                           export=export, calls=calls, run=run)


def test_only_frozen_winner_is_scored_once_and_epoch_zero_heads_stay_unavailable(selected_study):
    study = selected_study
    result = runner.evaluate_selected_test(study.out, study.identity, study.selection)
    assert result["selected_arm"] == "anchored_report"
    assert result["selected_step"] == 0
    assert result["used_for_selection"] is False
    assert study.calls == ["cohort_loaded_after_selection", "selected_test_scored"]
    assert runner.evaluate_selected_test(study.out, study.identity, study.selection) == result
    assert study.calls == ["cohort_loaded_after_selection", "selected_test_scored"]


def test_modified_selected_checkpoint_is_rejected_before_test_access(selected_study):
    study = selected_study
    changed = copy.deepcopy(study.export)
    changed["model_state"]["mock_parameter"] += 1
    torch.save(changed, study.run / "inference.pt")
    with pytest.raises(ValueError, match="does not match"):
        runner.evaluate_selected_test(study.out, study.identity, study.selection)
    assert study.calls == []


def test_existing_prediction_hash_mismatch_is_rejected_without_rescoring(selected_study):
    study = selected_study
    runner.evaluate_selected_test(study.out, study.identity, study.selection)
    (study.out / "selected_test_predictions.json").write_text("[]")
    with pytest.raises(ValueError, match="predictions changed"):
        runner.evaluate_selected_test(study.out, study.identity, study.selection)
    assert study.calls == ["cohort_loaded_after_selection", "selected_test_scored"]


def test_missing_arm_completion_is_rejected_before_test_access(selected_study):
    study = selected_study
    records = json.loads((study.out / "results.json").read_text())
    records.pop("compact_control")
    (study.out / "results.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="all arms"):
        runner.evaluate_selected_test(study.out, study.identity, study.selection)
    assert study.calls == []


@pytest.mark.parametrize("corruption", ["report_metadata", "head_status", "optimizer_steps"])
def test_inconsistent_epoch_zero_head_metadata_is_rejected_before_test_access(selected_study, corruption):
    study = selected_study
    changed = copy.deepcopy(study.export)
    if corruption == "report_metadata":
        changed["metadata"]["report_concept_head_trained"] = True
    elif corruption == "head_status":
        changed["head_training_status"]["pcr_head_trained"] = True
    else:
        changed["optimizer_steps"] = 1
    torch.save(changed, study.run / "inference.pt")
    torch.save(changed, study.run / "best.pt")
    study.selection["selected_inference_sha256"] = runner.file_sha256(study.run / "inference.pt")
    (study.out / "selection.json").write_text(json.dumps(study.selection))
    with pytest.raises(ValueError, match="training status"):
        runner.evaluate_selected_test(study.out, study.identity, study.selection)
    assert study.calls == []
