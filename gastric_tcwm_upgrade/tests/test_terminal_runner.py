import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("terminal_runner", ROOT / "scripts" / "run_terminal.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_terminal_default_study_and_objectives():
    args = runner.parse_args(["--out", "/tmp/terminal-test"])
    assert args.parallel_jobs == 3
    assert args.arms == list(runner.ARMS)
    configs = runner.configurations(args)
    assert [value["training"]["alignment_weight"] for value in configs.values()] == [0, .01, .1]
    assert all(value["model"]["objective"] == "terminal_state_v1" for value in configs.values())
    with pytest.raises(SystemExit):
        runner.parse_args(["--out", "/tmp/terminal-test", "--arms", "terminal_control", "terminal_control"])


def test_arm_selection_uses_validation_and_ignores_test():
    records = {
        "terminal_control": {"validation": {"selection_nll": .5}, "test": {"selection_nll": .9}},
        "terminal_align001": {"validation": {"selection_nll": .6}, "test": {"selection_nll": .1}},
    }
    assert runner.select_validation_arm(records) == "terminal_control"
    records["terminal_align001"]["validation"]["selection_nll"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner.select_validation_arm(records)


def test_report_pilot_has_explicit_separate_suite_and_same_validation_rule():
    args = runner.parse_args(["--out", "/tmp/report-pilot", "--suite", "B-report-pilot"])
    assert args.data_root == runner.REPORT_DATA_ROOT
    assert args.arms == list(runner.REPORT_ARMS)
    configs = runner.configurations(args)
    assert all(value["model"]["s1_report_concepts"] for value in configs.values())
    assert [value["training"]["report_concept_weight"] for value in configs.values()] == [0, .01, .1]
    assert all(value["training"]["alignment_weight"] == .1 for value in configs.values())
    with pytest.raises(SystemExit):
        runner.parse_args(["--out", "/tmp/report-pilot", "--suite", "B-report-pilot", "--arms", "terminal_control"])


def test_optimization_candidates_share_frozen_validation_only_protocol():
    args = runner.parse_args(["--out", "/tmp/optimization-study", "--suite", "optimization-v1"])
    assert args.arms == list(runner.OPTIMIZATION_ARMS)
    assert args.data_root == runner.REPORT_DATA_ROOT
    configs = runner.configurations(args)
    for config in configs.values():
        assert config["model"]["capacity_profile"] == "compact_v1"
        assert config["model"]["objective"] == "terminal_state_v1"
        assert config["model"]["clinical_normalization"] == "continuous_only"
        assert config["training"]["evaluate_test"] is False
        assert config["training"]["validate_initial"] is True
        assert config["training"]["validation_interval"] == 10
        assert config["training"]["alignment_weight"] == 0
    assert configs["compact_control"]["model"]["terminal_clinical_anchor"] is False
    assert configs["anchored_compact"]["model"]["terminal_clinical_anchor"] is True
    assert configs["anchored_report"]["model"]["s1_report_concepts"] is True
    assert configs["anchored_report"]["training"]["report_concept_weight"] == .01


def test_selected_test_refuses_unfrozen_or_wrong_selection(tmp_path):
    import json

    identity = {"diagnostic": False, "suite": "optimization-v1",
                "arms": ["compact_control", "anchored_compact"]}
    selection = {"selected_arm": "anchored_compact"}
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    records = {
        "compact_control": {"validation": {"selection_nll": .5}},
        "anchored_compact": {"validation": {"selection_nll": .6}},
    }
    (tmp_path / "results.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="validation-selected winner"):
        runner.evaluate_selected_test(tmp_path, identity, selection)
    with pytest.raises(ValueError, match="frozen validation selection"):
        runner.evaluate_selected_test(tmp_path, identity, {"selected_arm": "compact_control"})
    with pytest.raises(ValueError, match="completed optimization"):
        runner.evaluate_selected_test(tmp_path, dict(identity, diagnostic=True), selection)
