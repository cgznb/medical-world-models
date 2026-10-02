"""Safety and audit boundaries for the all-seed four-stage coordinator."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "four_stage_runner", Path(__file__).resolve().parents[1] / "scripts" / "run_four_stage.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_no_seed_subset_reordering_or_diagnostic_promotion():
    runner.require_seed_set(runner.SEEDS)
    runner.require_seed_set([17], diagnostic=True)
    for seeds in (runner.SEEDS[:-1], list(reversed(runner.SEEDS)), [17] * 10, [17]):
        with pytest.raises(ValueError, match="complete frozen"):
            runner.require_seed_set(seeds)
    with pytest.raises(ValueError, match="Diagnostics cannot"):
        runner.freeze_exports(Path("/unused"), {"diagnostic": True})


def test_identity_rejects_source_data_and_split_changes(tmp_path, monkeypatch):
    design = tmp_path / "design.md"
    design.write_text("frozen design")
    (tmp_path / "cohort.pt").write_bytes(b"frozen data")
    (tmp_path / "split.json").write_bytes(b"frozen split")
    split_digest = runner.file_sha256(tmp_path / "split.json")
    monkeypatch.setattr(runner, "SPLIT_SHA256", split_digest)
    monkeypatch.setattr(runner, "source_manifest", lambda root: {"src/model.py": "frozen-source"})
    monkeypatch.setattr(runner, "verify_source", lambda root, expected: (
        None if expected == {"src/model.py": "frozen-source"} else (_ for _ in ()).throw(ValueError("Source changed"))))
    protocol = {"seeds": list(runner.SEEDS), "diagnostic": False,
                "source_root": str(runner.ROOT), "source_sha256": {"src/model.py": "frozen-source"},
                "design_path": str(design), "design_sha256": runner.file_sha256(design),
                "data_root": str(tmp_path), "split_counts": runner.FIXED_COUNTS,
                "data_sha256": {name: runner.file_sha256(tmp_path / name) for name in ("cohort.pt", "split.json")}}
    runner.verify_identity(protocol)
    with pytest.raises(ValueError, match="Source changed"):
        runner.verify_identity(dict(protocol, source_sha256={"src/model.py": "new-source"}))
    with pytest.raises(ValueError, match="Incorrect fixed"):
        runner.verify_identity(dict(protocol, split_counts={"train": 521, "validation": 65, "test": 65}))
    (tmp_path / "cohort.pt").write_bytes(b"modified data")
    with pytest.raises(ValueError, match="Frozen data"):
        runner.verify_identity(protocol)


def test_evaluation_batch_excludes_future_targets_and_preserves_rows():
    tensors = {name: torch.arange(4).float().unsqueeze(1) for name in runner.INPUT_FIELDS}
    tensors.update(binary=torch.tensor([0., 1., 1., 0.]), binary_valid=torch.ones(4, dtype=torch.bool),
                   ct1=torch.randn(4, 27, 4), pcr=torch.ones(4), pcr_valid=torch.ones(4, dtype=torch.bool),
                   s1_report_concepts=torch.randn(4, 6))
    cohort = SimpleNamespace(tensors=tensors)
    rows = torch.tensor([3, 1])
    batch = runner.safe_evaluation_batch(cohort, rows, "cpu")
    assert set(batch) == set(runner.INPUT_FIELDS) | {"binary", "binary_valid", "row_index"}
    assert torch.equal(batch["row_index"], rows)
    assert torch.equal(batch["binary"], torch.tensor([0., 1.]))
    tensors["binary_valid"][3] = False
    with pytest.raises(ValueError, match="fixed denominator"):
        runner.safe_evaluation_batch(cohort, rows, "cpu")


def test_predictions_require_exact_row_label_alignment_and_valid_probabilities():
    cohort = SimpleNamespace(tensors={"binary": torch.tensor([0., 1., 0., 1.])})
    indices = torch.tensor([2, 0, 3, 1])
    prediction = {"probability": torch.tensor([.1, .2, .8, .9]), "labels": torch.tensor([0., 0., 1., 1.]),
                  "row_index": indices}
    metrics, _ = runner.validate_predictions(prediction, cohort, indices)
    assert metrics["auc"] == 1 and metrics["n"] == 4 and metrics["positive"] == 2
    # Swapping two rows with identical labels must also be detected.
    with pytest.raises(ValueError, match="rows or labels"):
        runner.validate_predictions(dict(prediction, row_index=torch.tensor([0, 2, 3, 1])), cohort, indices)
    with pytest.raises(ValueError, match="rows or labels"):
        runner.validate_predictions(dict(prediction, labels=torch.tensor([1., 1., 0., 0.])), cohort, indices)
    with pytest.raises(ValueError, match="Invalid recurrence"):
        runner.validate_predictions(dict(prediction, probability=torch.tensor([.1, .2, float("nan"), .9])), cohort, indices)


def make_ready_plan(tmp_path):
    jobs = {runner.run_key(seed): {} for seed in runner.SEEDS}
    jobs["D01"] = {}
    plan = {"jobs": jobs}
    runner.write(plan, tmp_path / "evaluation_plan.json")
    reproduction = {"evaluation_plan_sha256": runner.file_sha256(tmp_path / "evaluation_plan.json"),
                    "jobs": {key: {"pass": True} for key in jobs}}
    runner.write(reproduction, tmp_path / "validation_reproduction.json")
    return {"diagnostic": False, "seeds": list(runner.SEEDS)}, plan, reproduction


def test_test_gate_requires_every_frozen_export_and_validation_reproduction(tmp_path):
    protocol, plan, reproduction = make_ready_plan(tmp_path)
    runner.require_test_ready(tmp_path, protocol, plan)
    reproduction["jobs"].pop(runner.run_key(307))
    runner.write(reproduction, tmp_path / "validation_reproduction.json")
    with pytest.raises(ValueError, match="Every validation export"):
        runner.require_test_ready(tmp_path, protocol, plan)
    plan["jobs"].pop(runner.run_key(307))
    runner.write(plan, tmp_path / "evaluation_plan.json")
    with pytest.raises(ValueError, match="all ten seeds"):
        runner.require_test_ready(tmp_path, protocol, plan)
    with pytest.raises(ValueError, match="never score test"):
        runner.require_test_ready(tmp_path, dict(protocol, diagnostic=True), plan)


def test_no_test_predictor_call_if_validation_has_not_reproduced(tmp_path, monkeypatch):
    protocol, plan, reproduction = make_ready_plan(tmp_path)
    reproduction["jobs"][runner.run_key(43)]["pass"] = False
    runner.write(reproduction, tmp_path / "validation_reproduction.json")
    calls = []
    monkeypatch.setattr(runner, "predict_export", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Every validation export"):
        runner.score_test(tmp_path, protocol, plan, None, torch.arange(130))
    assert not calls


def test_full_training_comparison_disables_test_even_after_validation_passes(tmp_path, monkeypatch):
    args = runner.parse_args(["--out", str(tmp_path), "--mode", "weak_four_stage",
                              "--train-validation-only"])
    assert args.train_validation_only and not args.diagnostic
    protocol, plan, _ = make_ready_plan(tmp_path)
    protocol.update(train_validation_only=True, test_policy="disabled")
    calls = []
    monkeypatch.setattr(runner, "predict_export", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Train/validation-only"):
        runner.score_test(tmp_path, protocol, plan, None, torch.arange(130))
    assert not calls


def test_incomplete_tenth_run_blocks_freezing_even_if_nine_are_complete(tmp_path, monkeypatch):
    checked = []

    def completed(out, key, protocol):
        checked.append(key)
        return None if key == runner.run_key(307) else {"selected_step": 10, "selected_kind": "trained"}

    monkeypatch.setattr(runner, "completed_metrics", completed)
    monkeypatch.setattr(runner, "file_sha256", lambda path: "frozen-test-artifact")
    with pytest.raises(ValueError, match="All ten seeds"):
        runner.freeze_exports(tmp_path, {"diagnostic": False, "seeds": list(runner.SEEDS)})
    assert checked == [runner.run_key(seed) for seed in runner.SEEDS]
    assert not (tmp_path / "evaluation_plan.json").exists()


def test_existing_completed_artifacts_cannot_be_silently_reused_after_modification(tmp_path):
    protocol = {"diagnostic": False}
    runner.write(protocol, tmp_path / "study_protocol.json")
    directory = tmp_path / "runs" / runner.run_key(17)
    directory.mkdir(parents=True)
    artifact = directory / "inference.pt"
    artifact.write_bytes(b"selected checkpoint")
    receipt = {"job": runner.run_key(17), "protocol_sha256": runner.file_sha256(tmp_path / "study_protocol.json"),
               "diagnostic": False, "test_evaluated": False,
               "artifacts_sha256": {"inference.pt": runner.file_sha256(artifact)}}
    runner.write(receipt, directory / "completion.json")
    artifact.write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="Completed training artifact changed"):
        runner.completed_metrics(tmp_path, runner.run_key(17), protocol)


def test_interrupted_training_is_preserved_before_restart(tmp_path):
    directory = tmp_path / "runs" / runner.run_key(17)
    directory.mkdir(parents=True)
    (directory / "last.pt").write_bytes(b"partial training")
    runner.preserve_incomplete_run(tmp_path, runner.run_key(17))
    assert not directory.exists()
    archived = list((tmp_path / "interrupted_attempts").glob("*/last.pt"))
    assert len(archived) == 1 and archived[0].read_bytes() == b"partial training"


def test_epoch_protocol_distinguishes_formal_budget_from_two_step_diagnostic():
    defaults = {"batch_size": 8, "auxiliary_max_epochs": 100, "terminal_max_epochs": 100,
                "warmup_epochs": 30, "patience_epochs": 20, "validation_interval_epochs": 1}
    schedule = runner.training_schedule(defaults)
    assert schedule["training_patients"] == 456
    assert schedule["steps_per_epoch"] == 57
    assert schedule["maximum_steps"] == {"auxiliary": 5700, "terminal": 5700}
    assert schedule["early_stopping_count_starts_at_epoch"] == 31
    assert schedule["earliest_early_stopping_epoch"] == 50
    assert schedule["validation_interval_epochs"] == 1
    assert schedule["drop_last"] is False
    diagnostic = runner.training_schedule(defaults, diagnostic=True)
    assert diagnostic["maximum_steps"] == {"auxiliary": 2, "terminal": 2}
    assert diagnostic["diagnostic"] is True


def test_training_receipt_requires_consistent_epoch_counts_for_both_phases(tmp_path, monkeypatch):
    protocol = {"diagnostic": False, "training_schedule": {"steps_per_epoch": 57}}
    runner.write(protocol, tmp_path / "study_protocol.json")
    run = tmp_path / "runs" / runner.run_key(17)
    run.mkdir(parents=True)
    binary = {"auc": .6, "nll": .5, "ap": .3, "brier": .17, "n": 65, "positive": 15}
    metrics = {"seed": 17, "train": binary, "validation": binary, "test_evaluated": False,
               "selected_step": 57, "selected_epoch": 1, "selected_kind": "trained",
               "completed_steps": 2850, "completed_epochs": 50, "steps_per_epoch": 57,
               "stop_reason": "early_stopping",
               "auxiliary": {"selected_step": 114, "selected_epoch": 2,
                             "completed_steps": 2850, "completed_epochs": 50,
                             "stop_reason": "early_stopping"}}
    monkeypatch.setattr(runner, "verify_identity", lambda protocol: None)
    runner.write(metrics, run / "metrics.json")
    runner.run_receipt(tmp_path, runner.run_key(17), ("metrics.json",), protocol)
    assert (run / "completion.json").exists()
    for phase in (metrics, metrics["auxiliary"]):
        phase["completed_epochs"] = 49
        runner.write(metrics, run / "metrics.json")
        with pytest.raises(ValueError, match="Invalid training completed_epochs"):
            runner.run_receipt(tmp_path, runner.run_key(17), ("metrics.json",), protocol)
        phase["completed_epochs"] = 50
    metrics["steps_per_epoch"] = 15
    runner.write(metrics, run / "metrics.json")
    with pytest.raises(ValueError, match="steps_per_epoch differs"):
        runner.run_receipt(tmp_path, runner.run_key(17), ("metrics.json",), protocol)
