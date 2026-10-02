"""Data and provenance boundaries of the parallel auxiliary/adapter comparison."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "four_stage_ablation_runner", Path(__file__).resolve().parents[1] / "scripts" / "run_four_stage_ablation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_test_rows_cannot_be_indexed_and_are_not_materialized():
    original = torch.tensor([[1.], [float("nan")], [3.], [4.]])
    value = runner.TrainValidationTensor(original, torch.tensor([3, 0, 2]), 4)
    assert value._values.shape == (3, 1)
    assert torch.isfinite(value._values).all()
    assert torch.equal(value[torch.tensor([2, 3, 0])], torch.tensor([[3.], [4.], [1.]]))
    for rows in ([1], [0, 1], [-1], [4]):
        with pytest.raises(ValueError, match="inaccessible|outside cohort"):
            value[rows]
    original[0] = 99
    assert value[[0]].item() == 1


def test_no_test_seed_subset_or_resume_cli(tmp_path):
    for extra in ("--test", "--resume", "--seed", "--device"):
        with pytest.raises(SystemExit):
            runner.parse_args(["--out", str(tmp_path), extra])
    args = runner.parse_args(["--out", str(tmp_path), "--diagnostic"])
    assert args.diagnostic
    assert runner.ARMS == ("pcr_only_frozen", "ct_pcr_adapter")


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    old = tmp_path / "history.json"
    old.write_text("original result")
    monkeypatch.setattr(runner, "make_protocol", lambda args: pytest.fail("Must reject before protocol creation"))
    with pytest.raises(FileExistsError, match="unsafe resume"):
        runner.bind_protocol(SimpleNamespace(out=tmp_path))
    assert old.read_text() == "original result"


def test_parent_receipt_rejects_changed_artifact_and_test_scoring(tmp_path):
    artifact = tmp_path / "aux_best.pt"
    artifact.write_bytes(b"parent checkpoint")
    receipt = {"test_evaluated": False, "artifacts_sha256": {"aux_best.pt": runner.file_sha256(artifact)}}
    runner.write(receipt, tmp_path / "completion.json")
    runner.verify_receipt(tmp_path / "completion.json", tmp_path)
    artifact.write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify_receipt(tmp_path / "completion.json", tmp_path)
    runner.write(dict(receipt, test_evaluated=True), tmp_path / "completion.json")
    with pytest.raises(ValueError, match="train/validation-only"):
        runner.verify_receipt(tmp_path / "completion.json", tmp_path)


def test_receipt_cannot_escape_its_experiment(tmp_path):
    runner.write({"test_evaluated": False, "artifacts_sha256": {"../other.pt": "abc"}},
                 tmp_path / "completion.json")
    with pytest.raises(ValueError, match="escapes its experiment"):
        runner.verify_receipt(tmp_path / "completion.json", tmp_path)


def test_written_metadata_is_private(tmp_path):
    path = tmp_path / "private.json"
    runner.write({"test_evaluated": False}, path)
    assert path.stat().st_mode & 0o777 == 0o600
    csv = tmp_path / "aggregate.csv"
    runner.write_csv([{"arm": "pcr_only_frozen", "seed": 17}], csv)
    assert csv.stat().st_mode & 0o777 == 0o600


def test_recipe_lock_cannot_promote_diagnostic_or_change_arms(monkeypatch):
    from stageworld_tcwm import four_stage_training as training
    protocol = {"source_root": str(runner.ROOT), "seeds": [17], "arms": list(runner.ARMS),
                "diagnostic": True, "recipe": runner.RECIPE,
                "training_defaults": training.DEFAULTS,
                "training_experiments": getattr(training, "EXPERIMENTS", {}),
                "test_policy": "disabled_no_test_entry_point", "test_evaluated": False,
                "split_counts": runner.FIXED_COUNTS, "data_sha256": {"split.json": runner.SPLIT_SHA256}}
    for altered in (dict(protocol, diagnostic=False), dict(protocol, test_evaluated=True),
                    dict(protocol, arms=[runner.ARMS[0]]), dict(protocol, recipe={}),
                    dict(protocol, test_policy="all_seeds_test")):
        with pytest.raises(ValueError, match="identity mismatch"):
            runner.verify_identity(altered)


def test_completion_aggregation_keeps_fallback_and_trained_results_separate(tmp_path, monkeypatch):
    train = {"n": 456, "positive": 103, "auc": .67, "nll": .50, "ap": .36, "brier": .16}
    valid = {"n": 65, "positive": 15, "auc": .724, "nll": .477, "ap": .55, "brier": .154}
    parent = tmp_path / "parent"
    baseline = parent / "runs" / "D01"
    baseline.mkdir(parents=True)
    runner.write({"train": train, "validation": valid}, baseline / "metrics.json")
    run_metrics = {"train": train, "validation": valid, "selected_epoch": 0,
                   "selected_kind": "clinical_baseline_at_terminal_step0", "completed_epochs": 50,
                   "parameter_count": 3700, "terminal_trainable_parameter_count": 201,
                   "auxiliary": {"selected_epoch": 11, "completed_epochs": 50, "completed_steps": 2850,
                                 "train": {"pcr_nll": .45}, "validation": {"pcr_nll": .49}}}
    monkeypatch.setattr(runner, "completed_metrics", lambda *args: run_metrics)
    for arm in runner.ARMS:
        run = tmp_path / "runs" / runner.job_key(arm, 17)
        run.mkdir(parents=True)
        runner.write([{"step": 0, "epoch": 0, "train": train, "validation": valid},
                      {"step": 57, "epoch": 1, "train": train,
                       "validation": dict(valid, nll=.479, auc=.70)},
                      {"step": 114, "epoch": 2, "train": train,
                       "validation": dict(valid, nll=.480, auc=.75)}], run / "history.json")
    summary = runner.summarize(tmp_path, {"diagnostic": True, "seeds": [17], "parent_root": str(parent)})
    assert not summary["test_evaluated"]
    assert summary["best_trained_is_diagnostic_only"]
    assert summary["arms"]["ct_pcr_adapter"]["trained_dynamics_selected"] == 0
    assert summary["arms"]["ct_pcr_adapter"]["metrics"]["best_trained_validation_nll"]["mean"] == .479
    import csv
    with (tmp_path / "per_seed_train_validation.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    adapter = next(row for row in rows if row["arm"] == "ct_pcr_adapter")
    assert adapter["best_trained_epoch"] == "1"
    assert adapter["auxiliary_actual_new_steps"] == "0"


def test_missing_twentieth_completion_blocks_validation_export_freeze(tmp_path, monkeypatch):
    seen = []
    def completed(out, arm, seed, protocol):
        seen.append((arm, seed))
        if arm == runner.ARMS[-1] and seed == runner.SEEDS[-1]:
            raise FileNotFoundError("Missing twentieth completion")
        return {"selected_step": 0}
    monkeypatch.setattr(runner, "completed_metrics", completed)
    monkeypatch.setattr(runner, "file_sha256", lambda path: "frozen")
    monkeypatch.setattr(runner, "load_train_validation", lambda *args: pytest.fail("Premature scoring"))
    with pytest.raises(FileNotFoundError, match="twentieth"):
        runner.freeze_and_reproduce(tmp_path, {"seeds": list(runner.SEEDS)})
    assert len(seen) == 20
    assert not (tmp_path / "validation_export_plan.json").exists()
