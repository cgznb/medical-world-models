import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixed_modality_runner", ROOT / "scripts" / "run_modality.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_default_uses_one_fixed_651_run():
    args = runner.parse_args(["--out", "/tmp/fixed-run"])
    assert args.protocol == "fixed651_712"
    assert args.data_root == runner.DEFAULT_DATA_ROOT
    assert args.arms == ["dynamic_h128"]
    assert args.folds == []
    with pytest.raises(SystemExit):
        runner.parse_args(["--out", "/tmp/fixed-run", "--folds", "0"])


def test_historical_folds_require_explicit_protocol_and_data():
    with pytest.raises(SystemExit):
        runner.parse_args(["--out", "/tmp/legacy-run", "--protocol", "legacy-nested-521"])
    args = runner.parse_args(["--out", "/tmp/legacy-run", "--protocol", "legacy-nested-521",
                              "--data-root", "/tmp/historical-data"])
    assert args.folds == [0, 1, 2]
    assert args.arms == list(runner.ARMS)


def test_fixed_data_enforces_counts_and_protocol(tmp_path, monkeypatch):
    cohort = SimpleNamespace(ids=list(range(651)), metadata={
        "protocol": "fixed651_712", "split_seed": 17, "split_ratio": [7, 1, 2]})
    (tmp_path / "split.json").write_text("{}")
    monkeypatch.setattr(runner.TimelineCohort, "load", lambda _: cohort)
    roles = {role: list(range(count)) for role, count in runner.FIXED_COUNTS.items()}
    monkeypatch.setattr(runner, "split_indices", lambda *_: roles)
    assert runner.validate_fixed_data(tmp_path) == runner.FIXED_COUNTS
    roles["train"] = list(range(521))
    with pytest.raises(ValueError, match="train456"):
        runner.validate_fixed_data(tmp_path)
    roles["train"] = list(range(456))
    cohort.metadata["split_seed"] = 18
    with pytest.raises(ValueError, match="seed17"):
        runner.validate_fixed_data(tmp_path)


def test_fixed_study_dispatch_and_contract(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "cohort.pt").write_bytes(b"stub cohort")
    (data / "split.json").write_text("{}")
    monkeypatch.setattr(runner, "validate_fixed_data", lambda _: dict(runner.FIXED_COUNTS))
    seen = []

    def train(cohort, split, model, config, out, source_root, source, **kwargs):
        seen.append((cohort, split, out, kwargs))
        return {"status": "pass", "test": {"patients": 130}}

    monkeypatch.setattr(runner, "train_timeline", train)
    out = tmp_path / "run"
    assert runner.main(["--data-root", str(data), "--out", str(out), "--device", "cpu"]) == 0
    assert len(seen) == 1
    assert seen[0][:3] == (data / "cohort.pt", data / "split.json", out / "dynamic_h128")
    protocol = json.loads((out / "study_protocol.json").read_text())
    assert protocol["protocol"] == "fixed651_712"
    assert protocol["split_counts"] == runner.FIXED_COUNTS
    assert protocol["historical_holdouts_repartitioned"] is True
    assert "original_holdout65_accessed" not in protocol
    assert "fold_hashes" not in protocol
    assert json.loads((out / "status.json").read_text())["total_jobs"] == 1
