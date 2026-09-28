import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_local", Path(__file__).resolve().parents[1] / "scripts/prepare_local.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_preserve_existing_patient_split():
    split = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    source = {"patient_ids": split, "pool_id": "pool1", "seed": 17}
    pool = {"ids": ["d", "c", "b", "a"], "artifact_id": "pool1"}
    assert adapter.normalize_split(source, pool) is split


def test_event_split_requires_matching_pool_provenance():
    split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    source = {"patient_ids": split, "pool_id": "event1"}
    pool = {"ids": ["a", "b", "c"], "artifact_id": "pool1"}
    events = {"artifact_id": "event1", "source_pool_id": "pool1"}
    assert adapter.normalize_split(source, pool, events) is split
    events["source_pool_id"] = "pool2"
    with pytest.raises(ValueError):
        adapter.normalize_split(source, pool, events)


@pytest.mark.parametrize("change", ["overlap", "missing", "foreign_pool"])
def test_reject_split_identity_failures(change):
    split = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    source = {"patient_ids": split, "pool_id": "pool1"}
    if change == "overlap":
        split["test"].append("a")
    elif change == "missing":
        split["test"] = []
    else:
        source["pool_id"] = "pool2"
    with pytest.raises(ValueError):
        adapter.normalize_split(source, {"ids": ["a", "b", "c", "d"], "artifact_id": "pool1"})
