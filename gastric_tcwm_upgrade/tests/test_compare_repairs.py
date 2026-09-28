import importlib.util
from pathlib import Path

import pytest
import torch

from stageworld_tcwm.model import TreatmentBeliefWorld


spec = importlib.util.spec_from_file_location(
    "compare_repairs", Path(__file__).resolve().parents[1] / "scripts/compare_repairs.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_probe_respects_missing_observations_and_is_reproducible():
    present = torch.tensor([True, False, True, True, False])
    permutation = comparison.ct1_permutation(present, 17)
    assert torch.equal(permutation, comparison.ct1_permutation(present, 17))
    assert torch.equal(permutation[~present], torch.arange(5)[~present])
    assert (permutation[present] != torch.arange(5)[present]).all()
    assert torch.equal(permutation.sort().values, torch.arange(5))


def test_comparison_has_no_test_role_and_preserves_prior_boundary(cohort, config):
    model = TreatmentBeliefWorld(config)
    rows = torch.arange(8)
    model.fit_statistics(cohort.batch(rows))
    with pytest.raises(ValueError, match="restricted"):
        comparison.role_report(model, cohort, rows, "test", 2, 4, 17, 19)
    report = comparison.role_report(model, cohort, rows, "validation", 2, 4, 17, 19)
    assert report["stages"]["S0"]["n"] == 8
    assert report["pcr"]["n"] == 8
    assert report["ct1_permutation"]["observations_moved"] == 8
    assert report["ct1_permutation"]["stages"]["S0"]["maximum_absolute"] == 0
    assert report["ct1_permutation"]["pcr"]["maximum_absolute"] == 0
    assert report["ct1_permutation"]["stages"]["S1"]["maximum_absolute"] > 0


def test_comparison_rejects_wrong_split_contract(tmp_path):
    import json
    (tmp_path / "contract.json").write_text(json.dumps({
        "id": "run", "contract": {"cohort_sha256": "cohort", "split": "old"}}))
    with pytest.raises(ValueError, match="different cohort or patient split"):
        comparison.verify_contract({"contract_id": "run"}, tmp_path / "inference.pt",
                                   "cohort", "new", ["training"])
