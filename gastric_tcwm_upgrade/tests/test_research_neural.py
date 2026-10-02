"""Engineering controls use anonymous synthetic tensors, never study labels."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from modality_fixtures import modality_batch
from stageworld_tcwm.research_neural import (
    ResearchNeural, load_research_neural_export, terminal_eligible, train_research_neural,
)
from stageworld_tcwm.clinical import ClinicalAnchor


def batch(n=12):
    result = modality_batch(n=n, image_dim=8)
    result["modality_value"] = result["modality_value"].bool()
    return result


@pytest.mark.parametrize("experiment_id", ["N00", "N01", "N02", "N13"])
def test_initial_baseline_and_forbidden_future_invariance(experiment_id):
    torch.manual_seed(7)
    data = batch()
    model = ResearchNeural(experiment_id, image_dim=8, dropout=0.)
    baseline = model.fit_initial_state(data)
    model.eval()
    initial = model(data)
    if experiment_id != "N02":
        torch.testing.assert_close(initial.sigmoid(), torch.full((12,), .5))
    else:
        assert baseline["kind"] == "train_only_clinical_logistic_C1"
        torch.testing.assert_close(initial, model.clinical_linear(data["clinical"]).squeeze(-1))
        anchor = ClinicalAnchor()
        anchor.fit(data["clinical"], data["binary"], data["binary_valid"])
        torch.testing.assert_close(initial, anchor(data["clinical"]), rtol=2e-6, atol=2e-6)
    # Give the neural path nonzero influence so an accidental future-input edge
    # cannot be concealed by the exact-zero initialization.
    with torch.no_grad():
        model.risk_head.weight.fill_(.2)
    before = model(data)
    changed = {key: value.clone() for key, value in data.items()}
    changed["ct1"].fill_(1e6)
    changed["pcr"] = 1 - changed["pcr"]
    changed["binary"] = 1 - changed["binary"]
    changed["pcr_valid"].logical_not_()
    changed["report_concept_value"] = torch.randn(12, 4) * 1e6
    torch.testing.assert_close(before, model(changed), rtol=0, atol=0)
    single = torch.cat([model({key: value[i:i+1] for key, value in data.items()}) for i in range(12)])
    torch.testing.assert_close(before, single, atol=2e-6, rtol=2e-6)


def test_terminal_labels_exclude_incomplete_hypothetical_and_duplicate_paths():
    data = batch(6)
    assert terminal_eligible(data).all()
    data["event_mask"][0, 2] = False
    data["role"][1, 0] = 3
    data["role"][2, 1] = 1
    data["event_id"][3, 2] = data["event_id"][3, 1]
    data["phase"][4, 1] = 3
    assert terminal_eligible(data).tolist() == [False] * 5 + [True]
    with pytest.raises(ValueError, match="only exposes complete factual"):
        ResearchNeural("N13", image_dim=8)(data)


@pytest.mark.parametrize("experiment_id", ["N00", "N01", "N02", "N13"])
def test_zero_head_then_upstream_supervised_gradient(experiment_id):
    torch.manual_seed(8)
    data = batch()
    model = ResearchNeural(experiment_id, image_dim=8, dropout=0.)
    model.fit_initial_state(data)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    upstream = (model.clinical_encoder.weight if experiment_id == "N00"
                else model.image_encoder.weight)
    first = F.binary_cross_entropy_with_logits(model(data), data["binary"])
    first.backward()
    assert model.risk_head.weight.grad.abs().sum() > 0
    assert upstream.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    F.binary_cross_entropy_with_logits(model(data), data["binary"]).backward()
    assert upstream.grad.abs().sum() > 0
    if experiment_id == "N13":
        assert model.event_cell.weight_ih.grad.abs().sum() > 0


def test_dynamic_sequence_depends_on_action_and_prefix_is_future_invariant():
    torch.manual_seed(3)
    data = batch()
    model = ResearchNeural("N13", image_dim=8, dropout=0.).eval()
    model.fit_initial_state(data)
    states = model.event_states(data)
    changed = {key: value.clone() for key, value in data.items()}
    changed["modality_value"][:, 2, 0] = False
    revised = model.event_states(changed)
    torch.testing.assert_close(states[:, :3], revised[:, :3], rtol=0, atol=0)
    assert not torch.allclose(states[:, 3], revised[:, 3])
    # Unknown and confirmed absence retain distinct event encodings.
    changed["modality_known"][:, 2, 0] = False
    assert not torch.allclose(revised[:, 3], model.event_states(changed)[:, 3])


def test_export_checkpoint_roundtrip_train_only_statistics_and_no_test_access(tmp_path):
    torch.manual_seed(5)
    data = batch(16)
    data["ct0"][8:] += 1000
    data["clinical"][12:] = float("nan")
    data["ct0"][12:] = float("nan")
    cohort = SimpleNamespace(ids=[f"synthetic-{i}" for i in range(16)], tensors=data,
                             encoders={"fit_ids": [f"synthetic-{i}" for i in range(8)]})
    out = tmp_path / "run"
    result = train_research_neural("N13", cohort, list(range(8)), list(range(8, 12)), out,
                                  device="cpu", max_steps=2, diagnostic=True)
    selected, export = load_research_neural_export(out / "inference.pt")
    best, _ = load_research_neural_export(out / "best.pt")
    val = {key: value[8:12] for key, value in data.items()}
    torch.testing.assert_close(selected(val), best(val), rtol=0, atol=0)
    torch.testing.assert_close(selected.image_mean, data["ct0"][:8].mean((0, 1)))
    last = torch.load(out / "last.pt", weights_only=True)
    assert last["step"] == 2 and last["optimizer"]["state"]
    assert len(last["sampler_rng"]) and last["sampler_cursor"] > 0
    assert result["test_evaluated"] is False
    assert result["completed_steps"] == 2 and result["parameter_count"] < 100000
    assert not export["head_training_flags"]["treatment_strategy_interface_enabled"]
    assert result["train"]["n"] == 8 and result["validation"]["n"] == 4
    with pytest.raises(FileExistsError):
        train_research_neural("N13", cohort, list(range(8)), list(range(8, 12)), out,
                              device="cpu", max_steps=2, diagnostic=True)


def test_rejects_non_train_clinical_encoder_and_unmarked_short_run(tmp_path):
    data = batch()
    cohort = SimpleNamespace(ids=list(map(str, range(12))), tensors=data, encoders={"fit_ids": []})
    with pytest.raises(ValueError, match="shorter runs require"):
        train_research_neural("N00", cohort, list(range(8)), list(range(8, 12)), tmp_path / "a",
                              device="cpu", max_steps=2)
    with pytest.raises(ValueError, match="exactly this training"):
        train_research_neural("N00", cohort, list(range(8)), list(range(8, 12)), tmp_path / "b",
                              device="cpu", max_steps=2, diagnostic=True)
