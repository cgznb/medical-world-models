"""Synthetic measurements verify gating, not patient biology or causal effects."""
import copy

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.terminal_readiness import (CONCEPT_SCHEMA, concept_readiness,
                                               pcr_s1_mask, require_training_ready,
                                               validate_measured_concepts)


def measured():
    values = torch.full((3, 4, 1), float("nan"))
    values[:, :2, 0] = torch.tensor([[20., 12.], [30., 32.], [18., 14.]])
    observed = torch.isfinite(values)
    order = torch.full(values.shape, -1, dtype=torch.long)
    order[:, 0] = 0
    order[:, 1] = 1
    return {"schema": CONCEPT_SCHEMA, "ids": ["synthetic-a", "synthetic-b", "synthetic-c"],
            "names": ["ct_tumor_longest_diameter_mm"], "values": values,
            "observed": observed, "applicable": observed.clone(), "available_order": order,
            "occurred_at": torch.full_like(values, float("nan")),
            "available_at": torch.full_like(values, float("nan")),
            "provenance": {"ct_tumor_longest_diameter_mm": {
                "source_kind": "measured", "source_reference": "synthetic manual CT annotation fixture",
                "definition_verified": True, "units": "mm", "measurement": "CT"}}}


def test_pcr_requires_enabled_factual_nac_available_by_s1():
    t = modality_batch(7)
    t["modality_value"] = t["modality_value"].bool()
    t["modality_value"][1, 0, :] = False
    t["modality_known"][2, 0, 0] = False
    t["role"][3, 0] = 3
    t["phase"][4, 0] = 3
    t["scan_event_index"][5] = 0
    t["pcr_valid"][6] = False
    assert pcr_s1_mask(t).tolist() == [True, False, False, False, False, False, False]
    t["modality_value"][1, 0, 1] = True
    t["modality_known"][1, 0, 1] = True
    t["modality_applicable"][1, 0, 1] = True
    assert not pcr_s1_mask(t)[1]  # Disabled radiotherapy cannot create applicability.


def test_real_repeated_measurements_enable_only_contract_gate():
    payload = measured()
    indices = {"train": torch.tensor([0]), "validation": torch.tensor([1]), "test": torch.tensor([2])}
    result = concept_readiness(payload, payload["ids"], indices)
    assert result["training_ready"]
    assert result["supported_longitudinal_concepts"] == payload["names"]
    assert result["concepts"][payload["names"][0]]["train"]["by_stage"] == [1, 1, 0, 0]


@pytest.mark.parametrize("change", ["pseudo_source", "invalid_name", "unobserved_zero", "early_availability",
                                    "invalid_stage", "nonfinite_observed", "wrong_units", "bad_dates", "wrong_order"])
def test_invalid_concept_contract_is_rejected(change):
    p = copy.deepcopy(measured())
    if change == "pseudo_source":
        p["provenance"][p["names"][0]]["source_kind"] = "model_prediction"
    elif change == "invalid_name":
        p["names"] = ["micrometastatic_burden"]
    elif change == "unobserved_zero":
        p["values"][0, 3] = 0
    elif change == "early_availability":
        p["available_order"][0, 1] = 0
    elif change == "invalid_stage":
        p["applicable"][0, 2] = True
    elif change == "nonfinite_observed":
        p["values"][0, 0] = float("inf")
    elif change == "wrong_units":
        p["provenance"][p["names"][0]]["units"] = "cm"
    elif change == "bad_dates":
        p["occurred_at"][0, 0] = 10
        p["available_at"][0, 0] = 9
    elif change == "wrong_order":
        p["ids"].reverse()
    with pytest.raises(ValueError):
        validate_measured_concepts(p, measured()["ids"])


def test_absent_or_baseline_only_concepts_cannot_enable_b():
    indices = {"train": torch.tensor([0]), "validation": torch.tensor([1]), "test": torch.tensor([2])}
    p = measured()
    assert not concept_readiness(None, p["ids"], indices)["training_ready"]
    p["observed"][:, 1] = False
    p["values"][:, 1] = float("nan")
    p["available_order"][:, 1] = -1
    assert not concept_readiness(p, p["ids"], indices)["training_ready"]


def test_stage_gate_rejects_blocked_stages_and_invalid_names():
    report = {"stages": {"B": {"training_ready": True},
                         "C": {"training_ready": False, "blockers": ["No verified follow-up"]}}}
    require_training_ready(report, "B")
    with pytest.raises(ValueError, match="No verified follow-up"):
        require_training_ready(report, "C")
    with pytest.raises(ValueError, match="stage must be B or C"):
        require_training_ready(report, "fake")
