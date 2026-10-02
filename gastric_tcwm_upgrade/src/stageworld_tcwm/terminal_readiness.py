"""Aggregate readiness checks for measured concepts and causal strategy research."""
from __future__ import annotations

import math
from numbers import Real

import torch

from .data import fingerprint
from .modality_schema import MODALITIES, PHASE_ID, ROLE_ID, SCHEMA_ENABLED
from .timeline_data import split_indices

CONCEPT_SCHEMA = "measured-clinical-concepts-v1"
# Different measurement methods keep different names and cannot share targets.
CONCEPTS = {
    "ct_tumor_longest_diameter_mm": ("mm", "CT", (0, 1)),
    "ct_largest_node_short_axis_mm": ("mm", "CT", (0, 1, 2, 3)),
    "radiologic_t_stage": ("ordinal", "radiology", (0, 1)),
    "radiologic_n_stage": ("ordinal", "radiology", (0, 1)),
    "ecog_performance_status": ("ordinal", "clinical_assessment", (0, 1, 2, 3)),
    "pathologic_yp_t_stage": ("ordinal", "resection_pathology", (2,)),
    "pathologic_yp_n_stage": ("ordinal", "resection_pathology", (2,)),
}
CAUSAL_REQUIREMENTS = {
    "target_trial": "Eligibility, time zero, explicit complete strategies, endpoint, estimand and analysis plan.",
    "event_dates": "Verified occurrence and information-availability dates for each decision and measurement.",
    "followup": "Recurrence/death/censoring dates, endpoint definition and time-specific risk-set membership.",
    "decision_history": "Pretreatment and time-varying common causes measured before each treatment decision.",
    "strategy_support": "Action overlap conditional on relevant history, balance and effective sample size.",
    "selection": "Define cohort selection and include or account for progression, toxicity and failure to reach surgery.",
    "identification": "Document consistency, sequential exchangeability, positivity and censoring assumptions.",
    "validation": "Validated causal estimator, simulation checks and sensitivity analyses before treatment ranking.",
}


def pcr_s1_mask(tensors):
    """Require recorded NAC before S1; never infer negative pCR from no NAC."""
    t = tensors
    enabled = torch.tensor(SCHEMA_ENABLED, device=t["phase"].device)
    enabled[MODALITIES.index("surgery")] = False
    present = (t["modality_value"].bool() & t["modality_known"]
               & t["modality_applicable"] & enabled).any(-1)
    factual = ((t["role"] == ROLE_ID["observed_action"])
               | (t["role"] == ROLE_ID["retrospective_report"]))
    prefix = torch.arange(t["event_mask"].shape[1], device=present.device)[None, :]
    available = prefix < t["scan_event_index"][:, None]
    nac = t["event_mask"] & factual & available & (t["phase"] == PHASE_ID["neoadjuvant"])
    return (nac & present).any(-1) & t["pcr_valid"]


def validate_measured_concepts(payload, patient_ids):
    """Validate a genuine measurement sidecar; missing observations stay NaN."""
    required = {"schema", "ids", "names", "values", "observed", "applicable",
                "available_order", "occurred_at", "available_at", "provenance"}
    if set(payload) != required or payload.get("schema") != CONCEPT_SCHEMA:
        raise ValueError("Require the complete measured-clinical-concepts-v1 contract")
    if list(payload["ids"]) != list(patient_ids):
        raise ValueError("Concept sidecar must preserve the exact cohort patient order")
    names = tuple(payload["names"])
    if not names or len(set(names)) != len(names) or set(names) - set(CONCEPTS):
        raise ValueError("Unknown or duplicate measured concept names; latent and pseudo labels are forbidden")
    n, k = len(patient_ids), len(names)
    for key in ("values", "observed", "applicable", "available_order", "occurred_at", "available_at"):
        if not isinstance(payload[key], torch.Tensor) or payload[key].shape != (n, 4, k):
            raise ValueError("Concept tensors require matching [N,4,K] shapes")
    values, observed, applicable = (payload[key] for key in ("values", "observed", "applicable"))
    if observed.dtype != torch.bool or applicable.dtype != torch.bool:
        raise ValueError("Concept observed and applicable masks must be boolean")
    if not values.is_floating_point() or (observed & ~applicable).any():
        raise ValueError("Observed concepts require applicability and floating point values")
    if not torch.isfinite(values[observed]).all() or not torch.isnan(values[~observed]).all():
        raise ValueError("Observed concepts must be finite; unobserved concepts must stay NaN")
    order = payload["available_order"]
    stage = torch.arange(4, device=order.device)[:, None].expand(4, k)
    if order.dtype != torch.long or (order[~observed] != -1).any():
        raise ValueError("Concept available_order must be int64 with -1 for unobserved entries")
    if ((order < stage) | (order > 3))[observed].any():
        raise ValueError("Concept availability cannot precede its measurement stage")
    occurred, available = payload["occurred_at"], payload["available_at"]
    if not occurred.is_floating_point() or not available.is_floating_point():
        raise ValueError("Concept dates require floating point days or NaN")
    dated = torch.isfinite(occurred) & torch.isfinite(available)
    undated = torch.isnan(occurred) & torch.isnan(available)
    if not (dated | undated).all() or (dated & ~observed).any() or (available[dated] < occurred[dated]).any():
        raise ValueError("Concept dates must be verified pairs with availability after occurrence, or both NaN")
    if set(payload["provenance"]) != set(names):
        raise ValueError("Every measured concept requires its own provenance")
    for index, name in enumerate(names):
        units, measurement, stages = CONCEPTS[name]
        source = payload["provenance"][name]
        if (not isinstance(source, dict) or source.get("source_kind") != "measured" or source.get("definition_verified") is not True
                or not isinstance(source.get("source_reference"), str) or not source["source_reference"].strip()
                or source.get("units") != units or source.get("measurement") != measurement):
            raise ValueError("Concept provenance requires verified real measurement, units and source reference")
        invalid_stages = [slot for slot in range(4) if slot not in stages]
        if applicable[:, invalid_stages, index].any():
            raise ValueError("Concept measurement method is incompatible with an applicable stage")
        actual = values[:, :, index][observed[:, :, index]]
        if (actual < 0).any() or (units == "ordinal" and (actual != actual.long()).any()):
            raise ValueError("Concept values must satisfy their nonnegative measurement scale")
        maximum = {"radiologic_t_stage": 4, "radiologic_n_stage": 3,
                   "ecog_performance_status": 5, "pathologic_yp_t_stage": 4,
                   "pathologic_yp_n_stage": 3}.get(name)
        if maximum is not None and (actual > maximum).any():
            raise ValueError("Concept ordinal value exceeds its declared scale")
    return payload


def concept_readiness(payload, patient_ids, indices):
    if payload is None:
        return {"training_ready": False, "concepts": {},
                "blockers": ["No genuine longitudinal clinical concept sidecar is present."]}
    validate_measured_concepts(payload, patient_ids)
    counts = {}
    supported = []
    for index, name in enumerate(payload["names"]):
        mask = payload["observed"][:, :, index]
        counts[name] = {role: {"observations": int(mask[rows].sum()),
                              "patients_with_repeated_measurements": int((mask[rows].sum(1) >= 2).sum()),
                              "by_stage": mask[rows].sum(0).tolist()}
                        for role, rows in indices.items()}
        if all(counts[name][role]["patients_with_repeated_measurements"] > 0
               for role in ("train", "validation")):
            supported.append(name)
    return {"training_ready": bool(supported), "concepts": counts,
            "supported_longitudinal_concepts": supported,
            "blockers": [] if supported else ["No concept has repeated real measurements in both training and validation."],
            "scope": "Data-contract gate only; counts do not establish adequate power or clinical validity."}


def require_training_ready(report, stage):
    if stage not in {"B", "C"}:
        raise ValueError("Readiness stage must be B or C")
    result = report["stages"][stage]
    if result.get("training_ready") is not True:
        raise ValueError(f"Stage {stage} training blocked: " + "; ".join(result["blockers"]))


def audit_terminal_readiness(cohort, split, original, events, *, concepts=None):
    """Read cached artifacts and return counts only, without patient identities."""
    if (len(cohort.ids) != 651 or {role: len(ids) for role, ids in split.items()}
            != {"train": 456, "validation": 65, "test": 130}):
        raise ValueError("Readiness audit preserves fixed651_712 membership")
    indices = split_indices(cohort, split)
    if fingerprint(split) != cohort.metadata.get("split_sha256"):
        raise ValueError("Prepared cohort and fixed split fingerprint differ")
    if (set(original["ids"]) != set(cohort.ids)
            or set(events["patient_ids"]) != set(cohort.ids)
            or original["artifact_id"] != cohort.metadata.get("source_pool_id")
            or events["artifact_id"] != cohort.metadata.get("source_event_id")
            or events["source_pool_id"] != original["artifact_id"]):
        raise ValueError("Prepared and original source artifact bindings differ")
    t = cohort.tensors
    pcr_mask = pcr_s1_mask(t)
    nac_slots = t["event_mask"] & (t["phase"] == PHASE_ID["neoadjuvant"])
    modalities = [i for i, enabled in enumerate(SCHEMA_ENABLED) if enabled and MODALITIES[i] != "surgery"]
    absent = ((~t["modality_value"][:, :, modalities] & t["modality_known"][:, :, modalities]
               & t["modality_applicable"][:, :, modalities]).all(-1) & nac_slots).any(-1)
    complete_nac = (t["modality_known"][:, :, modalities] & t["modality_applicable"][:, :, modalities]).all(-1)
    combinations = {}
    for name, present_ids in (("chemotherapy_only", {0}), ("chemotherapy_and_immunotherapy_only", {0, 2})):
        expected = torch.tensor([index in present_ids for index in modalities], device=nac_slots.device)
        combinations[name] = ((t["modality_value"][:, :, modalities] == expected).all(-1)
                              & complete_nac & nac_slots).any(-1)
    surgery = ((t["modality_value"][:, :, 6] & t["modality_known"][:, :, 6]) & t["event_mask"]).any(-1)
    postop = (t["modality_value"][:, :, 0] & t["modality_known"][:, :, 0]
              & t["event_mask"] & (t["phase"] == PHASE_ID["postoperative"])).any(-1)
    partitions = {}
    for role, rows in {"all": torch.arange(len(cohort.ids)), **indices}.items():
        partitions[role] = {
            "patients": len(rows), "pcr_s1_applicable": int(pcr_mask[rows].sum()),
            "pcr_s1_positive": int((pcr_mask[rows] & (t["pcr"][rows] == 1)).sum()),
            "all_five_nac_absent_not_verified_direct_surgery": int(absent[rows].sum()),
            "surgery_present": int(surgery[rows].sum()), "postoperative_chemotherapy_present": int(postop[rows].sum()),
            "recorded_recurrence_targets": int(t["binary_valid"][rows].sum()),
            "ct0_targets": int(t["image_valid"][rows, 0].sum()),
            "ct1_targets": int(t["image_valid"][rows, 1].sum()),
            **{name: int(mask[rows].sum()) for name, mask in combinations.items()},
        }
    inventory = {name: {"shape": list(value.shape), "finite": int(torch.isfinite(value).sum()),
                        "elements": value.numel()} for name, value in sorted(t.items())}
    source_inventory = {source: {name: {"shape": list(value.shape), "finite": int(torch.isfinite(value).sum()),
                                      "elements": value.numel()}
                                for name, value in sorted(payload.items()) if isinstance(value, torch.Tensor)}
                        for source, payload in (("original_pool", original), ("event_cache", events))}
    fields = sorted({key for row in original["clinical"].values() for key in row})
    clinical = {name: {"nonmissing": sum(row.get(name) is not None for row in original["clinical"].values()),
                       "finite_numeric": sum(isinstance(row.get(name), Real) and math.isfinite(row[name])
                                             for row in original["clinical"].values())}
                for name in fields}
    event_mask = t["event_mask"]
    dates = {name: {"finite_real_events": int(torch.isfinite(t[name][event_mask]).sum()),
                    "real_events": int(event_mask.sum())} for name in ("occurred_at", "available_at")}
    causal = {"training_ready": False, "required_contract": CAUSAL_REQUIREMENTS,
              "blockers": [
                  "No reviewed target-trial, longitudinal confounder, follow-up or causal-validation contract is attached.",
                  "Cached stage order and legacy interval do not establish decision dates, risk sets or an endpoint horizon.",
                  "Records with all five NAC modalities absent are not verified direct-surgery paths; validation has no such record.",
                  "Surgery and postoperative chemotherapy have no observed absent comparator in this cohort.",
              ], "scope": "Action counts alone cannot establish conditional positivity or causal identification."}
    return {"schema": "terminal-bc-readiness-v1", "protocol": "fixed651_712",
            "split_sha256": fingerprint(split), "raw_workbook_read": False,
            "patient_identities_emitted": False, "partitions": partitions,
            "source_keys": {"original_pool": sorted(original), "event_cache": sorted(events),
                            "prepared_tensors": sorted(t)},
            "prepared_tensor_inventory": inventory, "source_tensor_inventory": source_inventory,
            "original_baseline_fields": clinical,
            "dates": dates, "original_interval": {"finite": int(torch.isfinite(original["interval"]).sum()),
                                                    "verified_calendar_dates": False},
            "observation_stages": {"S0": "CT0 and six baseline fields", "S1": "CT1 forecast target",
                                   "S2": "No cached observation", "S3": "No cached observation"},
            "pcr_scope": "Retrospective S1 response target with recorded NAC; specimen definition and availability still need verification.",
            "stages": {"B": concept_readiness(concepts, cohort.ids, indices), "C": causal}}
