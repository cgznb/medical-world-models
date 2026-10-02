"""Evidence-backed weak targets for the pre-resection S1 state, never inputs."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import torch

from .data import file_sha256, fingerprint
from .modality_schema import SCHEMA
from .terminal_readiness import pcr_s1_mask
from .timeline_data import TimelineCohort, split_indices

S1_CONCEPT_SCHEMA = "s1-report-concepts-v1"
S1_CONCEPT_NAMES = ("residual_viable_primary_tumour", "residual_viable_tumour_percent",
                    "nodes_positive", "primary_tumour_max_diameter_cm")
S1_CONCEPT_TRANSFORMS = ("present_1_absent_0", "divide_by_100", "log1p", "log1p")
REPORT_SCHEMA_SOURCE = Path(__file__).with_name("report_schema.py")


def load_report_schema():
    """Load the audited independent validator without importing its experiment."""
    if not REPORT_SCHEMA_SOURCE.is_file():
        raise ValueError("The approved independent pathology validator is unavailable")
    try:
        import jsonschema
    except ModuleNotFoundError:
        raise ValueError("Install stageworld-tcwm[timeline] to use the pathology validator") from None
    spec = importlib.util.spec_from_file_location("_audited_s1_report_schema", REPORT_SCHEMA_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance = {"path": str(REPORT_SCHEMA_SOURCE), "sha256": file_sha256(REPORT_SCHEMA_SOURCE),
                  "jsonschema_module": str(Path(jsonschema.__file__).resolve()),
                  "jsonschema_module_sha256": file_sha256(jsonschema.__file__)}
    return module, provenance


def extract_report_concepts(cohort, manifest, progress, load_record, validator):
    """Validate private evidence locally and return only target tensors/counts."""
    if (manifest.get("schema_version") != validator.SCHEMA_VERSION
            or manifest.get("source_pool_id") != cohort.metadata.get("source_pool_id")
            or manifest.get("patient_ids") != cohort.ids
            or manifest.get("role") != "training_only_auxiliary_supervision"
            or progress.get("manifest_id") != manifest.get("artifact_id")
            or progress.get("status") != "completed"
            or manifest.get("source_workbook", {}).get("column") != "AU"):
        raise ValueError("Structured report manifest/progress does not bind this cohort and source")
    source_rows = manifest.get("records", [])
    sources = {row["patient_id"]: row for row in source_rows}
    if len(source_rows) != len(cohort) or len(sources) != len(cohort) or set(sources) != set(cohort.ids):
        raise ValueError("Structured report source membership differs")
    if {"s1_concepts", "s1_concept_valid"} & set(cohort.tensors):
        raise ValueError("Source cohort already contains report concept targets")
    values = torch.full((len(cohort), len(S1_CONCEPT_NAMES)), float("nan"))
    valid = torch.zeros_like(values, dtype=torch.bool)
    statuses = {name: Counter() for name in S1_CONCEPT_NAMES}
    eligible_reports = 0
    for index, patient in enumerate(cohort.ids):
        if Path(patient).name != patient or patient in {".", ".."}:
            raise ValueError("Invalid private report join key")
        record = load_record(patient)
        if (record.get("manifest_id") != manifest["artifact_id"]
                or record.get("patient_id") != patient or record.get("status") != "completed"):
            raise ValueError("Structured report record binding or completion differs")
        clean, issues = validator.validate_result(record["result"], sources[patient]["report"])
        if issues or clean != record["result"]:
            raise ValueError("Structured report evidence validation failed")
        eligible = sources[patient].get("eligible") is True and clean["report_status"] == "valid"
        eligible_reports += int(eligible)
        for column, name in enumerate(S1_CONCEPT_NAMES):
            finding = clean["findings"][name]
            statuses[name][finding["status"]] += 1
            if not eligible or finding["status"] != "observed":
                continue
            value = finding["value"]
            if column == 0:
                value = {"present": 1.0, "absent": 0.0}[value]
            elif column == 1:
                value = value / 100.0
            else:
                value = math.log1p(value)
            values[index, column], valid[index, column] = value, True
    observed = valid.clone()
    # NAC applicability is independent of whether a separate pCR label exists.
    nac = pcr_s1_mask({**cohort.tensors, "pcr_valid": torch.ones(len(cohort), dtype=torch.bool)})
    valid[:, :2] &= nac[:, None]
    values[~valid] = float("nan")
    quality = {"records_validated": len(cohort), "eligible_reports": eligible_reports,
               "evidence_issues": 0, "source_status_counts": {name: dict(counts) for name, counts in statuses.items()},
               "source_observed_counts": observed.sum(0).tolist(), "applicable_counts": valid.sum(0).tolist(),
               "response_applicability": "factual_enabled_NAC_before_S1_independent_of_pcr_label_validity"}
    return {"s1_concepts": values, "s1_concept_valid": valid}, observed, quality


def validate_report_concepts(tensors, metadata):
    values, valid = tensors["s1_concepts"], tensors["s1_concept_valid"]
    contract = metadata["s1_report_concepts"]
    if (values.ndim != 2 or values.shape[1] != 4 or valid.shape != values.shape
            or not values.is_floating_point() or valid.dtype != torch.bool
            or not torch.isfinite(values[valid]).all() or not torch.isnan(values[~valid]).all()):
        raise ValueError("S1 report concepts require finite observed [N,4] targets and NaN missing targets")
    if (contract.get("schema") != S1_CONCEPT_SCHEMA or contract.get("schema_version") != 1
            or tuple(contract.get("names", ())) != S1_CONCEPT_NAMES
            or tuple(contract.get("transforms", ())) != S1_CONCEPT_TRANSFORMS
            or contract.get("validated_source") is not True
            or contract.get("weak_supervision") is not True
            or contract.get("clinically_adjudicated") is not False
            or contract.get("target_stage") != 1 or contract.get("available_stage") != 2
            or contract.get("forbidden_as_forward_input") is not True):
        raise ValueError("S1 report concept semantics/provenance differ from the weak supervision contract")
    binary = values[:, 0][valid[:, 0]]
    fraction = values[:, 1][valid[:, 1]]
    if not ((binary == 0) | (binary == 1)).all() or not ((fraction >= 0) & (fraction <= 1)).all():
        raise ValueError("S1 binary/fraction concepts violate their fixed scales")
    if (values[:, 2:][valid[:, 2:]] < 0).any():
        raise ValueError("Log1p specimen targets cannot be negative")


def attach_report_concepts(cohort, split, targets, observed, quality, provenance):
    indices = split_indices(cohort, split)
    coverage = {role: {"patients": len(rows), "source_observed": observed[rows].sum(0).tolist(),
                       "applicable": targets["s1_concept_valid"][rows].sum(0).tolist()}
                for role, rows in {"all": torch.arange(len(cohort)), **indices}.items()}
    metadata = deepcopy(cohort.metadata)
    metadata["s1_report_concepts"] = {
        "schema": S1_CONCEPT_SCHEMA, "schema_version": 1,
        "names": list(S1_CONCEPT_NAMES), "transforms": list(S1_CONCEPT_TRANSFORMS),
        "source_units": ["binary", "percent", "count", "cm"],
        "validated_source": True, "clinically_adjudicated": False, "weak_supervision": True,
        "target_stage": 1, "available_stage": 2, "verified_calendar_availability": False,
        "prediction_cutoff": "post_neoadjuvant_pre_resection_s1_prior",
        "measurement_scope": "pre_resection_disease_in_excised_surgical_specimen_not_postoperative_patient_residual_burden",
        "forbidden_as_forward_input": True, "no_derived_pcr_recurrence_or_ypTNM": True,
        "source": provenance, "quality": quality, "coverage": coverage,
        "weak_s1_report_pilot_ready": all(bool(targets["s1_concept_valid"][indices[role]].any())
                                           for role in ("train", "validation")),
        "longitudinal_concept_readiness": False, "causal_strategy_readiness": False,
    }
    tensors = {**cohort.tensors, **targets}
    validate_report_concepts(tensors, metadata)
    result = TimelineCohort({"schema": SCHEMA, "ids": list(cohort.ids), "tensors": tensors,
                             "metadata": metadata, "encoders": deepcopy(cohort.encoders)})
    if fingerprint(split) != result.metadata.get("split_sha256"):
        raise ValueError("Report preparation must preserve the source split fingerprint")
    return result
