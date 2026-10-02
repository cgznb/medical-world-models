"""Variable-event tensor contract; stage order and calendar days are distinct."""
from __future__ import annotations

import torch
from .data import atomic_save
from .modality_schema import SCHEMA, PHASES, OPERATIONS, ROLES, ROLE_ID, SCHEMA_ENABLED, status_ids

REQUIRED = ("ct0", "ct1", "clinical", "binary", "binary_valid", "pcr", "pcr_valid",
            "image_valid", "modality_value", "modality_known", "modality_applicable",
            "event_mask", "phase", "operation", "role", "event_order", "event_id",
            "time_features", "occurred_at", "available_at", "query_order", "query_mask",
            "scan_event_index")


class TimelineCohort:
    def __init__(self, payload):
        if payload.get("schema") != SCHEMA:
            raise ValueError("Require modality-event-v2; legacy caches cannot be silently loaded")
        self.ids = list(payload["ids"])
        self.tensors = dict(payload["tensors"])
        self.metadata = dict(payload["metadata"])
        self.encoders = dict(payload.get("encoders", {}))
        self.validate()

    def __len__(self):
        return len(self.ids)

    def validate(self):
        n, t = len(self), self.tensors
        if not n or len(set(self.ids)) != n or not all(isinstance(x, str) and x for x in self.ids):
            raise ValueError("Patient IDs must be unique nonempty strings")
        if set(REQUIRED) - set(t):
            raise ValueError(f"Missing timeline tensors: {set(REQUIRED) - set(t)}")
        forbidden = {"treatment", "drugs", "regimens", "text_embedding", "drug_embedding",
                     "unseen_treatment_names_count", "interval_days"}
        if set(t) & forbidden or set(self.encoders) - {"clinical", "fit_ids"}:
            raise ValueError("Timeline inputs cannot import old drug/scaler/support fields")
        if any(not isinstance(v, torch.Tensor) or v.ndim == 0 or len(v) != n for v in t.values()):
            raise ValueError("Timeline tensors require a consistent patient axis")
        basis = self.metadata.get("time_basis")
        if basis not in {"ordinal_stage", "calendar_days"}:
            raise ValueError("Declare one explicit time basis")
        if tuple(self.metadata.get("schema_enabled", SCHEMA_ENABLED)) != SCHEMA_ENABLED:
            raise ValueError("Unsupported modality schema mask")
        l = t["event_mask"].shape[1]
        q = t["query_mask"].shape[1]
        shapes = {"clinical": (n, 32), "image_valid": (n, 2), "binary": (n,),
                  "binary_valid": (n,), "pcr": (n,), "pcr_valid": (n,),
                  "scan_event_index": (n,), "time_features": (n, l, 6),
                  "query_mask": (n, q), "query_order": (n, q)}
        for key in ("modality_value", "modality_known", "modality_applicable"):
            shapes[key] = (n, l, 7)
        for key in ("event_mask", "phase", "operation", "role", "event_order", "event_id",
                    "occurred_at", "available_at"):
            shapes[key] = (n, l)
        if t["ct0"].ndim != 3 or t["ct0"].shape[1] != 27 or t["ct1"].shape != t["ct0"].shape:
            raise ValueError("CT must contain matching [N,27,D] feature sets")
        for name, shape in shapes.items():
            if t[name].shape != shape:
                raise ValueError(f"Invalid shape for {name}; expected {shape}")
        for name in ("event_mask", "query_mask", "binary_valid", "pcr_valid", "image_valid"):
            if t[name].dtype != torch.bool:
                raise ValueError(f"{name} must be bool")
        for name in ("phase", "operation", "role", "event_order", "event_id", "scan_event_index"):
            if t[name].dtype != torch.long:
                raise ValueError(f"{name} must be int64")
        status_ids(t["modality_value"], t["modality_known"], t["modality_applicable"])
        mask = t["event_mask"]
        if (t["modality_applicable"] & ~mask[..., None]).any():
            raise ValueError("PAD events cannot contain applicable modalities")
        for name, size in (("phase", len(PHASES)), ("operation", len(OPERATIONS)), ("role", len(ROLES))):
            if not ((t[name][mask] >= 0) & (t[name][mask] < size)).all():
                raise ValueError(f"Invalid {name} categorical ID")
        for name in ("ct0", "ct1", "clinical", "time_features", "query_order"):
            if not torch.isfinite(t[name]).all():
                raise ValueError(f"Nonfinite model input {name}")
        for name in ("binary", "pcr"):
            values = t[name][t[name + "_valid"]]
            if not ((values == 0) | (values == 1)).all():
                raise ValueError("Observed outcomes must be binary")
        if (t["scan_event_index"] < 0).any() or (t["scan_event_index"] > l).any():
            raise ValueError("Invalid pre-assimilation scan checkpoint")
        queries = t["query_order"][t["query_mask"]]
        if (queries != queries.long()).any() or (queries < 0).any() or (queries > l).any():
            raise ValueError("Query order counts consumed slots and must be integer 0..L")
        for index in range(n):
            orders = t["event_order"][index][mask[index]]
            if (orders <= 0).any() or (orders[1:] <= orders[:-1]).any():
                raise ValueError("Events must preserve strictly increasing causal order")
            patient_queries = t["query_order"][index][t["query_mask"][index]].long()
            positive_queries = patient_queries[patient_queries > 0]
            if not mask[index][positive_queries - 1].all():
                raise ValueError("Queries require a real event checkpoint, not a PAD slot")
        if basis == "ordinal_stage":
            if not torch.isnan(t["occurred_at"][mask]).all() or not torch.isnan(t["available_at"][mask]).all():
                raise ValueError("Ordinal stage indices must not be fabricated calendar dates")
        else:
            for name in ("occurred_at", "available_at"):
                if not torch.isfinite(t[name][mask]).all():
                    raise ValueError("Calendar training requires verified event and availability dates")
            observed = mask & ((t["role"] == ROLE_ID["observed_action"]) |
                               (t["role"] == ROLE_ID["retrospective_report"]))
            if (t["available_at"][observed] < t["occurred_at"][observed]).any():
                raise ValueError("Observed calendar events cannot be available before occurrence")
            for index in range(n):
                available = t["available_at"][index][mask[index]]
                if (available[1:] < available[:-1]).any():
                    raise ValueError("Calendar events must be ordered by information availability")
        return self

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def save(self, path):
        self.validate()
        atomic_save({"schema": SCHEMA, "ids": self.ids, "tensors": self.tensors,
                     "metadata": self.metadata, "encoders": self.encoders}, path)

    def batch(self, indices, device="cpu"):
        indices = torch.as_tensor(indices, dtype=torch.long)
        return {name: value[indices].to(device) for name, value in self.tensors.items()}


def split_indices(cohort, split):
    if (set(split) not in ({"train", "validation", "test"},
                          {"train", "validation", "outer_evaluation"}) or not all(split.values())):
        raise ValueError("Require train/validation/test or historical train/validation/outer_evaluation membership")
    flat = [patient for patients in split.values() for patient in patients]
    if len(flat) != len(set(flat)) or set(flat) != set(cohort.ids):
        raise ValueError("Timeline splits must disjointly cover the cohort")
    fit_ids = cohort.encoders.get("fit_ids", [])
    if len(fit_ids) != len(set(fit_ids)) or set(fit_ids) != set(split["train"]):
        raise ValueError("Clinical encoder must be fitted to exactly these training patients")
    lookup = {patient: index for index, patient in enumerate(cohort.ids)}
    return {role: torch.tensor([lookup[patient] for patient in patients], dtype=torch.long)
            for role, patients in split.items()}
