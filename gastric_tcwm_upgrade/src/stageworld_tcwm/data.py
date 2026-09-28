"""Native tensor contract, patient splits, safe loading and atomic writes.

Four treatment tokens are descriptor groups, NOT four temporal treatment steps.
All supplied future treatments and intervals are explicit scenarios.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
from typing import Any
import torch

REQUIRED = ("ct0", "ct1", "image_valid", "clinical", "treatment", "interval_days", "surgery",
            "prefix_valid", "binary", "binary_valid", "pcr", "pcr_valid", "ct1_available_stage")


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


class Cohort:
    def __init__(self, payload: dict[str, Any]):
        if payload.get("schema") != "tcwm-cohort-v1":
            raise ValueError("Require tcwm-cohort-v1; do not relabel an existing cache")
        self.ids = list(payload["ids"])
        self.tensors = dict(payload["tensors"])
        self.metadata = dict(payload["metadata"])
        self.encoders = dict(payload.get("encoders", {}))
        self.validate()

    def __len__(self):
        return len(self.ids)

    def validate(self):
        n = len(self)
        t = self.tensors
        if not n or len(set(self.ids)) != n or not all(isinstance(x, str) for x in self.ids):
            raise ValueError("Patient identifiers must be nonempty and unique")
        if self.metadata.get("treatment_semantics") != "explicit_interval_scenario":
            raise ValueError("Treatment scenario semantics must be explicit")
        if not self.metadata.get("endpoint_definition"):
            raise ValueError("Document the endpoint definition")
        if any(k not in t for k in REQUIRED):
            raise ValueError(f"Missing tensors: {set(REQUIRED) - set(t)}")
        for k, v in t.items():
            if not isinstance(v, torch.Tensor) or v.ndim == 0 or len(v) != n:
                raise ValueError(f"Tensor {k} has an invalid patient axis")
        if t["ct0"].ndim != 3 or t["ct0"].shape[1] != 27 or t["ct1"].shape != t["ct0"].shape:
            raise ValueError("Use [N,27,D] CT feature grids")
        shapes = {"image_valid": (n,2), "clinical": (n,32), "treatment": (n,4,82),
                  "interval_days": (n,), "surgery": (n,), "prefix_valid": (n,3),
                  "binary": (n,), "binary_valid": (n,), "pcr": (n,), "pcr_valid": (n,),
                  "ct1_available_stage": (n,)}
        for k, shape in shapes.items():
            if tuple(t[k].shape) != shape:
                raise ValueError(f"{k}: expected {shape}, got {tuple(t[k].shape)}")
        for k in ("image_valid", "prefix_valid", "binary_valid", "pcr_valid"):
            if t[k].dtype != torch.bool:
                raise ValueError(f"{k} must be bool")
        for k in ("surgery", "ct1_available_stage"):
            if t[k].dtype != torch.long:
                raise ValueError(f"{k} must be int64")
        if not ((t["surgery"] >= 0) & (t["surgery"] <= 3)).all():
            raise ValueError("Surgery must use absent/present/unknown/conflict = 0/1/2/3")
        if not ((t["ct1_available_stage"] >= 1) & (t["ct1_available_stage"] <= 3)).all():
            raise ValueError("CT1 cannot be observed at baseline; 3 means unavailable")
        for k in ("ct0", "ct1", "clinical", "treatment", "interval_days"):
            if not torch.isfinite(t[k]).all():
                raise ValueError(f"Nonfinite {k}; store missing observations as zero plus mask")
        if not (t["interval_days"] > 0).all():
            raise ValueError("CT query interval must be positive, never a fabricated surgery interval")
        for name in ("binary", "pcr"):
            v = t[name][t[name + "_valid"]]
            if not ((v == 0) | (v == 1)).all():
                raise ValueError(f"Valid {name} labels must be 0/1")
        if any(k in t for k in ("time", "event", "entry")):
            if not all(k in t for k in ("time", "event", "entry")):
                raise ValueError("Survival needs time, event and stage-specific entry")
            if not self.metadata.get("time_origin") or self.metadata.get("time_unit") != "months":
                raise ValueError("Declare survival time origin and unit=months")
            if t["time"].shape != (n,) or t["event"].shape != (n,) or t["entry"].shape != (n,3):
                raise ValueError("Invalid survival dimensions")
            if t["event"].dtype != torch.long or not ((t["event"] >= 0) & (t["event"] <= 2)).all():
                raise ValueError("event: 0 censoring, 1 recurrence, 2 competing death")
            if not torch.isfinite(t["time"]).all() or not torch.isfinite(t["entry"]).all():
                raise ValueError("Nonfinite survival time")
            if not (t["entry"] >= 0).all() or not (t["time"] > 0).all():
                raise ValueError("Invalid nonnegative follow-up times")
            bad = (t["time"][:,None] <= t["entry"]) & t["prefix_valid"]
            if bad.any():
                raise ValueError("A valid landmark must precede the event/censoring time")
        if "post" in t:
            if t["post"].ndim != 3 or "post_mask" not in t or t["post_mask"].shape != t["post"].shape[:2]:
                raise ValueError("post requires [N,L,D] features and [N,L] post_mask")
            if t["post_mask"].dtype != torch.bool or not torch.isfinite(t["post"]).all():
                raise ValueError("Invalid postoperative observation")
            if self.metadata.get("post_available_stage") != 2:
                raise ValueError("This three-stage version only reads postoperative observations at S2")
        return self

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def save(self, path):
        atomic_save({"schema": "tcwm-cohort-v1", "ids": self.ids,
                     "tensors": self.tensors, "metadata": self.metadata, "encoders": self.encoders}, path)

    def batch(self, indices, device="cpu"):
        indices = torch.as_tensor(indices, dtype=torch.long)
        return {k: v[indices].to(device) for k,v in self.tensors.items()}


def split_indices(cohort, split):
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Explicit train/validation/test patient lists are required")
    flat = [p for role in split for p in split[role]]
    if len(flat) != len(set(flat)) or set(flat) != set(cohort.ids):
        raise ValueError("Splits must be disjoint and exactly cover the cohort")
    if not all(split.values()):
        raise ValueError("Empty patient split")
    lookup = {p:i for i,p in enumerate(cohort.ids)}
    return {role: torch.tensor([lookup[p] for p in ids], dtype=torch.long) for role,ids in split.items()}


def make_split(ids, labels, seed=17):
    """One patient-level split, not an external-validation substitute."""
    from sklearn.model_selection import train_test_split
    import numpy as np
    y = np.asarray(labels)
    idx = np.arange(len(ids))
    def safe_strata(values):
        _, counts = np.unique(values, return_counts=True)
        return values if len(counts) > 1 and counts.min() >= 2 else None
    train, other = train_test_split(idx, test_size=0.3, random_state=seed, stratify=safe_strata(y))
    val, test = train_test_split(other, test_size=0.5, random_state=seed + 1, stratify=safe_strata(y[other]))
    return {k:[ids[i] for i in v] for k,v in (("train",train),("validation",val),("test",test))}
