"""Training-only modality support, without name vocabularies or drug counts."""
from __future__ import annotations

import torch


def _action_signature(applicable, known, value):
    return ",".join(str(-1 if not a else (0 if not k else 1 + int(v)))
                    for a, k, v in zip(applicable, known, value))


def _phase_action_patients(batch):
    required = {"phase", "role", "operation"}
    if not required.issubset(batch):
        return None
    active = batch["event_mask"].bool() & ((batch["role"] == 0) | (batch["role"] == 2))
    active = active & (batch["operation"] != 4)
    values = {name: batch[name].detach().cpu().tolist() for name in
              ("phase", "modality_applicable", "modality_known", "modality_value")}
    counts = {}
    for patient, mask in enumerate(active.cpu().tolist()):
        seen = set()
        for event, present in enumerate(mask):
            if not present:
                continue
            signature = _action_signature(*(values[name][patient][event] for name in
                                            ("modality_applicable", "modality_known", "modality_value")))
            seen.add((str(values["phase"][patient][event]), signature))
        for phase, signature in seen:
            phase_counts = counts.setdefault(phase, {})
            phase_counts[signature] = phase_counts.get(signature, 0) + 1
    return counts


def fit_modality_support(batch):
    valid = batch["event_mask"].bool().unsqueeze(-1)
    applicable = batch["modality_applicable"].bool() & valid
    known = batch["modality_known"].bool() & applicable
    present = batch["modality_value"].bool() & known
    # Count patients, not repeated administrations of one modality.
    support = {
        "schema": "modality-support-v2",
        "fit_patients": int(len(valid)),
        "known_patients": known.any(1).sum(0).tolist(),
        "present_patients": present.any(1).sum(0).tolist(),
        "scope": "patient_level_modality_only",
    }
    phase_counts = _phase_action_patients(batch)
    if phase_counts is not None:
        support["phase_action_patients"] = phase_counts
        support["phase_action_scope"] = "training_factual_patient_counts_exact_status_pattern"
    return support


def modality_support_flags(batch, support):
    if support.get("schema") != "modality-support-v2":
        raise ValueError("Require modality support; legacy named support cannot be migrated")
    counts = torch.as_tensor(support["present_patients"], device=batch["event_mask"].device)
    if counts.shape != (7,) or (counts < 0).any():
        raise ValueError("Require seven fixed-ID modality counts")
    present = (batch["modality_value"].bool() & batch["modality_known"].bool()
               & batch["modality_applicable"].bool() & batch["event_mask"].bool().unsqueeze(-1))
    return {"unseen_modalities": present.any(1) & (counts == 0),
            "sparse_modalities": present.any(1) & (counts > 0) & (counts < 5)}


def phase_action_support_flags(batch, support):
    """Descriptive action-pattern support; this is not conditional causal overlap."""
    if support.get("schema") != "modality-support-v2":
        raise ValueError("Require modality-support-v2")
    active = batch["event_mask"].bool()
    counts = torch.zeros_like(active, dtype=torch.long)
    audited = "phase_action_patients" in support
    if audited:
        table = support["phase_action_patients"]
        values = {name: batch[name].detach().cpu().tolist() for name in
                  ("phase", "modality_applicable", "modality_known", "modality_value")}
        for patient, mask in enumerate(active.cpu().tolist()):
            for event, present in enumerate(mask):
                if not present:
                    continue
                signature = _action_signature(*(values[name][patient][event] for name in
                                                ("modality_applicable", "modality_known", "modality_value")))
                count = table.get(str(values["phase"][patient][event]), {}).get(signature, 0)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("Phase/action support requires nonnegative patient counts")
                counts[patient, event] = count
    return {"phase_action_counts": counts, "unsupported_events": active & (counts == 0),
            "sparse_events": active & (counts > 0) & (counts < 5), "support_audited": audited}
