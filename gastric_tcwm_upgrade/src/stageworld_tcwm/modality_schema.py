"""Stable, versioned categorical vocabulary for treatment modalities."""
from __future__ import annotations

import math
import torch

SCHEMA = "modality-event-v2"
MODALITIES = ("chemotherapy", "radiotherapy", "immunotherapy", "targeted",
              "interventional", "hipec", "surgery")
PHASES = ("baseline", "neoadjuvant", "perioperative", "postoperative", "followup")
OPERATIONS = ("procedure", "treatment_start", "treatment_stop", "treatment_summary",
              "plan_update", "observation", "record_update")
ROLES = ("observed_action", "known_plan", "retrospective_report", "hypothetical")
MODALITY_ID = {name: index for index, name in enumerate(MODALITIES)}
PHASE_ID = {name: index for index, name in enumerate(PHASES)}
OPERATION_ID = {name: index for index, name in enumerate(OPERATIONS)}
ROLE_ID = {name: index for index, name in enumerate(ROLES)}
SCHEMA_ENABLED = (True, False, True, True, True, True, True)
STATUS_UNKNOWN, STATUS_ABSENT, STATUS_PRESENT = range(3)

# These are the audited grouped-reader columns, not the older AA/AB layout.
MODALITY_COLUMNS = (
    ("chemotherapy", "AE", 0, "\u5316\u7597(1=\u6709,0=\u65e0)"),
    ("immunotherapy", "AF", 2, "\u514d\u75ab(1=\u6709,2=\u65e0)"),
    ("targeted", "AG", 2, "\u9776\u5411(1=\u6709,2=\u65e0)"),
    ("interventional", "AK", 0, "\u4ecb\u5165(1=\u6709,0=\u65e0)"),
    ("hipec", "AL", 2, "HIPEC(1=\u6709,2=\u65e0)"),
)
POSTOPERATIVE_MAPPING = {
    "surgery": {"column": "AQ", "header": "\u662f\u5426\u884c\u80c3\u5207\u9664\u624b\u672f",
                "positive": [1], "negative": [0], "phase": "perioperative"},
    "chemotherapy": {"column": "CE", "header": "\u672f\u540e\u5316\u7597",
                     "positive": ["\u662f", "\u6709", "1", "1.0"],
                     "negative": ["\u5426", "\u65e0", "0", "0.0"],
                     "phase": "postoperative", "operation": "treatment_summary"},
    "occurred_at_column": None,
    "available_at_column": None,
    "source": "audited_event_data.prepare_pool",
}


def parse_modality_flag(value, data_type: str, negative_code: int):
    """Return 0/1/None without consulting treatment names or conflicts."""
    if data_type in {"e", "f"} or value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip() not in {"0", "1", "2"}:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number == 1:
        return 1
    return 0 if number == negative_code else None


def status_ids(value: torch.Tensor, known: torch.Tensor, applicable: torch.Tensor):
    if value.shape != known.shape or value.shape != applicable.shape or value.shape[-1] != 7:
        raise ValueError("Modality value/known/applicable must have matching [...,7] shapes")
    if any(t.dtype != torch.bool for t in (value, known, applicable)):
        raise ValueError("Modality value/known/applicable must be bool")
    if (value & ~known).any() or (known & ~applicable).any():
        raise ValueError("Unknown values must be zero; inapplicable fields cannot be known")
    enabled = torch.tensor(SCHEMA_ENABLED, device=value.device)
    if (applicable & ~enabled).any():
        raise ValueError("Radiotherapy has no audited source and is disabled")
    return known.long() + value.long()
