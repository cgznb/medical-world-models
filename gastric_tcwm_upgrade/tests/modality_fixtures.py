"""Anonymous synthetic timeline fixtures for engineering checks only."""
import torch


def modality_batch(n=2, image_dim=16):
    value = torch.zeros(n, 3, 7)
    known = torch.zeros(n, 3, 7, dtype=torch.bool)
    applicable = torch.zeros_like(known)
    applicable[:, 0, [0, 2, 3, 4, 5]] = True
    known[:, 0, [0, 2, 3, 4, 5]] = True
    value[:, 0, 0] = 1
    applicable[:, 1, 6] = True
    known[:, 1, 6] = True
    value[:, 1, 6] = 1
    applicable[:, 2, 0] = True
    known[:, 2, 0] = True
    value[:, 2, 0] = 1
    return {
        "ct0": torch.randn(n, 27, image_dim),
        "ct1": torch.randn(n, 27, image_dim),
        "clinical": torch.randn(n, 32),
        "image_valid": torch.ones(n, 2, dtype=torch.bool),
        "binary": torch.arange(n).remainder(2).float(),
        "binary_valid": torch.ones(n, dtype=torch.bool),
        "pcr": torch.arange(n).remainder(2).float(),
        "pcr_valid": torch.ones(n, dtype=torch.bool),
        "modality_value": value,
        "modality_known": known,
        "modality_applicable": applicable,
        "event_mask": torch.ones(n, 3, dtype=torch.bool),
        "phase": torch.tensor([1, 2, 3]).expand(n, -1).clone(),
        "operation": torch.tensor([3, 0, 3]).expand(n, -1).clone(),
        "role": torch.full((n, 3), 2, dtype=torch.long),
        "event_id": torch.tensor([1, 2, 3]).expand(n, -1).clone(),
        "event_order": torch.tensor([1, 2, 3]).expand(n, -1).clone(),
        "time_features": torch.tensor([0., 0., 0., 0., 0., 1.]).expand(n, 3, -1).clone(),
        "occurred_at": torch.full((n, 3), float("nan")),
        "available_at": torch.full((n, 3), float("nan")),
        "query_order": torch.tensor([0, 1, 2, 3]).expand(n, -1).clone(),
        "query_mask": torch.ones(n, 4, dtype=torch.bool),
        "scan_event_index": torch.ones(n, dtype=torch.long),
    }
