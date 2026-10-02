#!/usr/bin/env python
"""Run V3/V6 forward passes on synthetic tensors, without files or clinical claims."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from stageworld_tcwm.four_stage_model import FourStageModel
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import TimelineModel


def synthetic_batch(n=12):
    value = torch.zeros(n, 3, 7, dtype=torch.bool)
    applicable = torch.zeros_like(value)
    applicable[:, 0, [0, 2, 3, 4, 5]] = True
    applicable[:, 1, 6] = True
    applicable[:, 2, 0] = True
    value[:, 0, 0] = value[:, 1, 6] = value[:, 2, 0] = True
    return {
        "ct0": torch.randn(n, 27, 768), "ct1": torch.randn(n, 27, 768),
        "clinical": torch.randn(n, 32), "image_valid": torch.ones(n, 2, dtype=torch.bool),
        "binary": torch.arange(n).remainder(2).float(), "binary_valid": torch.ones(n, dtype=torch.bool),
        "pcr": torch.arange(n).remainder(2).float(), "pcr_valid": torch.ones(n, dtype=torch.bool),
        "modality_value": value, "modality_known": applicable.clone(), "modality_applicable": applicable,
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


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    batch = synthetic_batch()
    v3 = TimelineModel(TimelineConfig(objective="terminal_state_v1", s1_report_concepts=True))
    v3.fit_statistics(batch)
    v3.eval()
    v6 = FourStageModel(state_adapter_rank=2)
    v6.fit_statistics(batch)
    v6.configure_terminal_adaptation()
    v6.eval()
    with torch.no_grad():
        output3, output6 = v3(batch), v6(batch)
    report = {
        "synthetic_only": True, "trained_or_clinically_validated": False,
        "V3": {
            "parameters": sum(p.numel() for p in v3.parameters()),
            "state_shapes": [{"Z": list(s.z.shape), "M": list(s.memory.shape), "C": list(s.clinical.shape)}
                             for s in output3["checkpoint_states"]],
            "s1_concept_logits_shape": list(output3["s1_concept_logits"].shape),
        },
        "V6": {
            "parameters": sum(p.numel() for p in v6.parameters()),
            "terminal_trainable_parameters": sum(p.numel() for p in v6.parameters() if p.requires_grad),
            "states_shape": list(output6["states"].shape),
            "s3_logits_shape": list(output6["logits"].shape),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
