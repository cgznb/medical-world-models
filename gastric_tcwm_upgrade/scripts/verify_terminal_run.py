#!/usr/bin/env python
"""Check a completed terminal run using training patients only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stageworld_tcwm.data import write_json
from stageworld_tcwm.modality_inference import ModalityPredictor
from stageworld_tcwm.timeline_data import TimelineCohort, split_indices
from stageworld_tcwm.timeline_model import EVENT_FIELDS


@torch.inference_mode()
def verify(run, data_root, device):
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    predictor = ModalityPredictor.load(run / "inference.pt", device)
    exported = torch.load(run / "inference.pt", map_location="cpu", weights_only=True)
    world_trained = exported.get("head_training_status", {}).get("world_model_trained", True)
    if predictor.model.cfg.objective != "terminal_state_v1":
        raise ValueError("Require a terminal_state_v1 export")
    cohort = TimelineCohort.load(data_root / "cohort.pt")
    split = json.loads((data_root / "split.json").read_text())
    train_indices = split_indices(cohort, split)["train"]
    indices = train_indices[:8]
    batch = cohort.batch(indices, device)
    full = predictor.model(batch)
    assert not full["query_mask"][:, :3].any()
    assert full["query_mask"][:, 3].all()
    independent = {key: value for key, value in batch.items() if key not in
                   {"ct1", "binary", "pcr", "s1_concepts", "s1_concept_valid"}}
    clean = predictor.model(independent)
    torch.testing.assert_close(clean["logits"], full["logits"], rtol=0, atol=1e-6)
    torch.testing.assert_close(clean["pcr_logits"], full["pcr_logits"], rtol=0, atol=1e-6)
    has_concepts = "s1_concept_logits" in full
    if has_concepts:
        torch.testing.assert_close(clean["s1_concept_logits"], full["s1_concept_logits"], rtol=0, atol=1e-6)
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    changed["ct1"] = torch.full_like(changed["ct1"], 1e6)
    changed["modality_value"][:, 1:, :] = 0
    future_changed = predictor.model(changed)
    s1_future_difference = float((future_changed["pcr_logits"] - full["pcr_logits"]).abs().max())
    assert s1_future_difference <= 1e-6
    if has_concepts:
        torch.testing.assert_close(future_changed["s1_concept_logits"], full["s1_concept_logits"], rtol=0, atol=1e-6)
    truncated = dict(independent)
    for name in EVENT_FIELDS:
        truncated[name] = batch[name][:, :1]
    for name in list(truncated):
        if name.startswith("query_"):
            del truncated[name]
    truncated["query_order"] = torch.ones((len(indices), 1), device=device, dtype=torch.long)
    truncated["query_mask"] = torch.ones_like(truncated["query_order"], dtype=torch.bool)
    prefix = predictor.model(truncated)
    assert not prefix["query_mask"].any()
    pcr_difference = float((prefix["pcr_logits"] - full["pcr_logits"]).abs().max())
    assert pcr_difference <= 1e-5
    if has_concepts:
        torch.testing.assert_close(prefix["s1_concept_logits"], full["s1_concept_logits"], rtol=0, atol=1e-5)
    rollouts = {}
    for stage in (0, 1, 2):
        history = dict(independent)
        future = {}
        for name in EVENT_FIELDS:
            history[name] = batch[name][:, :stage]
            future[name] = batch[name][:, stage:]
        if not world_trained:
            try:
                predictor.predict_strategy(history, future, strategy_name="untrained_world_rejection")
            except ValueError:
                rollouts[f"S{stage}"] = {"untrained_world_rejected": True}
            else:
                raise AssertionError("An untrained world model must not expose strategy rollouts")
            continue
        result = predictor.predict_strategy(history, future, strategy_name="explicit_factual_action_replay")
        valid = result["query_mask"][:, 0]
        if stage in (1, 2):
            assert valid.all(), "Observed surgery/postoperative actions should have training support"
        difference = None
        if valid.any():
            difference = float((result["risk"][:, 0][valid] - full["logits"][:, 3].sigmoid()[valid]).abs().max())
            assert difference <= 1e-5
        rollouts[f"S{stage}"] = {"supported_patients": int(valid.sum()), "maximum_risk_difference": difference}
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert exported["selected_step"] == best["selected_step"]
    assert set(exported["fit_ids"]) == set(split["train"])
    assert set(exported["model_state"]) == set(best["model_state"])
    assert all(torch.equal(value, best["model_state"][key]) for key, value in exported["model_state"].items())
    if predictor.model.cfg.terminal_clinical_anchor:
        from stageworld_tcwm.clinical import ClinicalAnchor

        fitting = cohort.batch(train_indices, device)
        ref = ClinicalAnchor().to(device)
        ref.fit(fitting["clinical"], fitting["binary"], fitting["binary_valid"])
        anchor = predictor.model.outcome.clinical_anchor
        for name, value in ref.state_dict().items():
            torch.testing.assert_close(value, anchor.state_dict()[name], rtol=0, atol=1e-6)
        if not world_trained:
            torch.testing.assert_close(full["logits"][:, 3], ref(batch["clinical"]), rtol=0, atol=1e-6)
    if has_concepts:
        from stageworld_tcwm.timeline_losses import report_training_mask

        values = cohort.tensors["s1_concepts"][train_indices]
        mask = report_training_mask(cohort.batch(train_indices, "cpu"))
        stats = exported["target_statistics"]
        torch.testing.assert_close(stats["report_counts"], mask.sum(0))
        for column in range(4):
            observed = values[:, column][mask[:, column]]
            torch.testing.assert_close(stats["report_mean"][column], observed.mean(), rtol=1e-6, atol=1e-6)
            if column:
                torch.testing.assert_close(stats["report_scale"][column], observed.std(unbiased=False).clamp_min(.05),
                                           rtol=1e-6, atol=1e-6)
        named = predictor.predict(independent)
        trained = exported["metadata"]["report_concept_head_trained"]
        assert named["metadata"]["s1_report_concepts"]["head_trained"] == trained
        if not trained:
            assert all(value.isnan().all() for value in named["s1_report_predictions"].values())
    rows = json.loads((run / "validation_predictions.json").read_text())
    assert len(rows) == 65 and all(row["query_order"] == 3 for row in rows)
    test_rows = run / "test_predictions.json"
    if test_rows.exists():
        rows = json.loads(test_rows.read_text())
        assert len(rows) == 130 and all(row["query_order"] == 3 for row in rows)
    return {"status": "pass", "source_root": str(ROOT), "run": str(run.resolve()),
            "inference_probe_role": "train", "probe_patients": len(indices),
            "labels_and_CT1_not_needed_for_prediction": True,
            "s1_report_head_checked": has_concepts,
            "s1_report_statistics_train_only_checked": has_concepts,
            "raw_intermediate_risk_masked": True, "s1_pcr_future_perturbation_difference": s1_future_difference,
            "s1_pcr_truncation_difference": pcr_difference, "explicit_factual_strategy_replay": rollouts,
            "export_matches_selected_checkpoint": True, "fit_patients": len(split["train"]),
            "world_model_trained": world_trained,
            "clinical_anchor_train_only_checked": predictor.model.cfg.terminal_clinical_anchor,
            "prediction_artifacts_terminal_only": True, "causal_validation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = verify(args.run, args.data_root, args.device)
    write_json(result, args.out)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
