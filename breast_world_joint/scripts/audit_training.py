"""Inspect completed training and replay its validation without optimizer updates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    import numpy as np
    import torch
    from responsewm import training
    from responsewm.data import ManifestStore
    from responsewm.io import seed_all, write_json
    from responsewm.metrics import patient_bootstrap, paired_bootstrap_delta

    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run),
        "protocol": "Replay original validate(), K=4, Heun20, original validation seed; no optimizer updates",
        "threshold": 0.5,
        "stages": {},
        "replays": {},
        "limitations": [
            "Training curves contain every tenth minibatch, not full-cohort training evaluation.",
            "Only best and last validation records survived; intermediate validation curves are unavailable.",
            "Readout and joint validation consume the random generator differently, so their per-patient draws are not matched.",
            "Bootstrap intervals condition on selected checkpoints and do not correct development selection bias.",
            "Accuracy uses a fixed 0.5 threshold with pCR as the positive class; no threshold was tuned.",
        ],
    }

    def save():
        write_json(args.output, report)

    for stage in training.STAGES:
        folder = args.run / stage
        rows = [json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines() if line.strip()]
        keys = [key for key in rows[0] if isinstance(rows[0][key], (float, int))]

        def means(selected):
            return {key: statistics.mean(row[key] for row in selected) for key in keys if all(key in row for row in selected)}

        window = min(50, len(rows))
        record = {
            "logged_records": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
            "finite_logged_values": all(np.isfinite(row[key]) for row in rows for key in keys if key in row),
            "continuous_every_10_steps": [row["step"] for row in rows] == list(range(10, rows[-1]["step"] + 1, 10)),
            "window_logged_batches": window,
            "first_window_mean": means(rows[:window]), "last_window_mean": means(rows[-window:]),
            "gradient_norm_max": max(row["gradient_norm"] for row in rows),
            "logged_gradient_clip_fraction": statistics.mean(row["gradient_norm"] > 1.0 for row in rows),
            "validation_best": json.loads((folder / "validation_best.json").read_text()),
            "validation_last": json.loads((folder / "validation_last.json").read_text()),
            "checkpoints": {},
        }
        for name in ("best", "last"):
            payload = torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True, mmap=True)
            record["checkpoints"][name] = {key: payload[key] for key in ("stage", "step", "completed", "best_score")}
            if name == "best":
                selected = [row for row in rows if row["step"] <= payload["step"]][-window:]
                record["window_ending_at_best_mean"] = means(selected)
            del payload
        report["stages"][stage] = record
        print(json.dumps({"event": "stage_audited", "stage": stage, "best_step": record["checkpoints"]["best"]["step"]}), flush=True)
    save()

    store = ManifestStore(args.manifest)
    original_metrics = training.classification_metrics
    captured = {}

    def capture_metrics(y, p):
        captured["y"] = np.asarray(y, dtype=int)
        captured["p"] = np.asarray(p, dtype=float)
        return original_metrics(y, p)

    training.classification_metrics = capture_metrics
    predictions = {}
    all_labels = None
    patient_ids = None
    for stage, name in (("readout", "best"), ("readout", "last"), ("joint", "best"), ("joint", "last")):
        key = f"{stage}/{name}"
        started = time.monotonic()
        print(json.dumps({"event": "replay_started", "checkpoint": key}), flush=True)
        model, payload = training.load_trained(args.run / stage / f"{name}.pt", device="cuda")
        store.set_statistics(payload["metadata"]["statistics"])
        seed_all(model.cfg.training.seed, model.cfg.training.threads, model.cfg.training.strict_determinism)
        metrics = training.validate(model, store, model.cfg, stage)
        y, p = captured["y"], captured["p"]
        ids = training._validation_indices(store, model.cfg.training.validation_cases)
        current_ids = [store.cases[i]["patient_id"] for i in ids if store.cases[i]["target"]["pcr"] is not None]
        assert len(set(current_ids)) == len(y) == 102
        if all_labels is None:
            all_labels, patient_ids = y.copy(), current_ids
        assert np.array_equal(all_labels, y) and current_ids == patient_ids
        predicted = p >= 0.5
        tp = int(((y == 1) & predicted).sum())
        tn = int(((y == 0) & ~predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        stored = report["stages"][stage][f"validation_{name}"]
        delta = {metric: metrics[metric] - stored[metric] for metric in ("auroc", "auprc", "brier", "nll", "selection_score")}
        report["replays"][key] = {
            "step": payload["step"], "metrics": metrics, "delta_from_stored": delta,
            "accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp), "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "elapsed_seconds": time.monotonic() - started,
        }
        predictions[key] = p.copy()
        print(json.dumps({"event": "replay_complete", "checkpoint": key, "accuracy": (tp + tn) / len(y), "metric_delta": delta, "seconds": time.monotonic() - started}), flush=True)
        save()
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()
    training.classification_metrics = original_metrics
    report["negative_only_accuracy"] = float((all_labels == 0).mean())
    report["bootstrap_repetitions"] = 2000
    report["joint_best_bootstrap_95ci"] = patient_bootstrap(all_labels, predictions["joint/best"], patient_ids, 2000, 20260929)
    report["joint_best_minus_readout_best_95ci"] = paired_bootstrap_delta(all_labels, predictions["joint/best"], predictions["readout/best"], patient_ids, 2000, 20260929)
    report["joint_last_minus_joint_best_95ci"] = paired_bootstrap_delta(all_labels, predictions["joint/last"], predictions["joint/best"], patient_ids, 2000, 20260929)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({"event": "audit_complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
