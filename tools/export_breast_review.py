"""Export cohort-level breast training diagnostics through numeric allowlists.

The source directory remains private. This exporter does not copy checkpoints,
patient predictions, calibration bins, deployment paths, or raw minibatch logs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


STAGES = ("representation", "flow", "readout", "joint")
METRICS = (
    "n", "positives", "auroc", "auprc", "brier", "nll", "ece_10_equal_width",
    "selection_score", "generation_objective", "cases", "synthetic",
    "accuracy", "sensitivity", "specificity", "balanced_accuracy",
)
TERMS = (
    "loss", "gradient_norm", "lr", "reconstruction", "phase_difference",
    "masked_jepa", "variance", "covariance", "real_pcr", "observed_pcr",
    "residual_l2", "fm_image", "fm_state", "repa", "marginal_pcr",
    "grounding", "energy", "prediction_grounding",
)
INTERVAL_KEYS = ("lower", "upper", "valid_replicates", "mean_delta_A_minus_B")
SELECTION = {
    "representation": "reconstruction + masked_jepa + real_pcr",
    "flow": "forward flow validation objective",
    "readout": "marginal NLL",
    "joint": "marginal NLL + 0.05 * generation objective",
}


def numbers(value, keys):
    selected = {}
    for key in keys:
        if key not in value:
            continue
        item = value[key]
        if not isinstance(item, (int, float, bool)) or not math.isfinite(item):
            raise ValueError(f"Expected a finite scalar for {key}")
        selected[key] = item
    return selected


def intervals(value):
    return {key: numbers(value[key], INTERVAL_KEYS) for key in ("auroc", "auprc", "brier", "nll") if key in value}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def export(source, output):
    source, output = Path(source), Path(output)
    audit = read(source / "training_audit.json")
    components = read(source / "representation_components.json")
    baseline = read(source / "baseline_diagnostics.json")
    if "completed_utc" not in audit or len(audit["replays"]) != 4:
        raise ValueError("A completed four-checkpoint audit is required")
    summary = {
        "schema": "public-breast-training-review-v1",
        "study_date": "2026-09-29",
        "status": "completed",
        "started_at_asia_shanghai": "2026-09-28T18:58:20+08:00",
        "completed_at_asia_shanghai": "2026-09-29T08:24:57+08:00",
        "evaluation_scope": "existing_development_validation",
        "independent_test": False,
        "patients": {"train": 764, "validation": 102, "validation_positive": 32, "train_t3_paired": 524, "validation_t3_paired": 86},
        "task": "T0 to T3 single interval, with four optimization stages",
        "sampling": {"trajectories": 4, "solver": "heun", "steps": 20},
        "threshold": 0.5,
        "stages": {}, "replays": {}, "representation_components": {}, "baselines": {},
        "limitations": [
            "The validation patients also participated in frozen VQ selection; no independent test claim.",
            "Training summaries average sampled minibatches, not full training-set evaluations.",
            "Only best and last validation records were retained; no full validation curve is available.",
            "Best checkpoints minimize their declared objectives, not maximize AUROC.",
            "Readout and joint validation consume random numbers differently; cross-stage draws are not matched.",
            "Bootstrap intervals condition on selected checkpoints and do not correct development selection bias.",
            "The existing observed-only branches are diagnostic ablations, not separately retrained classifiers.",
            "The auprc field is sklearn average_precision_score (AP).",
            "Generation loss is not decoded MRI quality or evidence of benefit from generated futures.",
        ],
    }
    curves = {"schema": "public-breast-training-windows-v1", "window_logged_batches": 50, "log_every": 10, "stages": {}}
    for stage in STAGES:
        original = audit["stages"][stage]
        info = numbers(original, ("logged_records", "first_step", "last_step", "finite_logged_values", "continuous_every_10_steps", "window_logged_batches", "gradient_norm_max", "logged_gradient_clip_fraction"))
        info["selection_rule"] = SELECTION[stage]
        for key in ("first_window_mean", "window_ending_at_best_mean", "last_window_mean"):
            info[key] = numbers(original[key], TERMS)
        for key in ("validation_best", "validation_last"):
            info[key] = numbers(original[key], METRICS)
        info["checkpoints"] = {name: numbers(original["checkpoints"][name], ("step", "completed", "best_score")) for name in ("best", "last")}
        summary["stages"][stage] = info
        rows = [json.loads(line) for line in (source / f"{stage}_training.jsonl").read_text().splitlines() if line.strip()]
        checkpoints = {250, info["checkpoints"]["best"]["step"], info["last_step"]}
        windows = []
        for index, row in enumerate(rows):
            if row["step"] % 500 and row["step"] not in checkpoints:
                continue
            window = rows[max(0, index - 49):index + 1]
            means = {key: statistics.mean(numbers(item, (key,))[key] for item in window) for key in TERMS if all(key in item for item in window)}
            windows.append({"step": row["step"], "records_in_window": len(window), "mean": means})
        curves["stages"][stage] = windows

    for stage in ("readout", "joint"):
        for name in ("best", "last"):
            key = f"{stage}/{name}"
            value = audit["replays"][key]
            selected = numbers(value, ("step", "accuracy", "sensitivity", "specificity", "balanced_accuracy"))
            selected["metrics"] = numbers(value["metrics"], METRICS)
            selected["delta_from_stored"] = numbers(value["delta_from_stored"], ("auroc", "auprc", "brier", "nll", "selection_score"))
            selected["confusion"] = numbers(value["confusion"], ("tp", "tn", "fp", "fn"))
            summary["replays"][key] = selected
        for name in ("clinical_prior", "observed_only"):
            key = f"{stage}/{name}"
            summary["baselines"][key] = numbers(baseline["branches"][key], METRICS)
    for name in ("best", "last"):
        summary["representation_components"][name] = {
            "step": numbers(components[name], ("step",))["step"],
            "validation": numbers(components[name]["validation"], METRICS),
            "component_means": numbers(components[name]["component_means"], TERMS),
        }
    summary.update(numbers(audit, ("negative_only_accuracy", "bootstrap_repetitions")))
    for key in ("joint_best_bootstrap_95ci", "joint_best_minus_readout_best_95ci", "joint_last_minus_joint_best_95ci"):
        summary[key] = intervals(audit[key])
    output.mkdir(parents=True, exist_ok=True)
    write(output / "summary.json", summary)
    write(output / "training_curves.json", curves)
    print(f"Exported {len(summary['stages'])} stages and {len(summary['replays'])} checkpoint replays.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
