#!/usr/bin/env python
"""Evaluate locked historical token priors without supplying CT1 to the model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_ct_cv import load_folds, verify_run
from stageworld_tcwm.data import file_sha256, write_json
from stageworld_tcwm.forecast_evaluation import (
    ForecastSpace, distribution_scores, merge_point_scores, point_scores,
    prior_feature_draws, score_feature_sets,
)
from stageworld_tcwm.diagnostic_baselines import legal_images
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.monte_carlo import cohort_case_keys


def evaluate(folds_dir, runs_dir, out, ranks=(8, 32), samples=32, batch_size=8,
             seed=17, device="cpu", fold_indices=None):
    if samples < 2 or batch_size < 1 or not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("Require K>=2, positive batch size and unique ranks")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output directory; forecast evaluations are immutable")
    folds, _ = load_folds(folds_dir)
    indices = list(range(len(folds))) if fold_indices is None else list(fold_indices)
    if not indices or len(set(indices)) != len(indices) or not set(indices) <= set(range(len(folds))):
        raise ValueError("Require distinct existing fold indices")
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    start = time.monotonic()
    report = {
        "schema": "tcwm-prior-forecast-evaluation-v1", "endpoint": "CT1_cached_feature_forecasting",
        "development_evidence_only": True, "original_validation_or_test_scored": False,
        "ranks": list(ranks), "samples": samples, "mc_seed": seed, "mc_seed_policy": "case_key",
        "batch_size": batch_size, "device": str(device), "fold_indices": indices,
        "full_outer_oof": len(indices) == len(folds),
        "forward_inputs": "CT0+clinical+explicit_scenario; CT1_post_labels_physically_removed",
        "point_forecast": "mean_over_decoded_draws_of_fixed_token_mean_sd_features",
        "set_scoring": "permutation_invariant_set_loss; no_cross_time_tokenwise_MSE",
        "energy_score": "unbiased_finite_ensemble_pair_term_K_times_Kminus1; Euclidean_fixed_target_space",
        "intervals": "marginal_latent_feature_90pct_prediction_coverage_not_clinical_confidence",
        "limitations": [
            "Reused development outer folds are not independent confirmation.",
            "Different ranks define different targets; compare each skill to its own training mean.",
            "Sample set loss is distinct from error of the decoded MC aggregate point prediction.",
            "Ridge and Gaussian reference statistics use inner training only; ridge lambda is preset at 0.1.",
            "Token grids are independently cropped sets, not anatomically registered CT pairs.",
        ],
        "source_sha256": {name: file_sha256(path) for name, path in {
            "forecast_evaluation.py": ROOT / "src/stageworld_tcwm/forecast_evaluation.py",
            "evaluate_token_forecast.py": Path(__file__),
        }.items()}, "results": {str(rank): {"folds": []} for rank in ranks},
    }
    torch.backends.mha.set_fastpath_enabled(False)
    for fold in folds:
        if fold["index"] not in indices:
            continue
        path = Path(runs_dir) / f"fold-{fold['index']}" / "inference.pt"
        predictor = Predictor(path, device=device)
        verify_run(predictor, path, fold)
        if predictor.cfg.architecture != "token_world":
            raise ValueError("This evaluation requires a token_world prior checkpoint")
        cohort = fold["cohort"]
        training = cohort.batch(fold["rows"]["train"])
        spaces = {rank: ForecastSpace(training, fold["split"]["train"], rank) for rank in ranks}
        outer = fold["rows"]["outer_evaluation"]
        outer = outer[legal_images(cohort.batch(outer)).all(1)]
        if not len(outer):
            raise ValueError("No legal paired outer observations for forecasting")
        accumulators = {rank: {"points": {}, "distributions": {}, "sets": {}, "patients": 0} for rank in ranks}
        for rows in outer.split(batch_size):
            batch = cohort.batch(rows)
            keys = cohort_case_keys(cohort, rows)
            draws = prior_feature_draws(predictor.model, batch, keys, samples, seed).cpu()
            for rank, space in spaces.items():
                current = accumulators[rank]
                target = space.encode(batch["ct1"])
                projected = space.encode_draws(draws)
                points = {"training_CT1_mean": space.target_mean.expand_as(target),
                          "copy_CT0": space.encode(batch["ct0"]), "ridge_CT0_condition": space.ridge(batch),
                          "prior_single_sample": projected[:, 0], "prior_decoded_MC_mean": projected.mean(1)}
                for name, value in point_scores(points, target, space.target_mean).items():
                    current["points"].setdefault(name, []).append(value)
                for name, values in (("prior", projected), ("training_Gaussian", space.gaussian(keys, samples, seed))):
                    for metric, value in distribution_scores(values, target).items():
                        current["distributions"].setdefault(name, {}).setdefault(metric, []).append(value)
                sets = score_feature_sets(draws, batch["ct1"], batch["ct0"], space.training_raw_mean_tokens,
                                          predictor.model.image_scale.cpu())
                for name, value in sets.items():
                    current["sets"][name] = current["sets"].get(name, 0.) + value * len(rows)
                current["patients"] += len(rows)
        for rank, current in accumulators.items():
            record = {"fold": fold["index"], "paired_outer_patients": len(outer),
                "training_patients": len(training["clinical"]),
                "bundle_sha256": file_sha256(path), "selected_epoch_zero_based": predictor.bundle["selected_epoch"],
                "provenance": spaces[rank].provenance,
                "points": {name: merge_point_scores(parts) for name, parts in current["points"].items()},
                "distributions": {name: {metric: float(torch.cat(parts).mean()) for metric, parts in values.items()}
                                  for name, values in current["distributions"].items()},
                "sets_and_channel_moments": {name: value / len(outer) for name, value in current["sets"].items()}}
            report["results"][str(rank)]["folds"].append(record)
        print(json.dumps({"completed_fold": fold["index"], "paired_outer_patients": len(outer)}), flush=True)
        del predictor, spaces, training
    for rank, result in report["results"].items():
        records = result["folds"]
        total = sum(record["paired_outer_patients"] for record in records)
        result["pooled_outer"] = {"patients": total,
            "points": {name: merge_point_scores([record["points"][name] for record in records]) for name in records[0]["points"]},
            "distributions": {name: {metric: sum(record["distributions"][name][metric] * record["paired_outer_patients"] for record in records) / total
                                    for metric in records[0]["distributions"][name]} for name in records[0]["distributions"]},
            "sets_and_channel_moments": {name: sum(record["sets_and_channel_moments"][name] * record["paired_outer_patients"] for record in records) / total
                                          for name in records[0]["sets_and_channel_moments"]}}
    report["elapsed_seconds"] = time.monotonic() - start
    write_json(report, out / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="Directory containing historical token_prior/fold-N bundles")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    evaluate(args.folds, args.runs, args.out, args.ranks, args.samples, args.batch_size,
             args.seed, args.device, args.fold_indices)


if __name__ == "__main__":
    main()
