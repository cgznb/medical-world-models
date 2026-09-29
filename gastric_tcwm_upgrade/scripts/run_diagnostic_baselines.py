#!/usr/bin/env python
"""Run B0-B4 fixed-feature diagnostics on the existing nested development folds."""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_ct_cv import load_folds
from stageworld_tcwm.data import atomic_save, file_sha256, fingerprint, write_json
from stageworld_tcwm.diagnostic_baselines import INFORMATION, LAMBDAS, PAIRS, fit_fold, metrics, paired_comparison, point_losses


def run(folds_dir, out, ranks=(8, 32), fold_indices=None, bootstrap=1000, max_iter=2000):
    if (not ranks or len(set(ranks)) != len(ranks) or any(rank < 1 for rank in ranks)
            or bootstrap < 0 or not 1 <= max_iter <= 2000):
        raise ValueError("Require unique positive ranks, nonnegative bootstrap and 1..2000 iterations")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output directory; diagnostic runs are immutable")
    folds, all_ids = load_folds(folds_dir)
    indices = list(range(len(folds))) if fold_indices is None else list(fold_indices)
    if not indices or len(set(indices)) != len(indices) or not set(indices) <= set(range(len(folds))):
        raise ValueError("Select distinct existing fold indices")
    selected = [fold for fold in folds if fold["index"] in indices]
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    # The secret is intentionally not exported; patient keys cannot be dictionary
    # matched against an identifier list from a public report.
    key_secret = secrets.token_bytes(32)
    anonymous = lambda identifier: hmac.new(key_secret, identifier.encode(), hashlib.sha256).hexdigest()
    report = {
        "schema": "tcwm-fixed-feature-diagnostic-v1",
        "study_scope": "development_reuse_of_existing_nested_original_training_folds",
        "original_validation_or_test_scored": False,
        "full_outer_oof": len(selected) == len(folds), "fold_indices": indices,
        "ranks": list(ranks), "lambda_grid": list(LAMBDAS),
        "selection": "minimum_inner_selection_unpenalized_NLL_among_converged_fits; "
                     "tie_prefers_larger_lambda; no_outer_training_refit",
        "original_training_membership_sha256": fingerprint(sorted(all_ids)),
        "endpoint": "recorded_recurrence_binary_without_fixed_horizon",
        "scenario_semantics": "explicit_retrospective_treatment_and_query_interval; no_causal_effect_claim",
        "B4_scope": "observed_CT1_diagnostic_not_S0_deployment",
        "method_changed_from_historical_direct_ct": True,
        "source_sha256": {name: file_sha256(path) for name, path in {
            "diagnostic_baselines.py": ROOT / "src/stageworld_tcwm/diagnostic_baselines.py",
            "run_diagnostic_baselines.py": Path(__file__),
        }.items()},
        "limitations": [
            "Existing outer OOF informed previous candidate selection and remains development evidence.",
            "Bootstrap conditions on fitted predictions and excludes refitting and candidate selection uncertainty.",
            "Different ranks and folds define separate training-fitted projection spaces.",
            "CT dependence or wider prediction ranges do not establish incremental prognostic benefit.",
        ], "results": {},
    }
    private = {"schema": report["schema"], "patient_key_scheme": "run_specific_secret_HMAC_SHA256", "ranks": {}}
    for rank in ranks:
        records, parts = [], []
        for fold in selected:
            record, values, state = fit_fold(fold, rank, max_iter)
            records.append(record)
            parts.append(values)
            atomic_save(state, out / f"rank-{rank}" / f"fold-{fold['index']}.pt")
            print(json.dumps({"rank": rank, "completed_fold": fold["index"],
                              "outer": {name: value["outer"] for name, value in record["models"].items()}}), flush=True)
        ids = [patient for part in parts for patient in part["patient_ids"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Outer patient predictions must be unique")
        y = np.concatenate([part["labels"] for part in parts])
        fold_ids = np.concatenate([part["fold"] for part in parts])
        predictions = {name: np.concatenate([part["probabilities"][name] for part in parts]) for name in INFORMATION}
        comparisons, deltas = {}, {}
        for newer, reference in PAIRS:
            name = f"{newer}_vs_{reference}"
            comparisons[name] = paired_comparison(y, predictions[newer], predictions[reference], fold_ids, bootstrap)
            a, b = point_losses(y, predictions[newer]), point_losses(y, predictions[reference])
            deltas[name] = {"probability": torch.from_numpy(predictions[newer] - predictions[reference]),
                            **{key: torch.from_numpy(a[key] - b[key]) for key in a}}
        report["results"][str(rank)] = {"folds": records,
            "pooled_outer": {name: metrics(y, values) for name, values in predictions.items()},
            "paired_comparisons": comparisons,
            "next_step": "ROI_and_representation_audit_unless_B4_over_B3_signal_is_consistent_across_folds"}
        private["ranks"][str(rank)] = {"case_keys": [anonymous(patient) for patient in ids],
            "fold": torch.from_numpy(fold_ids), "labels": torch.from_numpy(y),
            "probabilities": {name: torch.from_numpy(value) for name, value in predictions.items()},
            "paired_deltas": deltas}
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(report, out / "report.json")
        atomic_save(private, out / "paired_predictions_private.pt")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    run(args.folds, args.out, args.ranks, args.fold_indices, args.bootstrap, args.max_iter)


if __name__ == "__main__":
    main()
