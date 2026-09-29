#!/usr/bin/env python
"""Report locked next-round candidates using permitted development OOF rows."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_ct_cv import load_folds, verify_run
from stageworld_tcwm.data import atomic_save, write_json, fingerprint, file_sha256
from stageworld_tcwm.evaluation import collect_predictions, evaluate_predictions, binary_metrics
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.monte_carlo import cohort_case_keys
from stageworld_tcwm.survival import mixture_binary_nll


def metrics(labels, probabilities, nll):
    report = {f"S{k}": binary_metrics(labels.numpy(), probabilities[:, k].numpy()) for k in range(3)}
    report["selection_nll"] = float((nll[:, 0]*.5+nll[:, 1]*.5).mean())
    report["legacy_three_stage_nll"] = float(nll.mean())
    return report


def paired_summary(labels, reference, candidate, seed=20260929, draws=2000):
    rng = np.random.default_rng(seed)
    y = labels.numpy()
    strata = [np.flatnonzero(y == value) for value in (0, 1)]
    delta = []
    for _ in range(draws):
        rows = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        a = binary_metrics(y[rows], reference[rows])
        b = binary_metrics(y[rows], candidate[rows])
        delta.append([b[k]-a[k] for k in ("nll", "brier", "auroc", "average_precision")])
    a, b = binary_metrics(y, reference), binary_metrics(y, candidate)
    return {key: {"candidate_minus_reference": b[key]-a[key],
                  "fixed_oof_paired_bootstrap_95ci": np.quantile(np.asarray(delta)[:, i], [.025, .975]).tolist()}
            for i, key in enumerate(("nll", "brier", "auroc", "average_precision"))}


def verify_study_case(study, case, fold, contract, bundle, training, provenance, config_path):
    if case not in study["configs"] or fold["index"] not in study["fold_indices"]:
        raise ValueError("Case/fold absent from the locked study")
    expected = study["configs"][case]
    if any(fingerprint(contract[key]) != fingerprint(expected[key]) for key in ("model", "train")):
        raise ValueError("Case configuration differs from the locked study")
    if study["fold_hashes"][fold["index"]] != fold["hashes"]:
        raise ValueError("Fold differs from the locked study")
    if provenance != {"study_id": fingerprint(study), "case": case, "fold": fold["index"],
                       "config_sha256": file_sha256(config_path)}:
        raise ValueError("Run provenance differs from the locked study")
    for key in ("contract_id", "selected_kind", "selected_epoch", "selected_optimizer_steps", "selected_supervised_steps"):
        if key not in training or training[key] != bundle[key]:
            raise ValueError("Training report differs from the exported candidate: " + key)
    if (training["optimizer_steps"] < bundle["selected_optimizer_steps"] or
            training["supervised_steps"] < bundle["selected_supervised_steps"]):
        raise ValueError("Selected updates exceed actual training updates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("G0", "G1", "G2", "G3"), default=["G0", "G1"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Evaluation requires a new directory")
    folds, _ = load_folds(args.folds)
    study = json.loads((args.runs / "study_protocol.json").read_text())
    report = {"development_only": True, "independent_confirmation": False,
        "original_validation_or_test_scored": False, "samples": args.samples, "mc_seed": args.seed,
        "mc_seed_policy": "case_key", "mc_antithetic": True, "stage_weights": [.5, .5, 0],
        "outcome": "recorded_recurrence_binary", "cases": {},
        "uncertainty_limit": "Fixed OOF bootstrap excludes retraining and candidate selection uncertainty"}
    private = {}
    anchor_parts, label_parts = [], []
    for case in args.cases:
        probabilities, nlls, records, keys = [], [], [], []
        for fold in folds:
            run = args.runs / case / f"fold-{fold['index']}"
            predictor = Predictor(run / "inference.pt", args.device)
            contract = verify_run(predictor, run / "inference.pt", fold)
            training = json.loads((run / "training_report.json").read_text())
            provenance = json.loads((run / "study_provenance.json").read_text())
            verify_study_case(study, case, fold, contract, predictor.bundle, training,
                              provenance, args.runs / "configs" / f"{case}.json")
            if contract["train"].get("stage_weights") != [.5, .5, 0]:
                raise ValueError("This report requires the explicit next-round stage weights")
            cohort, rows = fold["cohort"], fold["rows"]["outer_evaluation"]
            batch = cohort.batch(rows)
            if not (batch["binary_valid"].all() and batch["prefix_valid"].all()):
                raise ValueError("OOF report requires the complete recorded-status cohort")
            pred = collect_predictions(predictor.model, cohort, rows, samples=args.samples,
                batch_size=16, seed=args.seed, mc_seed_policy="case_key", mc_antithetic=True)
            probabilities.append(pred.sigmoid().mean(2))
            nlls.append(torch.stack([mixture_binary_nll(pred[:, k], batch["binary"]) for k in range(3)], 1))
            keys.extend(cohort_case_keys(cohort, rows))
            record = {"fold": fold["index"], "train_patients": len(fold["rows"]["train"]),
                "train_events": int(cohort.tensors["binary"][fold["rows"]["train"]].sum()),
                "outer_patients": len(rows), "outer_events": int(batch["binary"].sum()),
                "rank": None, "target_space": "original_frozen_CT_tokens", "stage_weights": [.5, .5, 0],
                "information_sets": {"S0": "clinical_CT0_explicit_scenario", "S1": "S0_plus_legal_CT1", "S2": "S1_no_new_observation"},
                "seed": contract["train"]["seed"], "mc_samples": args.samples,
                "changed_evaluation_contract_from_historical": True,
                "training": training, "outer": evaluate_predictions(pred, cohort, rows, predictor.cfg,
                    fold["rows"]["train"], stage_weights=(.5, .5, 0))}
            records.append(record)
            if case == args.cases[0]:
                with torch.inference_mode():
                    anchor = predictor.model.recurrence_anchor(batch["clinical"].to(args.device)).cpu()
                anchor_parts.append(anchor)
                label_parts.append(batch["binary"])
        labels = torch.cat(label_parts)
        probability, nll = torch.cat(probabilities), torch.cat(nlls)
        private[case] = {"case_keys": keys, "probability": probability, "nll": nll, "labels": labels}
        anchor_probability = torch.cat(anchor_parts).sigmoid().numpy()
        report["cases"][case] = {"folds": records, "oof": metrics(labels, probability, nll),
            "S1_vs_clinical_anchor": paired_summary(labels, anchor_probability, probability[:, 1].numpy())}
    labels = torch.cat(label_parts)
    anchors = torch.cat(anchor_parts)
    report["clinical_anchor"] = binary_metrics(labels.numpy(), anchors.sigmoid().numpy())
    for candidate, reference in (("G1", "G0"), ("G2", "G1"), ("G3", "G1")):
        if candidate in private and reference in private:
            report[f"{candidate}_vs_{reference}"] = {f"S{k}": paired_summary(labels,
                private[reference]["probability"][:, k].numpy(),
                private[candidate]["probability"][:, k].numpy()) for k in (0, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(report, args.out / "report.json")
    atomic_save(private, args.out / "private_oof.pt")
    print(json.dumps({case: report["cases"][case]["oof"] for case in args.cases}, indent=2))


if __name__ == "__main__":
    main()
