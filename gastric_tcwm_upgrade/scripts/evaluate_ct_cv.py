#!/usr/bin/env python
"""Evaluate explicit nested outer folds without scoring original holdouts."""
from pathlib import Path
import argparse
import json
import os
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_repairs import ct1_permutation, differences, distribution, verify_contract
from stageworld_tcwm.clinical import ClinicalAnchor
from stageworld_tcwm.data import Cohort, atomic_save, file_sha256, fingerprint, split_indices, write_json
from stageworld_tcwm.evaluation import binary_metrics
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.survival import mixture_binary_nll

LATENT_KEYS = ("predicted_ct_latent", "target_ct_latent", "baseline_ct_latent")


def validate_fold(cohort, split, index):
    rows = split_indices(cohort, split)
    metadata = cohort.metadata
    if metadata.get("nested_selection") is not True or metadata.get("excluded_scoring_permitted") is not False:
        raise ValueError("Require nested folds with original holdout scoring prohibited")
    outer = metadata.get("outer_evaluation_ids", [])
    excluded = metadata.get("excluded_ids", [])
    if not outer or not excluded or len(set(outer)) != len(outer) or len(set(excluded)) != len(excluded):
        raise ValueError("Require nonempty unique outer and excluded patient lists")
    if set(outer) & set(excluded) or set(outer + excluded) != set(split["test"]):
        raise ValueError("Outer and excluded patients must exactly partition the test role")
    lookup = {patient: i for i, patient in enumerate(cohort.ids)}
    for name, ids in (("outer_evaluation", outer), ("excluded", excluded)):
        expected = [lookup[patient] for patient in ids]
        if metadata.get(name + "_indices") != expected:
            raise ValueError("Metadata patient indices do not match explicit patient membership")
        rows[name] = torch.tensor(expected, dtype=torch.long)
    hidden = rows["excluded"]
    if any(bool(cohort.tensors[name][hidden].any()) for name in
           ("binary", "pcr", "binary_valid", "pcr_valid", "prefix_valid")):
        raise ValueError("Excluded original holdout labels and masks must be redacted")
    fit_ids = cohort.encoders.get("fit_ids", [])
    if len(fit_ids) != len(split["train"]) or set(fit_ids) != set(split["train"]):
        raise ValueError("Fold encoders must bind exactly the inner training patients")
    inner = metadata.get("inner_fold", {})
    original = split["train"] + split["validation"] + outer
    if inner.get("index") != index or inner.get("nested_selection") is not True:
        raise ValueError("Nested fold metadata and directory disagree")
    if fingerprint(sorted(original)) != inner.get("original_train_membership_sha256"):
        raise ValueError("Original training membership hash differs from this fold")
    return rows, set(original)


def load_folds(root):
    folders = sorted((p for p in Path(root).glob("fold-*") if p.is_dir()),
                     key=lambda p: int(p.name.removeprefix("fold-")))
    if not folders:
        raise ValueError("No nested fold directories found")
    folds, expected, excluded, outer_seen = [], None, None, []
    for index, folder in enumerate(folders):
        if folder.name != f"fold-{index}":
            raise ValueError("Fold directories must be contiguous from fold-0")
        path = folder / "cohort.pt"
        cohort = Cohort.load(path)
        split = json.loads((folder / "split.json").read_text())
        rows, membership = validate_fold(cohort, split, index)
        if cohort.metadata["inner_fold"].get("folds") != len(folders):
            raise ValueError("Incomplete set of nested folds")
        if expected is None:
            expected, excluded = membership, set(cohort.metadata["excluded_ids"])
        if membership != expected or set(cohort.metadata["excluded_ids"]) != excluded:
            raise ValueError("Fold cohorts do not share the same original training/excluded population")
        hashes = {"cohort": file_sha256(path), "split": fingerprint(split)}
        preparation = json.loads((folder / "preparation.json").read_text())
        if preparation.get("cohort_sha256") != hashes["cohort"] or preparation.get("split_sha256") != file_sha256(folder / "split.json"):
            raise ValueError("Fold preparation hashes do not match current files")
        outer_seen.extend(cohort.metadata["outer_evaluation_ids"])
        folds.append({"index": index, "cohort": cohort, "split": split, "rows": rows, "hashes": hashes})
    if len(outer_seen) != len(set(outer_seen)) or set(outer_seen) != expected:
        raise ValueError("Every original training patient must appear in exactly one outer fold")
    return folds, sorted(expected)


def verify_run(predictor, path, fold):
    contract = verify_contract(predictor.bundle, path, fold["hashes"]["cohort"],
                               fold["hashes"]["split"], fold["split"]["train"])
    if fingerprint(contract) != predictor.bundle["contract_id"]:
        raise ValueError("Training contract fingerprint is invalid")
    fit_ids = predictor.bundle.get("encoders", {}).get("fit_ids", [])
    if len(fit_ids) != len(fold["split"]["train"]) or set(fit_ids) != set(fold["split"]["train"]):
        raise ValueError("Inference bundle does not bind exactly the inner training patients")
    metadata = predictor.bundle.get("metadata", {})
    for key in ("outer_evaluation_ids", "excluded_ids", "inner_fold"):
        if metadata.get(key) != fold["cohort"].metadata[key]:
            raise ValueError("Bundle and fold nested metadata differ")
    if predictor.cfg.endpoint != "binary":
        raise ValueError("Nested CT evaluation currently requires binary endpoints")
    return contract


def truth(cohort, rows):
    batch = cohort.batch(rows)
    return {"binary": batch["binary"], "pcr": batch["pcr"],
            "binary_valid": batch["binary_valid"], "pcr_valid": batch["pcr_valid"],
            "prefix_valid": batch["prefix_valid"]}


@torch.inference_mode()
def clinical_baseline(fold):
    cohort, rows = fold["cohort"], fold["rows"]
    training = cohort.batch(rows["train"])
    outer = cohort.batch(rows["outer_evaluation"])
    logits = {}
    for endpoint in ("binary", "pcr"):
        anchor = ClinicalAnchor()
        anchor.fit(training["clinical"], training[endpoint], training[endpoint + "_valid"])
        logits[endpoint] = anchor(outer["clinical"]).cpu()
    probabilities = logits["binary"].sigmoid()[:, None].repeat(1, 3)
    return {
        **truth(cohort, rows["outer_evaluation"]),
        "patient_ids": list(cohort.metadata["outer_evaluation_ids"]),
        "probability": probabilities, "permuted_probability": probabilities.clone(),
        "nll": mixture_binary_nll(logits["binary"][:, None], outer["binary"])[:, None].repeat(1, 3),
        "pcr_probability": logits["pcr"].sigmoid(),
        "permuted_pcr_probability": logits["pcr"].sigmoid(),
        "pcr_nll": mixture_binary_nll(logits["pcr"][:, None], outer["pcr"]),
    }


@torch.inference_mode()
def target_training_mean(model, cohort, rows, batch_size, dimension):
    device = next(model.parameters()).device
    paired = cohort.tensors["image_valid"][rows].all(1)
    selected = rows[paired]
    if not len(selected):
        raise ValueError("Absolute CT1 evaluation needs paired inner training observations")
    total = torch.zeros(dimension, dtype=torch.float64)
    for part in selected.split(batch_size):
        encoded = model.encode_image(cohort.tensors["ct1"][part].to(device)).detach().double().cpu()
        if encoded.shape != (len(part), dimension) or not torch.isfinite(encoded).all():
            raise ValueError("Training CT1 encoding has incompatible latent dimensions")
        total += encoded.sum(0)
    return total / len(selected)


def ct_error_sums(predicted, target, baseline, training_mean, paired):
    if predicted.shape != target.shape or baseline.shape != target.shape or target.ndim != 2:
        raise ValueError("Absolute CT latent outputs must share shape [patients, dimensions]")
    p, y, before = [value[paired].double() for value in (predicted, target, baseline)]
    return {"patients": len(y), "dimensions": target.shape[1], "elements": y.numel(),
            "predicted_sse": float((p - y).square().sum()),
            "copy_ct0_sse": float((before - y).square().sum()),
            "training_mean_sse": float((training_mean - y).square().sum())}


def ct_summary(parts):
    supported = [part for part in parts if part is not None]
    if not supported:
        return {"supported": False, "reason": "model_does_not_export_absolute_CT_latents"}
    if len(supported) != len(parts) or len({part["dimensions"] for part in supported}) != 1:
        raise ValueError("CT output support or latent dimensions differ within a case")
    total = {key: sum(part[key] for part in supported) for key in
             ("patients", "elements", "predicted_sse", "copy_ct0_sse", "training_mean_sse")}
    denominator = total["training_mean_sse"]
    return {"supported": True, **total, "dimensions": supported[0]["dimensions"],
            "prediction_mse": total["predicted_sse"] / total["elements"] if total["elements"] else None,
            "copy_ct0_mse": total["copy_ct0_sse"] / total["elements"] if total["elements"] else None,
            "training_mean_mse": denominator / total["elements"] if total["elements"] else None,
            "prediction_r2_vs_training_mean": 1 - total["predicted_sse"] / denominator if denominator else None,
            "copy_ct0_r2_vs_training_mean": 1 - total["copy_ct0_sse"] / denominator if denominator else None,
            "target": "absolute_CT1_in_each_inner_training_fitted_latent_space",
            "pooled_definition": "sum_fold_squared_errors_divided_by_sum_fold_elements_or_baseline_errors"}


@torch.inference_mode()
def model_outer(model, fold, samples, batch_size, seed, permutation_seed):
    cohort, rows = fold["cohort"], fold["rows"]["outer_evaluation"]
    device = next(model.parameters()).device
    model.eval()
    permutation = ct1_permutation(cohort.tensors["image_valid"][rows, 1], permutation_seed)
    chunks = {key: [] for key in ("probability", "permuted_probability", "nll", "pcr_probability",
                                  "permuted_pcr_probability", "pcr_nll")}
    latent = {key: [] for key in LATENT_KEYS}
    latent_supported = None
    for start in range(0, len(rows), batch_size):
        section = slice(start, start + batch_size)
        batch = cohort.batch(rows[section], device)
        output = model(batch, samples=samples, seed=seed + start, max_stage=2, compute_aux=True)
        modified = dict(batch)
        modified["ct1"] = cohort.tensors["ct1"][rows[permutation[section]]].to(device)
        altered = model(modified, samples=samples, seed=seed + start, max_stage=2, compute_aux=True)
        if not torch.equal(output["predictions"][:, 0], altered["predictions"][:, 0]):
            raise ValueError("S0 changed after CT1 permutation: stage boundary violated")
        if not torch.equal(output["pcr_logits"], altered["pcr_logits"]):
            raise ValueError("Prior pCR changed after CT1 permutation: stage boundary violated")
        for prefix, result in (("", output), ("permuted_", altered)):
            chunks[prefix + "probability"].append(result["predictions"].float().sigmoid().mean(2).cpu())
            chunks[prefix + "pcr_probability"].append(result["pcr_logits"].float().sigmoid().mean(1).cpu())
        chunks["nll"].append(torch.stack([mixture_binary_nll(output["predictions"][:, stage], batch["binary"]) for stage in range(3)], 1).cpu())
        chunks["pcr_nll"].append(mixture_binary_nll(output["pcr_logits"], batch["pcr"]).cpu())
        presence = [key in output for key in LATENT_KEYS]
        if any(presence) and not all(presence):
            raise ValueError("Model exported an incomplete absolute CT1 latent contract")
        if latent_supported is not None and latent_supported != all(presence):
            raise ValueError("CT latent contract changes across batches")
        latent_supported = all(presence)
        if latent_supported:
            for key in LATENT_KEYS:
                latent[key].append(output[key].detach().float().cpu())
    values = {key: torch.cat(value) for key, value in chunks.items()}
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite outer predictions")
    ct = None
    if latent_supported:
        predicted, target, baseline = [torch.cat(latent[key]) for key in LATENT_KEYS]
        if not all(torch.isfinite(value).all() for value in (predicted, target, baseline)):
            raise ValueError("Nonfinite CT latent predictions")
        mean = target_training_mean(model, cohort, fold["rows"]["train"], batch_size, target.shape[1])
        ct = ct_error_sums(predicted, target, baseline, mean, cohort.tensors["image_valid"][rows].all(1))
    return {**truth(cohort, rows), "patient_ids": list(cohort.metadata["outer_evaluation_ids"]), **values}, ct


def summarize(values):
    stages = {}
    for stage in range(3):
        valid = values["prefix_valid"][:, stage] & values["binary_valid"]
        probability = values["probability"][:, stage]
        stats = binary_metrics(values["binary"][valid].numpy(), probability[valid].numpy())
        stats["nll"] = float(values["nll"][valid, stage].mean()) if valid.any() else None
        stats["probability_distribution"] = distribution(probability[values["prefix_valid"][:, stage]])
        stats["ct1_permutation"] = differences(probability[values["prefix_valid"][:, stage]],
            values["permuted_probability"][values["prefix_valid"][:, stage], stage])
        stages[f"S{stage}"] = stats
    valid = values["pcr_valid"]
    pcr = binary_metrics(values["pcr"][valid].numpy(), values["pcr_probability"][valid].numpy())
    pcr["nll"] = float(values["pcr_nll"][valid].mean()) if valid.any() else None
    pcr["probability_distribution"] = distribution(values["pcr_probability"][valid])
    pcr["ct1_permutation"] = differences(values["pcr_probability"][valid], values["permuted_pcr_probability"][valid])
    paired = values["prefix_valid"][:, :2].all(1)
    endpoint_valid = values["prefix_valid"] & values["binary_valid"][:, None]
    counts = endpoint_valid.sum(1)
    available = counts > 0
    patient_nll = torch.where(endpoint_valid, values["nll"], 0.).sum(1)[available] / counts[available]
    return {"patients": len(values["patient_ids"]), "stages": stages, "pcr": pcr,
            "mean_prefix_nll": float(patient_nll.mean()) if available.any() else None,
            "s0_s1_difference": differences(values["probability"][paired, 0], values["probability"][paired, 1])}


def pool_outer(chunks, expected_ids):
    ids = [patient for chunk in chunks for patient in chunk["patient_ids"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("OOF requires every permitted patient exactly once")
    lookup = {patient: row for row, patient in enumerate(ids)}
    order = torch.tensor([lookup[patient] for patient in expected_ids])
    return {"patient_ids": list(expected_ids), **{key: torch.cat([chunk[key] for chunk in chunks])[order]
             for key in chunks[0] if key != "patient_ids"}}


def evaluate(folds_dir, runs_dir, cases, out, device="cpu", samples=32, batch_size=16,
             seed=1729, permutation_seed=2718):
    if samples < 2 or batch_size < 1 or not cases or len(set(cases)) != len(cases):
        raise ValueError("Require unique cases, samples >= 2 and positive batch size")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case) for case in cases):
        raise ValueError("Case names must be simple run-directory names")
    os.umask(0o077)
    torch.backends.mha.set_fastpath_enabled(False)
    folds, expected_ids = load_folds(folds_dir)
    baseline_parts = [clinical_baseline(fold) for fold in folds]
    baseline = pool_outer(baseline_parts, expected_ids)
    report = {"schema": "tcwm-nested-outer-evaluation-v1", "original_validation_or_test_scored": False,
              "outer_oof_patients": len(expected_ids), "folds": len(folds),
              "original_training_membership_sha256": fingerprint(sorted(expected_ids)),
              "samples": samples, "batch_size": batch_size, "seed": seed,
              "permutation_seed": permutation_seed, "device": str(device),
              "clinical_baseline": {"protocol": "ClinicalAnchor_C1_fit_on_same_inner_training_only",
                                    "folds": [summarize(part) for part in baseline_parts], "oof": summarize(baseline)},
              "cases": {}, "limitations": [
                  "Outer folds evaluate the original development training population, not an independent external cohort.",
                  "Each checkpoint is selected using its separate inner validation partition.",
                  "Cases and seeds are not selected by this evaluation script.",
                  "Pooled CT errors use fold-specific training-fitted latent spaces; different projection ranks define different targets.",
                  "Permutation sensitivity does not establish predictive benefit or causal effects."]}
    private = {"schema": report["schema"], "clinical_baseline": baseline, "cases": {}}
    for case in cases:
        parts, ct_parts, records = [], [], []
        for fold in folds:
            path = Path(runs_dir) / case / f"fold-{fold['index']}" / "inference.pt"
            predictor = Predictor(path, device=device)
            contract = verify_run(predictor, path, fold)
            values, ct = model_outer(predictor.model, fold, samples, batch_size, seed, permutation_seed)
            parts.append(values)
            ct_parts.append(ct)
            records.append({"fold": fold["index"], "inner_training_patients": len(fold["rows"]["train"]),
                            "inner_selection_patients": len(fold["rows"]["validation"]),
                            "bundle_sha256": file_sha256(path), "contract_id": predictor.bundle["contract_id"],
                            "selected_epoch_zero_based": predictor.bundle["selected_epoch"],
                            "model_config": predictor.bundle["model_config"], "training_config": contract.get("train"),
                            "outer": summarize(values), "absolute_ct1": ct_summary([ct])})
            print(json.dumps({"case": case, "completed_fold": fold["index"], "outer_patients": len(values["patient_ids"])}), flush=True)
            del predictor
        pooled = pool_outer(parts, expected_ids)
        for key in ("binary", "pcr", "binary_valid", "pcr_valid", "prefix_valid"):
            if not torch.equal(pooled[key], baseline[key]):
                raise ValueError("Cases do not evaluate the same OOF targets and validity masks")
        report["cases"][case] = {"folds": records, "oof": summarize(pooled), "absolute_ct1": ct_summary(ct_parts)}
        private["cases"][case] = pooled
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(report, out / "report.json")
    atomic_save(private, out / "oof_probabilities.pt")
    for name in ("report.json", "oof_probabilities.pt"):
        os.chmod(out / name, 0o600)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--permutation-seed", type=int, default=2718)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    evaluate(args.folds, args.runs, args.cases, args.out, args.device, args.samples,
             args.batch_size, args.seed, args.permutation_seed)


if __name__ == "__main__":
    main()
