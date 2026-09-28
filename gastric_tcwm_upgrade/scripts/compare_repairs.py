#!/usr/bin/env python
"""Compare selected binary bundles on train/validation with fixed CT1 probes."""
from pathlib import Path
import argparse
import json
import os
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stageworld_tcwm.data import Cohort, file_sha256, fingerprint, split_indices, write_json
from stageworld_tcwm.evaluation import binary_metrics, evaluate_predictions
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.survival import mixture_binary_nll


def ct1_permutation(present, seed):
    """A fixed cycle permutes available observations without moving their masks."""
    permutation = torch.arange(len(present))
    available = present.nonzero().flatten()
    if len(available) > 1:
        generator = torch.Generator().manual_seed(seed)
        order = available[torch.randperm(len(available), generator=generator)]
        permutation[order] = order.roll(1)
    return permutation


def distribution(value):
    value = value.double()
    if not value.numel():
        return {"n": 0}
    return {"n": value.numel(), "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)), "min": float(value.min()),
            "max": float(value.max())}


def differences(left, right):
    if not left.numel():
        return {"n": 0}
    delta = (left - right).double()
    return {"n": delta.numel(), "mean_absolute": float(delta.abs().mean()),
            "maximum_absolute": float(delta.abs().max()),
            "root_mean_square": float(delta.square().mean().sqrt())}


@torch.inference_mode()
def collect_with_probe(model, cohort, rows, samples, batch_size, seed, permutation_seed):
    model.eval()
    device = next(model.parameters()).device
    present = cohort.tensors["image_valid"][rows, 1]
    permutation = ct1_permutation(present, permutation_seed)
    pieces = {key: [] for key in ("logits", "pcr_logits", "permuted_logits", "permuted_pcr_logits")}
    for start in range(0, len(rows), batch_size):
        local = slice(start, start + batch_size)
        batch = cohort.batch(rows[local], device)
        original = model(batch, samples=samples, seed=seed + start, compute_aux=True)
        modified = dict(batch)
        source_rows = rows[permutation[local]]
        modified["ct1"] = cohort.tensors["ct1"][source_rows].to(device)
        permuted = model(modified, samples=samples, seed=seed + start, compute_aux=True)
        for prefix, output in (("", original), ("permuted_", permuted)):
            for key, output_key in (("logits", "predictions"), ("pcr_logits", "pcr_logits")):
                value = output[output_key].detach().float().cpu()
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite comparison predictions")
                pieces[prefix + key].append(value)
    return {key: torch.cat(values) for key, values in pieces.items()}, {
        "available_ct1": int(present.sum()),
        "observations_moved": int((permutation != torch.arange(len(rows))).sum()),
        "seed": permutation_seed, "policy": "within_partition_available_CT1_cycle_only",
    }


def role_report(model, cohort, rows, role, samples, batch_size, seed, permutation_seed):
    if role not in ("train", "validation"):
        raise ValueError("Repair comparisons are restricted to train and validation")
    prediction, permutation = collect_with_probe(
        model, cohort, rows, samples, batch_size, seed, permutation_seed)
    batch = cohort.batch(rows)
    report = evaluate_predictions(prediction["logits"], cohort, rows, model.cfg)
    probability = prediction["logits"].sigmoid().mean(2)
    permuted = prediction["permuted_logits"].sigmoid().mean(2)
    for stage in range(3):
        eligible = batch["prefix_valid"][:, stage]
        report["stages"][f"S{stage}"]["probability_distribution"] = distribution(probability[eligible, stage])
    pcr_valid = batch["pcr_valid"]
    pcr = prediction["pcr_logits"].sigmoid().mean(1)
    pcr_permuted = prediction["permuted_pcr_logits"].sigmoid().mean(1)
    report["pcr"] = binary_metrics(batch["pcr"][pcr_valid].numpy(), pcr[pcr_valid].numpy())
    if pcr_valid.any():
        report["pcr"]["nll"] = float(mixture_binary_nll(
            prediction["pcr_logits"][pcr_valid], batch["pcr"][pcr_valid]).mean())
    report["pcr"]["probability_distribution"] = distribution(pcr[pcr_valid])
    report["pcr"]["interpretation"] = "auxiliary_preoperative_prior_head"
    stage_pair = batch["prefix_valid"][:, 0] & batch["prefix_valid"][:, 1]
    report["s0_s1_difference"] = differences(probability[stage_pair, 0], probability[stage_pair, 1])
    report["ct1_permutation"] = {
        **permutation,
        "stages": {f"S{stage}": differences(
            probability[batch["prefix_valid"][:, stage], stage],
            permuted[batch["prefix_valid"][:, stage], stage]) for stage in range(3)},
        "pcr": differences(pcr[pcr_valid], pcr_permuted[pcr_valid]),
        "same_monte_carlo_noise_as_original": True,
        "interpretation": "sensitivity_probe_not_an_accuracy_or_causal_effect_estimate",
    }
    return report


def verify_contract(bundle, bundle_path, cohort_hash, split_hash, training_ids):
    contract_path = bundle_path.parent / "contract.json"
    if not contract_path.is_file():
        raise ValueError("Place the inference bundle beside its training contract.json")
    document = json.loads(contract_path.read_text())
    if document.get("id") != bundle.get("contract_id"):
        raise ValueError("Inference bundle and training contract do not match")
    contract = document["contract"]
    if contract.get("cohort_sha256") != cohort_hash or contract.get("split") != split_hash:
        raise ValueError("Run was fitted on a different cohort or patient split")
    fit_ids = bundle.get("encoders", {}).get("fit_ids")
    if fit_ids is not None and set(fit_ids) != set(training_ids):
        raise ValueError("Bundle encoders were fitted on a different training partition")
    return contract


def parse_runs(values):
    runs = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        path = Path(raw_path if separator else value).expanduser().resolve()
        if path.is_dir():
            path = path / "inference.pt"
        if not separator:
            name = path.parent.name
        if not name or name in runs:
            raise ValueError("Each run needs a unique nonempty name, e.g. original=/path/to/run")
        runs[name] = path
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--permutation-seed", type=int, default=2718)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size < 1 or args.threads < 1:
        parser.error("Need samples >= 2, batch-size >= 1, threads >= 1")
    os.umask(0o077)
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    cohort = Cohort.load(args.data)
    split = json.loads(args.split.read_text())
    roles = split_indices(cohort, split)
    cohort_hash, split_hash = file_sha256(args.data), fingerprint(split)
    report = {
        "schema": "tcwm-repair-comparison-v1", "test_scored": False,
        "cohort_sha256": cohort_hash, "split_fingerprint": split_hash,
        "samples": args.samples, "batch_size": args.batch_size,
        "seed": args.seed, "permutation_seed": args.permutation_seed,
        "device": args.device, "precision": "float32_no_autocast",
        "limitations": [
            "Training metrics are resubstitution estimates.",
            "Validation was used for model development and checkpoint selection.",
            "Permutation sensitivity is not proof of useful prediction or a causal effect.",
            "Changing evaluation batch size changes Monte Carlo noise assignments.",
        ], "runs": {},
    }
    for name, path in parse_runs(args.run).items():
        predictor = Predictor(path, device=args.device)
        if predictor.cfg.endpoint != "binary":
            raise ValueError("This repair comparison supports recorded binary endpoints only")
        contract = verify_contract(predictor.bundle, path, cohort_hash, split_hash, split["train"])
        item = {
            "bundle_sha256": file_sha256(path), "contract_id": predictor.bundle["contract_id"],
            "selected_epoch_zero_based": predictor.bundle["selected_epoch"],
            "model_config": predictor.bundle["model_config"],
            "training_config": contract.get("train"),
            "parameter_count": sum(parameter.numel() for parameter in predictor.model.parameters()),
        }
        for role in ("train", "validation"):
            item[role] = role_report(predictor.model, cohort, roles[role], role,
                args.samples, args.batch_size, args.seed, args.permutation_seed)
            print(json.dumps({"run": name, "completed_role": role,
                              "selection_nll": item[role]["selection_nll"]}), flush=True)
        report["runs"][name] = item
        print(json.dumps({"run": name, "validation_nll": item["validation"]["selection_nll"],
                          "validation_S0_AUROC": item["validation"]["stages"]["S0"]["auroc"]}), flush=True)
    write_json(report, args.out)


if __name__ == "__main__":
    main()
