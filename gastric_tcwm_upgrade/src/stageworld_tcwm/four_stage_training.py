"""Train/validation-only, staged training of the explicitly weak four-stage model.

Real longitudinal concept supervision is not available in this experiment.
The auxiliary interface is latent; it must never be exported as measured biology.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import atomic_save, file_sha256, fingerprint, write_json
from .four_stage_model import FourStageModel, terminal_eligible
from .research_neural import INPUT_FIELDS, _cpu_tree, _metrics

EXPORT_SCHEMA = "weak-four-stage-export-v1"
DEFAULTS = {
    "hidden_dim": 8, "rank": 2, "dropout": .15, "batch_size": 8,
    "medical_condition_encoding": "masked_inapplicable",
    "auxiliary_max_epochs": 100, "auxiliary_learning_rate": .001,
    "auxiliary_pcr_weight": .2,
    "terminal_max_epochs": 100, "terminal_learning_rate": .0005,
    "warmup_epochs": 30, "patience_epochs": 20,
    "validation_interval_epochs": 1, "weight_decay": .05,
    "gradient_clip": 1., "min_delta": 1e-5,
}
TARGET_FIELDS = ("ct1", "pcr", "pcr_valid", "scan_event_index", "binary", "binary_valid")
EXPERIMENTS = {
    "ct_pcr_frozen": {"aux_ct_weight": 1., "aux_pcr_weight": .2,
                      "terminal_aux_weight": 0., "state_adapter_rank": 0,
                      "adaptation_learning_rate": None, "reuse_auxiliary": False},
    "pcr_only_frozen": {"aux_ct_weight": 0., "aux_pcr_weight": 1.,
                        "terminal_aux_weight": 0., "state_adapter_rank": 0,
                        "adaptation_learning_rate": None, "reuse_auxiliary": False},
    "ct_pcr_adapter": {"aux_ct_weight": 1., "aux_pcr_weight": .2,
                       "terminal_aux_weight": .1, "state_adapter_rank": 2,
                       "adaptation_learning_rate": .0001, "reuse_auxiliary": True},
}


def training_schedule(n_train, phase="auxiliary", diagnostic=False):
    """A full shuffled pass is an epoch; retain any final smaller batch."""
    if n_train < 1 or phase not in ("auxiliary", "terminal"):
        raise ValueError("Require training patients and a valid training phase")
    steps_per_epoch = math.ceil(n_train / DEFAULTS["batch_size"])
    maximum_steps = 2 if diagnostic else DEFAULTS[f"{phase}_max_epochs"] * steps_per_epoch
    return {"steps_per_epoch": steps_per_epoch, "maximum_steps": maximum_steps,
            "maximum_epochs": maximum_steps / steps_per_epoch,
            "validation_interval_steps": 1 if diagnostic else DEFAULTS["validation_interval_epochs"] * steps_per_epoch,
            "batch_size": DEFAULTS["batch_size"], "warmup_epochs": DEFAULTS["warmup_epochs"],
            "patience_epochs": DEFAULTS["patience_epochs"]}


@dataclass
class EpochEarlyStopping:
    """Select all checkpoints, but count patience only after the warm-up."""

    warmup_epochs: int = 30
    patience_epochs: int = 20
    min_delta: float = 1e-5
    best: float = float("inf")
    best_epoch: float = 0
    stale: int = 0
    last_epoch: float = -1

    def __post_init__(self):
        if self.warmup_epochs < 0 or self.patience_epochs < 1 or self.min_delta < 0:
            raise ValueError("Invalid epoch early-stopping settings")

    def update(self, epoch, score):
        if not math.isfinite(epoch) or epoch < 0 or epoch <= self.last_epoch:
            raise ValueError("Validation epochs must be finite and strictly increasing")
        if not math.isfinite(score):
            raise ValueError("Validation score must be finite")
        self.last_epoch = epoch
        improved = score < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch = score, epoch
        if epoch <= self.warmup_epochs or improved:
            self.stale = 0
        else:
            self.stale += 1
        return improved, epoch > self.warmup_epochs and self.stale >= self.patience_epochs


def _subset(batch, rows):
    return {key: value[rows] for key, value in batch.items()}


def _auxiliary_loss(model, output, batch, experiment="ct_pcr_frozen"):
    recipe = EXPERIMENTS[experiment]
    pcr_mask = model.pcr_mask(batch)
    zero = output["pcr_logits"].sum() * 0.
    # The pCR-only experiment never reads CT1 targets, including masked NaNs.
    ct, ct_patients = zero, 0
    if recipe["aux_ct_weight"]:
        targets = model.auxiliary_targets(batch)
        ct_mask = targets["ct1_mask"]
        ct_patients = int(ct_mask.sum())
        ct = (F.smooth_l1_loss(output["ct1_forecast"][ct_mask], targets["ct1_target"][ct_mask])
              if ct_mask.any() else zero)
    pcr = (F.binary_cross_entropy_with_logits(output["pcr_logits"][pcr_mask], batch["pcr"][pcr_mask].float())
           if pcr_mask.any() else zero)
    return recipe["aux_ct_weight"] * ct + recipe["aux_pcr_weight"] * pcr, {
        "ct1_smooth_l1": float(ct.detach()) if recipe["aux_ct_weight"] else None,
        "ct1_decoder_trained": bool(recipe["aux_ct_weight"]), "pcr_nll": float(pcr.detach()),
        "ct1_patients": ct_patients, "pcr_patients": int(pcr_mask.sum()),
    }


@torch.no_grad()
def _evaluate_auxiliary(model, batch, experiment="ct_pcr_frozen"):
    model.eval()
    output = model(batch)
    loss, metrics = _auxiliary_loss(model, output, batch, experiment)
    metrics["selection_loss"] = float(loss)
    mask = model.pcr_mask(batch)
    if mask.any():
        pcr_metrics = _metrics(batch["pcr"][mask], output["pcr_logits"][mask])
        metrics.update({f"pcr_{key}": pcr_metrics[key] for key in ("auc", "ap", "brier")})
    return metrics


@torch.no_grad()
def evaluate_four_stage(model, batch):
    """Score factual S3 only, preserving input row order; never need future targets."""
    model.eval()
    valid = batch["binary_valid"].bool() & terminal_eligible(batch)
    if not valid.all():
        raise ValueError("Four-stage evaluation requires all rows to have factual S3 labels")
    rows = torch.arange(len(valid), device=valid.device)
    logits = torch.cat([model(_subset(batch, chunk))["logits"] for chunk in rows.split(64)])
    metrics = _metrics(batch["binary"], logits)
    predictions = {"probability": logits.double().sigmoid().cpu(),
                   "labels": batch["binary"].cpu(), "logits": logits.cpu()}
    if "row_index" in batch:
        predictions["row_index"] = batch["row_index"].cpu()
    return metrics, predictions


def load_four_stage_export(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("Unsupported four-stage export schema")
    if payload.get("semantics") != "weak_latent" or payload.get("concept_validated") is not False:
        raise ValueError("This implementation cannot load or claim validated concept semantics")
    config = dict(payload["config"])
    # Old exports must retain their original transition function; silently
    # changing their encoding would invalidate checkpoint reproduction.
    config.setdefault("medical_condition_encoding", "legacy_one_hot")
    model = FourStageModel(**config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


def _validate_partitions(cohort, train_indices, validation_indices, diagnostic):
    rows = {}
    for role, supplied in (("train", train_indices), ("validation", validation_indices)):
        value = torch.as_tensor(supplied, dtype=torch.long).cpu()
        if value.ndim != 1 or not len(value) or len(value.unique()) != len(value):
            raise ValueError("Partitions must contain unique nonempty one-dimensional indices")
        if value.min() < 0 or value.max() >= len(cohort.ids):
            raise ValueError("Patient row outside cohort")
        rows[role] = value
    if set(rows["train"].tolist()) & set(rows["validation"].tolist()):
        raise ValueError("Training and validation overlap")
    train_ids = [cohort.ids[int(i)] for i in rows["train"]]
    fitted = cohort.encoders.get("fit_ids", [])
    if len(fitted) != len(train_ids) or set(fitted) != set(train_ids):
        raise ValueError("Clinical preprocessing must fit exactly the training patients")
    if not diagnostic and (len(cohort.ids), len(rows["train"]), len(rows["validation"])) != (651, 456, 65):
        raise ValueError("Formal training requires the fixed 651/456/65 cohort")
    return rows


def _reuse_auxiliary(model, batches, run, checkpoint_path, out, generator):
    """Import the paired, selected CT+pCR state; never replay its training."""
    path = Path(checkpoint_path)
    selected = torch.load(path, map_location="cpu", weights_only=True)
    parent_path = path.parent / "metrics.json"
    parent = json.loads(parent_path.read_text())
    if (selected.get("schema") != EXPORT_SCHEMA or selected.get("phase") != "auxiliary"
            or selected.get("seed") != run["seed"] or parent.get("seed") != run["seed"]):
        raise ValueError("Auxiliary checkpoint schema, phase or seed mismatch")
    for payload in (selected, parent):
        if (payload.get("partition_sha256") != run["partition_sha256"]
                or payload.get("partition_counts") != run["partition_counts"]):
            raise ValueError("Auxiliary checkpoint partition mismatch")
        if (payload.get("semantics") != "weak_latent" or payload.get("concept_validated") is not False
                or payload.get("experiment", "ct_pcr_frozen") != "ct_pcr_frozen"):
            raise ValueError("Require the paired weak CT+pCR frozen control")
    expected_config = dict(model.config, state_adapter_rank=0)
    source_config = dict(selected["config"])
    source_config.setdefault("state_adapter_rank", 0)
    if source_config != expected_config or source_config["medical_condition_encoding"] != "masked_inapplicable":
        raise ValueError("Auxiliary checkpoint model config mismatch")
    if selected["step"] != parent["auxiliary"]["selected_step"]:
        raise ValueError("Auxiliary checkpoint was not the parent's selected auxiliary state")
    # Buffers fitted here use this run's training patients. Compare before load
    # so imported preprocessing cannot silently come from another membership.
    for name, value in model.named_buffers():
        if not torch.equal(value.detach().cpu(), selected["model"][name]):
            raise ValueError(f"Auxiliary training statistics mismatch: {name}")
    missing, unexpected = model.load_state_dict(selected["model"], strict=False)
    adapter_keys = {name for name in model.state_dict() if name.startswith("state_adapter.")}
    if set(missing) != adapter_keys or unexpected:
        raise ValueError("Auxiliary checkpoint may omit only the new zero-initialized adapter")
    if any(parameter.detach().count_nonzero() for parameter in model.risk_head.parameters()):
        raise ValueError("Auxiliary checkpoint must start at the fixed clinical endpoint reference")
    evaluated = {role: _evaluate_auxiliary(model, batch, "ct_pcr_adapter") for role, batch in batches.items()}
    source_summary = parent["auxiliary"]
    for role in batches:
        for key in ("ct1_smooth_l1", "pcr_nll", "selection_loss"):
            if abs(evaluated[role][key] - source_summary[role][key]) > 2e-6:
                raise ValueError("Imported auxiliary predictions did not reproduce")
    # The original terminal stage started after the *last* auxiliary epoch,
    # although its weights came from aux_best. Reuse that sampler/RNG boundary
    # so the paired terminal comparison sees the same minibatch sequence.
    last_path = path.parent / "aux_last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    if (last.get("phase") != "auxiliary" or last.get("seed") != run["seed"]
            or last.get("partition_sha256") != run["partition_sha256"]
            or last.get("step") != source_summary["completed_steps"]):
        raise ValueError("Auxiliary last-checkpoint sampler identity mismatch")
    generator.set_state(last["sampler_rng"])
    torch.set_rng_state(last["torch_rng"])
    if str(next(model.parameters()).device).startswith("cuda"):
        torch.cuda.set_rng_state_all(last["cuda_rng"])
    provenance = {"path": str(path.resolve()), "sha256": file_sha256(path),
                  "parent_metrics_path": str(parent_path.resolve()), "parent_metrics_sha256": file_sha256(parent_path),
                  "source_selected_epoch": selected["epoch"], "source_selected_step": selected["step"],
                  "source_completed_epochs": source_summary["completed_epochs"],
                  "sampler_checkpoint_path": str(last_path.resolve()), "sampler_checkpoint_sha256": file_sha256(last_path),
                  "terminal_sampler_policy": "restore_parent_aux_last_rng; same paired terminal minibatch sequence",
                  "executed_steps_this_run": 0, "reproduced": True}
    run["auxiliary_source"] = provenance
    summary = {**copy.deepcopy(source_summary), **evaluated, "reused": True,
               "executed_steps_this_run": 0, "executed_epochs_this_run": 0,
               "stop_reason": "reused_selected_checkpoint", "source": provenance}
    imported = {**run, "phase": "auxiliary", "step": selected["step"], "epoch": selected["epoch"],
                "model": _cpu_tree(model.state_dict()), "reused": True,
                "snapshot_kind": "imported_selected_auxiliary; not a new last-training snapshot",
                "executed_steps_this_run": 0}
    for name in ("aux_best.pt", "aux_last.pt"):
        atomic_save(imported, out / name)
    write_json([{"step": selected["step"], "epoch": selected["epoch"], "reused": True, **evaluated}],
               out / "aux_history.json")
    return summary


def train_four_stage(cohort, train_indices, validation_indices, out_dir,
                     seed=17, device="cpu", diagnostic=False,
                     experiment="ct_pcr_frozen", auxiliary_checkpoint=None):
    """Fit both stages without reading test rows. The parent freezes all exports."""
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown four-stage experiment")
    recipe = copy.deepcopy(EXPERIMENTS[experiment])
    if bool(auxiliary_checkpoint is not None) != recipe["reuse_auxiliary"]:
        raise ValueError("Only ct_pcr_adapter requires a paired auxiliary checkpoint")
    out = Path(out_dir)
    rows = _validate_partitions(cohort, train_indices, validation_indices, diagnostic)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Refuse to overwrite a four-stage training attempt")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    fields = tuple(dict.fromkeys(INPUT_FIELDS + TARGET_FIELDS))
    batches = {role: {key: cohort.tensors[key][index].to(device) for key in fields}
               for role, index in rows.items()}
    for batch in batches.values():
        if not (terminal_eligible(batch) & batch["binary_valid"]).all():
            raise ValueError("Require complete factual labelled S3 paths in both partitions")
        if not torch.isfinite(batch["clinical"]).all() or not torch.isfinite(batch["ct0"]).all():
            raise ValueError("Nonfinite permitted prediction inputs")
    model = FourStageModel(image_dim=batches["train"]["ct0"].shape[-1],
                           hidden_dim=DEFAULTS["hidden_dim"], rank=DEFAULTS["rank"],
                           dropout=DEFAULTS["dropout"],
                           medical_condition_encoding=DEFAULTS["medical_condition_encoding"],
                           state_adapter_rank=recipe["state_adapter_rank"]).to(device)
    initialization = model.fit_statistics(batches["train"])
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count >= 8000:
        raise ValueError("Weak four-stage candidate exceeds the declared 8k parameter budget")
    for batch in batches.values():
        if not model.pcr_mask(batch).any():
            raise ValueError("Auxiliary stage requires applicable factual S1 pCR")
        if recipe["aux_ct_weight"] and not model.auxiliary_targets(batch)["ct1_mask"].any():
            raise ValueError("CT auxiliary stage requires observed CT1")
    if not recipe["aux_ct_weight"]:
        model.ct1_decoder.requires_grad_(False)
    run = {"schema": EXPORT_SCHEMA, "semantics": "weak_latent", "concept_validated": False,
           "causal_strategy_validated": False, "intermediate_risk_interface_enabled": False,
           "seed": int(seed), "config": copy.deepcopy(model.config), "diagnostic": bool(diagnostic),
           "hyperparameters": copy.deepcopy(DEFAULTS), "parameter_count": parameter_count,
           "experiment": experiment, "experiment_recipe": recipe,
           "partition_counts": {key: len(value) for key, value in rows.items()},
           "partition_sha256": {key: fingerprint(sorted(cohort.ids[int(i)] for i in value))
                                for key, value in rows.items()},
           "test_evaluated": False, "initialization": initialization,
           "endpoint": "recorded binary recurrence at factual S3; no fixed followup horizon",
           "historical_test_exposure": True,
           "schedule_unit": "epoch", "steps_per_epoch": training_schedule(len(rows["train"]))["steps_per_epoch"],
           "early_stopping_policy": "validate each epoch; zero patience count through epoch30; count from epoch31",
           "training_strategy": ("paired S1 CT/pCR pretraining reused; small adapter and transitions jointly optimized with S3 and S1 losses"
                                 if recipe["reuse_auxiliary"] else
                                 "S1 pCR-only pretraining then frozen representation; S3-only terminal optimization"
                                 if not recipe["aux_ct_weight"] else
                                 "S1 CT/pCR pretraining, then frozen S0/S1 representation; S3-only terminal optimization"),
           "s2_s3_measurements_available": False,
           "information_boundary": "CT0 and baseline clinical; factual ordinal modality sequence; targets never enter forward",
           "stage_semantics": "four computational latent states, not clinically validated concept states"}
    start = time.monotonic()
    generator = torch.Generator().manual_seed(seed)
    aux_summary = None

    for phase in ("auxiliary", "terminal"):
        if phase == "auxiliary" and recipe["reuse_auxiliary"]:
            aux_summary = _reuse_auxiliary(model, batches, run, auxiliary_checkpoint, out, generator)
            run["auxiliary"] = aux_summary
            continue
        parameters = ([parameter for parameter in model.parameters() if parameter.requires_grad]
                      if phase == "auxiliary" else
                      model.configure_terminal_adaptation() if recipe["state_adapter_rank"] else model.freeze_representation())
        if not parameters:
            raise RuntimeError("No parameters selected for optimization")
        if phase == "terminal":
            frozen_before = {name: parameter.detach().cpu().clone()
                             for name, parameter in model.named_parameters() if not parameter.requires_grad}
            run["terminal_trainable_parameter_count"] = sum(p.numel() for p in parameters)
        optimizer_parameters = parameters
        if phase == "terminal" and recipe["state_adapter_rank"]:
            adaptation_names = ("state_adapter.", "medical_transition.")
            adapting, terminal = [], []
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    (adapting if name.startswith(adaptation_names) or name == "nac_gate" else terminal).append(parameter)
            optimizer_parameters = [{"params": adapting, "lr": recipe["adaptation_learning_rate"]},
                                    {"params": terminal, "lr": DEFAULTS["terminal_learning_rate"]}]
            run["terminal_parameter_groups"] = {
                "adaptation": {"parameters": sum(p.numel() for p in adapting), "lr": recipe["adaptation_learning_rate"]},
                "endpoint": {"parameters": sum(p.numel() for p in terminal), "lr": DEFAULTS["terminal_learning_rate"]}}
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=DEFAULTS[f"{phase}_learning_rate"],
                                      weight_decay=DEFAULTS["weight_decay"])
        schedule = training_schedule(len(rows["train"]), phase, diagnostic)
        maximum = schedule["maximum_steps"]
        interval = schedule["validation_interval_steps"]
        steps_per_epoch = schedule["steps_per_epoch"]
        stopping = EpochEarlyStopping(DEFAULTS["warmup_epochs"], DEFAULTS["patience_epochs"], DEFAULTS["min_delta"])
        history, best, best_step, stale = [], float("inf"), 0, 0
        permutation, cursor = torch.empty(0, dtype=torch.long), 0
        best_file = out / ("aux_best.pt" if phase == "auxiliary" else "best.pt")
        last_file = out / ("aux_last.pt" if phase == "auxiliary" else "last.pt")
        history_file = out / ("aux_history.json" if phase == "auxiliary" else "history.json")

        def checkpoint(step):
            return {**run, "phase": phase, "step": int(step), "epoch": step / steps_per_epoch,
                    "training_schedule": schedule, "model": _cpu_tree(model.state_dict()),
                    "optimizer": _cpu_tree(optimizer.state_dict()), "history": copy.deepcopy(history),
                    "best_step": best_step, "best_score": best, "stale_checks": stale,
                    "best_epoch": best_step / steps_per_epoch, "early_stopping": copy.deepcopy(vars(stopping)),
                    "torch_rng": torch.get_rng_state(), "sampler_rng": generator.get_state(),
                    "sampler_permutation": permutation.clone(), "sampler_cursor": cursor,
                    "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []}

        for step in range(maximum + 1):
            minibatch_loss = None
            if step:
                model.train()
                if cursor >= len(permutation):
                    permutation = torch.randperm(len(rows["train"]), generator=generator)
                    cursor = 0
                local = permutation[cursor:cursor + DEFAULTS["batch_size"]]
                cursor += len(local)
                batch = _subset(batches["train"], local.to(device))
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                loss = (_auxiliary_loss(model, output, batch, experiment)[0] if phase == "auxiliary" else
                        F.binary_cross_entropy_with_logits(output["logits"], batch["binary"].float()))
                if phase == "terminal" and recipe["terminal_aux_weight"]:
                    loss = loss + recipe["terminal_aux_weight"] * _auxiliary_loss(model, output, batch, experiment)[0]
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite four-stage loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, DEFAULTS["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                minibatch_loss = float(loss.detach())
            if step % interval and step != maximum:
                continue
            if phase == "auxiliary":
                evaluated = {role: _evaluate_auxiliary(model, batch, experiment) for role, batch in batches.items()}
                score = evaluated["validation"]["selection_loss"]
            else:
                evaluated = {role: evaluate_four_stage(model, batch)[0] for role, batch in batches.items()}
                score = evaluated["validation"]["nll"]
            epoch = step / steps_per_epoch
            improved, should_stop = stopping.update(epoch, score)
            best, stale = stopping.best, stopping.stale
            history.append({"step": step, "epoch": epoch, **evaluated,
                            "minibatch_loss": minibatch_loss, "stale_epochs": stale})
            if phase == "terminal" and recipe["terminal_aux_weight"]:
                history[-1]["auxiliary"] = {role: _evaluate_auxiliary(model, batch, experiment)
                                             for role, batch in batches.items()}
            if improved:
                best_step = step
                atomic_save(checkpoint(step), best_file)
            atomic_save(checkpoint(step), last_file)
            write_json(history, history_file)
            write_json({"status": "running", "phase": phase, "step": step, "epoch": epoch,
                        "best_step": best_step, "best_epoch": best_step / steps_per_epoch,
                        "stale_epochs": stale, "steps_per_epoch": steps_per_epoch,
                        "maximum_epochs": schedule["maximum_epochs"],
                        "best_validation_loss": best, "elapsed_seconds": time.monotonic()-start}, out / "status.json")
            if not diagnostic and should_stop:
                break
        phase_stop_reason = ("diagnostic_steps" if diagnostic else
                             "max_epochs" if step == maximum else "early_stopping")
        selected = torch.load(best_file, map_location=device, weights_only=True)
        model.load_state_dict(selected["model"])
        if phase == "auxiliary":
            aux_summary = {"selected_step": best_step, "completed_steps": step,
                           "selected_epoch": best_step / steps_per_epoch,
                           "completed_epochs": step / steps_per_epoch,
                           "steps_per_epoch": steps_per_epoch, "stop_reason": phase_stop_reason,
                           "reused": False, "executed_steps_this_run": step, "executed_epochs_this_run": step / steps_per_epoch,
                           "train": _evaluate_auxiliary(model, batches["train"], experiment),
                           "validation": _evaluate_auxiliary(model, batches["validation"], experiment)}
            run["auxiliary"] = aux_summary
        else:
            for name, parameter in model.named_parameters():
                if name in frozen_before and not torch.equal(parameter.detach().cpu(), frozen_before[name]):
                    raise RuntimeError("Terminal training changed a frozen representation parameter")

    metrics = {role: evaluate_four_stage(model, batch)[0] for role, batch in batches.items()}
    selected_kind = "clinical_baseline_at_terminal_step0" if best_step == 0 else "trained_weak_four_stage"
    result = {**run, **metrics, "selected_step": best_step, "completed_steps": step,
              "selected_epoch": best_step / steps_per_epoch, "completed_epochs": step / steps_per_epoch,
              "selected_kind": selected_kind, "terminal_dynamics_selected": best_step > 0,
              "elapsed_seconds": time.monotonic()-start,
              "stop_reason": phase_stop_reason}
    result["terminal_auxiliary"] = {role: _evaluate_auxiliary(model, batch, experiment) for role, batch in batches.items()}
    payload = {**run, "model": _cpu_tree(model.state_dict()), "metrics": result,
               "selected_step": best_step, "selected_epoch": best_step / steps_per_epoch,
               "selected_kind": selected_kind}
    atomic_save(payload, out / "inference.pt")
    write_json(result, out / "metrics.json")
    write_json({"status": "completed", "auxiliary_selected_step": aux_summary["selected_step"],
                "auxiliary_selected_epoch": aux_summary["selected_epoch"],
                "auxiliary_completed_epochs": aux_summary["completed_epochs"],
                "selected_step": best_step, "completed_steps": step,
                "selected_epoch": best_step / steps_per_epoch, "completed_epochs": step / steps_per_epoch,
                "steps_per_epoch": steps_per_epoch, "stop_reason": phase_stop_reason,
                "selected_kind": selected_kind, "elapsed_seconds": time.monotonic()-start}, out / "status.json")
    return result
