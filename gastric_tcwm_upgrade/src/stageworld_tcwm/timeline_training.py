"""Locked, restartable fixed-split and historical nested-fold timeline training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import atomic_save, file_sha256, fingerprint, write_json
from .timeline_losses import (FixedCTMoments, eligible_query_mask, patient_query_bce,
                              pcr_label_mask, report_concept_mask, report_training_mask,
                              scan_valid, timeline_loss)


@dataclass
class TimelineTrainConfig:
    seed: int = 17
    batch_size: int = 8
    accumulation_steps: int = 2
    learning_rate: float = 1e-4
    image_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_optimizer_steps: int = 500
    validation_interval: int = 50
    checkpoint_interval: int = 10
    minimum_optimizer_steps: int = 100
    patience: int = 5
    min_delta: float = 1e-5
    forecast_weight: float = 0.1
    pcr_weight: float = 0.1
    alignment_weight: float = 0.0
    report_concept_weight: float = 0.0
    assimilation_weight: float = 0.0
    drift_weight: float = 1e-4
    device: str = "cuda"
    evaluate_test: bool = True
    validate_initial: bool = False
    freeze_image_projection: bool = False
    collect_full_train_metrics: bool = False

    def validate(self):
        for name in ("evaluate_test", "validate_initial", "freeze_image_projection", "collect_full_train_metrics"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("batch_size", "accumulation_steps", "max_optimizer_steps", "validation_interval", "checkpoint_interval", "minimum_optimizer_steps", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "image_learning_rate", "weight_decay", "gradient_clip", "min_delta",
                     "forecast_weight", "pcr_weight", "alignment_weight", "report_concept_weight",
                     "assimilation_weight", "drift_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.minimum_optimizer_steps > self.max_optimizer_steps:
            raise ValueError("Minimum optimizer steps exceeds budget")
        if self.assimilation_weight != 0:
            raise ValueError("CT1 assimilation is disabled in this audited first round")
        if any(getattr(self, name) < 0 for name in ("forecast_weight", "pcr_weight", "alignment_weight", "report_concept_weight",
                                                 "drift_weight", "weight_decay", "min_delta")):
            raise ValueError("Loss and regularization weights must be nonnegative")
        if min(self.learning_rate, self.image_learning_rate, self.gradient_clip) <= 0:
            raise ValueError("Learning rates and clipping must be positive")
        return self


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class StatefulPatientSampler:
    """Fixed-size batches spanning reshuffles, with an explicit recovery cursor."""

    def __init__(self, indices, seed):
        self.indices = torch.as_tensor(indices, dtype=torch.long).cpu()
        if not len(self.indices):
            raise ValueError("Training patients are empty")
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(self.indices), generator=self.generator)
        self.position, self.epochs = 0, 0

    def next(self, count):
        chunks, remaining = [], count
        while remaining:
            if self.position == len(self.order):
                self.order = torch.randperm(len(self.indices), generator=self.generator)
                self.position = 0
                self.epochs += 1
            take = min(remaining, len(self.order) - self.position)
            chunks.append(self.indices[self.order[self.position:self.position + take]])
            self.position += take
            remaining -= take
        return torch.cat(chunks)

    def state_dict(self):
        return {"indices": self.indices, "order": self.order, "position": self.position,
                "epochs": self.epochs, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if not torch.equal(self.indices, state["indices"]):
            raise ValueError("Recovery sampler patients differ from the locked split")
        if (sorted(state["order"].tolist()) != list(range(len(self.indices)))
                or not 0 <= state["position"] <= len(self.indices)):
            raise ValueError("Recovery sampler order/cursor is invalid")
        self.order, self.position, self.epochs = state["order"], state["position"], state["epochs"]
        self.generator.set_state(state["generator"])


def source_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): file_sha256(p)
            for directory in (root / "src", root / "scripts") for p in sorted(directory.rglob("*.py"))}


def verify_source(root, expected):
    if source_manifest(root) != expected:
        raise ValueError("Source changed during the locked study; use a new study directory")


def optimizer_groups(model, config):
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    image_prefixes = ("image.", "position.")
    groups, manifest = [], []
    for image, name, lr in ((True, "image_projection", config.image_learning_rate), (False, "event_state_readout", config.learning_rate)):
        selected = [(key, parameter) for key, parameter in named if key.startswith(image_prefixes) == image]
        if selected:
            groups.append({"params": [p for _, p in selected], "lr": lr})
            manifest.append({"name": name, "learning_rate": lr, "parameters": [k for k, _ in selected],
                             "parameter_count": sum(p.numel() for _, p in selected)})
    has_image = any(row["name"] == "image_projection" for row in manifest)
    if not has_image and not config.freeze_image_projection:
        raise ValueError("Cannot identify the image projection for its lower learning rate")
    if has_image and config.freeze_image_projection:
        raise ValueError("Image projection parameters must be frozen before optimizer construction")
    identities = [id(p) for group in groups for p in group["params"]]
    if len(identities) != len(set(identities)) or set(identities) != {id(p) for _, p in named}:
        raise ValueError("Optimizer contains duplicate or omitted parameters")
    return groups, manifest


def checkpoint_head_status(model_config, training_config, optimizer_steps):
    """Describe the selected checkpoint, separately from fitted baseline priors."""
    trained = optimizer_steps > 0
    return {"optimizer_steps": int(optimizer_steps), "world_model_trained": trained,
            "terminal_head_trained": trained,
            "clinical_anchor_fitted": bool(getattr(model_config, "terminal_clinical_anchor", False)),
            "pcr_head_trained": trained and training_config.pcr_weight > 0,
            "forecast_head_trained": trained and training_config.forecast_weight > 0,
            "s1_alignment_trained": trained and training_config.alignment_weight > 0,
            "report_concept_head_trained": (trained and getattr(model_config, "s1_report_concepts", False)
                                            and training_config.report_concept_weight > 0)}


def set_head_training_status(model, status):
    model.prediction_head_status = dict(status)
    model.report_concept_head_trained = status["report_concept_head_trained"]
    return model


def gradient_health(model):
    branches = {}
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            if parameter.requires_grad:
                missing.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Nonfinite gradient in {name}")
        key = name.split(".")[0]
        branches[key] = branches.get(key, 0.0) + float(parameter.grad.detach().float().square().sum())
    norms = {key: math.sqrt(value) for key, value in branches.items()}
    if not norms or max(norms.values()) <= 0:
        raise FloatingPointError("No nonzero gradients reached model parameters")
    return {"all_finite": True, "branch_norms": norms, "missing_gradient_parameters": missing}


def binary_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not len(labels):
        return {"patients": 0, "auc": None, "average_precision": None, "brier": None, "nll": None}
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {"patients": int(len(labels)),
            "auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            "average_precision": float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "nll": float(np.mean(-labels * np.log(p) - (1 - labels) * np.log1p(-p)))}


def validate_report_concept_contract(metadata):
    from .report_concept_data import S1_CONCEPT_NAMES, S1_CONCEPT_TRANSFORMS

    contract = metadata.get("s1_report_concepts")
    if not isinstance(contract, dict):
        raise ValueError("Report concepts require an explicit verified source metadata contract")
    required = {"schema": "s1-report-concepts-v1", "schema_version": 1,
                "target_stage": 1, "available_stage": 2,
                "prediction_cutoff": "post_neoadjuvant_pre_resection_s1_prior"}
    if (any(contract.get(key) != value for key, value in required.items())
            or contract.get("validated_source") is not True
            or contract.get("weak_supervision") is not True
            or contract.get("clinically_adjudicated") is not False
            or contract.get("forbidden_as_forward_input") is not True
            or contract.get("names") != list(S1_CONCEPT_NAMES)
            or contract.get("transforms") != list(S1_CONCEPT_TRANSFORMS)):
        raise ValueError("Report concept metadata does not match the verified weak S1 supervision contract")
    return contract


def report_concept_metrics(labels, predictions, targets, trained):
    from .report_concept_data import S1_CONCEPT_NAMES, S1_CONCEPT_TRANSFORMS

    result = {"weak_supervision": True, "clinically_adjudicated": False,
              "full_longitudinal_concepts": False, "target_stage": 1, "available_stage": 2,
              "report_concept_head_trained": bool(trained),
              "head_role": "auxiliary_prediction" if trained else "untrained_auxiliary_control",
              "per_target": {}}
    for column, (name, transform) in enumerate(zip(S1_CONCEPT_NAMES, S1_CONCEPT_TRANSFORMS, strict=True)):
        actual, predicted = np.asarray(labels[column]), np.asarray(predictions[column])
        mean, scale = float(targets.report_mean[column]), float(targets.report_scale[column])
        row = {"patients": len(actual), "training_patients": int(targets.report_counts[column]),
               "target_transform": transform, "training_constant": mean}
        if column == 0:
            row.update(model=binary_metrics(actual, predicted),
                       training_constant_baseline=binary_metrics(actual, np.full(len(actual), mean)))
        else:
            def errors(value):
                return {"transformed_mae": float(np.mean(np.abs(value - actual))) if len(actual) else None,
                        "normalized_mse": float(np.mean(((value - actual) / scale) ** 2)) if len(actual) else None}
            row.update(model=errors(predicted), training_constant_baseline=errors(np.full(len(actual), mean)))
        result["per_target"][name] = row
    return result


@torch.no_grad()
def collect_predictions(model, cohort, indices, batch_size, device, targets=None,
                        report_concept_head_trained=None):
    previous, rng = model.training, capture_rng()
    model.eval()
    rows, query_scores, forecast_errors, anchor_errors = [], [], [], []
    pcr_labels, pcr_probabilities = [], []
    anchor_labels, anchor_probabilities, residual_logits = [], [], []
    report_labels, report_predictions = [[] for _ in range(4)], [[] for _ in range(4)]
    report_enabled = targets is not None and targets.report_concepts
    head_status = getattr(model, "prediction_head_status", None)
    initial = head_status is not None and head_status["optimizer_steps"] == 0
    pcr_available = head_status is None or head_status["pcr_head_trained"]
    forecast_available = head_status is None or head_status["forecast_head_trained"]
    anchor_available = head_status is None or head_status["s1_alignment_trained"]
    objective, pcr_cutoff = None, None
    try:
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            batch = cohort.batch(selected, device)
            output = model(batch)
            batch_objective = output.get("objective", "legacy_multistage")
            if objective is not None and batch_objective != objective:
                raise ValueError("Evaluation objective changed between patient batches")
            objective = batch_objective
            logits = output["logits"]
            mask = eligible_query_mask(output, batch)
            if not torch.isfinite(logits[mask]).all():
                raise FloatingPointError("Nonfinite query prediction during evaluation")
            probabilities = logits.sigmoid().cpu()
            if batch_objective == "terminal_state_v1" and getattr(getattr(model, "cfg", None), "terminal_clinical_anchor", False):
                reference = model.outcome.clinical_anchor_logits(batch["clinical"])
                observed_terminal = mask & batch["binary_valid"][:, None]
                expanded_reference = reference[:, None].expand_as(logits)
                anchor_labels.extend(batch["binary"][:, None].expand_as(logits)[observed_terminal].cpu().tolist())
                anchor_probabilities.extend(expanded_reference[observed_terminal].sigmoid().cpu().tolist())
                residual_logits.extend((logits-expanded_reference)[observed_terminal].cpu().tolist())
            for local, index in enumerate(selected.tolist()):
                observed = bool(batch["binary_valid"][local])
                label = float(batch["binary"][local]) if observed else None
                patient_logits = logits[local:local + 1]
                if observed and mask[local].any():
                    value = patient_query_bce(patient_logits, batch["binary"][local:local + 1],
                                              mask[local:local + 1], batch["binary_valid"][local:local + 1])
                    query_scores.append(float(value))
                for query in range(mask.shape[1]):
                    if not bool(mask[local, query]):
                        continue
                    row = {"patient_id": cohort.ids[index], "query_index": query,
                           "probability": float(probabilities[local, query]), "label": label,
                           "endpoint": "recorded_binary_recurrence"}
                    if objective == "terminal_state_v1":
                        row.update(query_name="terminal", prediction_semantics="retrospective_terminal_recorded_status")
                    for key in ("query_order", "query_event_index", "query_prospective", "query_available", "query_replay"):
                        if key in batch:
                            row[key] = batch[key][local, query].item()
                    for key in ("retrospective", "hypothetical"):
                        if key in output:
                            row[key] = bool(output[key][local, query])
                    rows.append(row)
            pcr_cutoff = output.get("pcr_cutoff", "baseline")
            if "pcr_logits" in output and pcr_available:
                pcr_valid = pcr_label_mask(output, batch)
                pcr_cutoff = output.get("pcr_cutoff", "baseline")
                if not torch.isfinite(output["pcr_logits"][pcr_valid]).all():
                    raise FloatingPointError("Nonfinite pCR prediction during evaluation")
                pcr_labels.extend(batch["pcr"][pcr_valid].cpu().tolist())
                pcr_probabilities.extend(output["pcr_logits"][pcr_valid].sigmoid().cpu().tolist())
            if report_enabled and not initial:
                concept_mask = report_concept_mask(output, batch)
                concept_logits = output["s1_concept_logits"]
                if concept_logits.shape != concept_mask.shape or not torch.isfinite(concept_logits[concept_mask]).all():
                    raise FloatingPointError("Invalid or nonfinite eligible report concept predictions")
                decoded = targets.decode_report(concept_logits)
                for column in range(4):
                    valid = concept_mask[:, column]
                    report_labels[column].extend(batch["s1_concepts"][valid, column].cpu().tolist())
                    report_predictions[column].extend(decoded[valid, column].cpu().tolist())
            if targets is not None and (forecast_available or anchor_available):
                valid = scan_valid(batch, output)
                if valid.any():
                    if forecast_available:
                        errors = (targets(output["forecast"][valid]) - targets(batch["ct1"][valid])).square().mean(-1)
                        forecast_errors.extend(errors.cpu().tolist())
                    if targets.anchor_dim is not None and "s1_anchor" in output and anchor_available:
                        errors = (output["s1_anchor"][valid].float() - targets.anchor(batch["ct1"][valid])).square().mean(-1)
                        anchor_errors.extend(errors.cpu().tolist())
        if not query_scores:
            raise ValueError("No labelled patients with eligible queries for evaluation")
        by_query = {}
        names = cohort.metadata.get("query_names", [])
        for query in sorted({row["query_index"] for row in rows}):
            selected_rows = [row for row in rows if row["query_index"] == query and row["label"] is not None]
            name = names[query] if query < len(names) else f"query_{query}"
            by_query[name] = binary_metrics([r["label"] for r in selected_rows], [r["probability"] for r in selected_rows])
        if objective == "terminal_state_v1":
            selected_rows = [row for row in rows if row["label"] is not None]
            by_query = {"terminal": binary_metrics([r["label"] for r in selected_rows],
                                                    [r["probability"] for r in selected_rows])}
        metrics = {"selection_nll": float(np.mean(query_scores)), "patients": len(query_scores),
                   "per_query": by_query, "forecast_moment_mse": float(np.mean(forecast_errors)) if forecast_errors else None,
                   "endpoint": "recorded_binary_recurrence", "calendar_horizon_supported": False}
        if head_status is not None:
            metrics["head_training_status"] = dict(head_status)
        if objective == "terminal_state_v1":
            metrics.update(objective=objective, selection_semantics="factual_terminal_patient_mean_BCE",
                           pcr_cutoff="post_neoadjuvant_s1", terminal_event_order=3,
                           terminal_boundary="last_treatment_summary_not_fixed_followup_horizon",
                           pcr={"cutoff": pcr_cutoff, **binary_metrics(pcr_labels, pcr_probabilities)},
                           s1_anchor_mse=float(np.mean(anchor_errors)) if anchor_errors else None,
                           s1_anchor_patients=len(anchor_errors),
                           clinical_causal_effects_identified=False)
        if report_enabled:
            trained = (report_concept_head_trained if report_concept_head_trained is not None
                       else getattr(model, "report_concept_head_trained", False))
            metrics["s1_report_concepts"] = report_concept_metrics(report_labels, report_predictions, targets, trained)
            metrics["report_concept_head_trained"] = bool(trained)
            if initial:
                metrics["s1_report_concepts"]["available"] = False
                metrics["s1_report_concepts"]["unavailable_reason"] = "selected_initial_checkpoint_without_head_training"
        if objective == "terminal_state_v1" and head_status is not None:
            metrics["pcr"].update(head_trained=pcr_available, available=pcr_available)
        baseline_probability = getattr(model, "training_recurrence_probability", None)
        if baseline_probability is not None:
            actual = [row["label"] for row in rows if row["label"] is not None]
            metrics["training_constant_baseline"] = binary_metrics(actual, np.full(len(actual), baseline_probability))
        if anchor_labels:
            residual = np.asarray(residual_logits)
            metrics["clinical_anchor_reference"] = binary_metrics(anchor_labels, anchor_probabilities)
            metrics["terminal_residual_logit_rms"] = float(np.sqrt(np.mean(residual ** 2)))
            metrics["terminal_residual_logit_std"] = float(residual.std())
            metrics["terminal_residual_logit_abs_max"] = float(np.abs(residual).max())
        return rows, metrics
    finally:
        restore_rng(rng)
        model.train(previous)


@torch.no_grad()
def query_consistency(model, batch):
    previous, rng = model.training, capture_rng()
    model.eval()
    try:
        full = model(batch)
        columns = [k for k, value in batch.items() if k.startswith("query_") and value.ndim >= 2]
        maximum, comparisons, requests = 0.0, 0, 0
        for index in range(full["logits"].shape[1]):
            valid = full["query_mask"][:, index].bool()
            if not valid.any():
                continue
            single = dict(batch)
            for key in columns:
                single[key] = batch[key][:, index:index + 1]
            candidate = model(single)
            if not torch.equal(candidate["query_mask"][:, 0].bool(), valid):
                raise AssertionError("Query subset changed the eligibility of a terminal/history query")
            difference = candidate["logits"][:, 0][valid] - full["logits"][:, index][valid]
            if not torch.isfinite(difference).all():
                raise FloatingPointError("Nonfinite eligible prediction during query consistency check")
            maximum = max(maximum, float(difference.abs().max()))
            comparisons += int(valid.sum())
            requests += 1
        if not comparisons:
            raise ValueError("Query consistency requires at least one eligible query; all-masked checks are invalid")
        if maximum > 1e-5:
            raise AssertionError(f"Query set changed persistent state/risk: maximum difference {maximum}")
        return {"passed": True, "max_absolute_logit_difference": maximum, "tolerance": 1e-5,
                "compared_patient_queries": comparisons, "subset_requests": requests}
    finally:
        restore_rng(rng)
        model.train(previous)


def train_timeline(cohort_path, split_path, model_config, train_config, out_dir,
                   source_root, expected_source=None, resume=False, diagnostic=False):
    from .modality_support import fit_modality_support
    from .timeline_data import TimelineCohort, split_indices
    from .timeline_model import TimelineModel

    train_config.validate()
    model_config.validate()
    terminal_objective = getattr(model_config, "objective", "legacy_multistage") == "terminal_state_v1"
    report_enabled = getattr(model_config, "s1_report_concepts", False)
    if train_config.alignment_weight and not terminal_objective:
        raise ValueError("S1 alignment requires the terminal_state_v1 model objective")
    if train_config.report_concept_weight and not (terminal_objective and report_enabled):
        raise ValueError("Report concept loss requires the terminal model and its S1 report head")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    with (out / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = source_manifest(source_root) if expected_source is None else expected_source
        verify_source(source_root, source)
        cohort = TimelineCohort.load(cohort_path)
        report_contract = validate_report_concept_contract(cohort.metadata) if report_enabled else None
        if report_enabled:
            from .report_concept_data import validate_report_concepts
            validate_report_concepts(cohort.tensors, cohort.metadata)
        split = json.loads(Path(split_path).read_text())
        roles = split_indices(cohort, split)
        fixed_split = "test" in roles
        evaluation_role = "test" if fixed_split else "outer_evaluation"
        protocol = cohort.metadata.get("protocol", "fixed_train_validation_test") if fixed_split else "legacy_nested_521"
        if fixed_split and protocol == "fixed651_712" and {
                role: len(patients) for role, patients in split.items()} != {
                    "train": 456, "validation": 65, "test": 130}:
            raise ValueError("fixed651_712 requires exactly 456 training, 65 validation, and 130 test patients")
        contract = {"schema": "modality-event-v2", "cohort_sha256": file_sha256(cohort_path),
                    "split_sha256": file_sha256(split_path), "model": asdict(model_config),
                    "training": asdict(train_config), "source_sha256": source,
                    "diagnostic": bool(diagnostic), "precision": "FP32", "initialization": "from_scratch",
                    "source_root": str(Path(source_root).resolve()), "holdout65_accessed": fixed_split}
        if fixed_split:
            contract.update(protocol=protocol, evaluation_role=evaluation_role,
                            split_patients={role: len(patients) for role, patients in split.items()},
                            historical_holdouts_repartitioned=True, test_used_for_selection=False)
        contract = json.loads(json.dumps(contract))
        contract_id = fingerprint(contract)
        contract_path = out / "contract.json"
        if contract_path.exists():
            old = json.loads(contract_path.read_text())
            if old != {"id": contract_id, "contract": contract} or not resume:
                raise ValueError("Existing output requires --resume and identical code/data/split/config")
            if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get("status") == "pass":
                return json.loads((out / "metrics.json").read_text())
            if not (out / "last.pt").exists():
                raise ValueError("Resume requires a recovery checkpoint")
        elif any(p.name != "run.lock" for p in out.iterdir()):
            raise ValueError("Refusing to overwrite unbound existing run artifacts")
        else:
            write_json({"id": contract_id, "contract": contract}, contract_path)
        write_json({"status": "in_progress", "pid": os.getpid(), "contract_id": contract_id,
                    "started_at_unix": time.time(), "optimizer_steps": 0}, out / "status.json")
        try:
            fit_ids = cohort.encoders.get("fit_ids")
            if fit_ids is None or set(fit_ids) != set(split["train"]):
                raise ValueError("Clinical encoders must be bound to exactly these training patients")
            if model_config.image_dim != cohort.tensors["ct0"].shape[-1]:
                raise ValueError("Image dimension does not match the feature cache")
            if model_config.time_basis != cohort.metadata["time_basis"]:
                raise ValueError("Model time basis must match the cohort's verified time basis")
            random.seed(train_config.seed)
            np.random.seed(train_config.seed)
            torch.manual_seed(train_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(train_config.seed)
            torch.backends.mha.set_fastpath_enabled(False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            device = torch.device(train_config.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable")
            model = TimelineModel(model_config).to(device)
            if report_enabled:
                model.report_concept_head_trained = train_config.report_concept_weight > 0
            fitting = cohort.batch(roles["train"], device)
            model.fit_statistics(fitting)
            outcome_priors = {"initialized": False}
            if (getattr(model_config, "capacity_profile", "protocol_v1") == "compact_v1"
                    or getattr(model_config, "terminal_clinical_anchor", False)):
                outcome_priors = model.fit_outcome_priors(fitting)
            recurrence_values = fitting["binary"][fitting["binary_valid"]]
            model.training_recurrence_probability = outcome_priors.get(
                "terminal_prior_probability", float(recurrence_values.float().mean()))
            targets = FixedCTMoments(model_config.image_dim,
                                     anchor_dim=model_config.hidden if terminal_objective else None,
                                     report_concepts=report_enabled).to(device)
            targets.fit(fitting["ct1"], scan_valid(fitting))
            if report_enabled:
                targets.fit_report_concepts(fitting["s1_concepts"], report_training_mask(fitting))
                if getattr(model_config, "capacity_profile", "protocol_v1") == "compact_v1":
                    with torch.no_grad():
                        model.s1_report_output.weight.zero_()
                        model.s1_report_output.bias.zero_()
                        probability = targets.report_mean[0].clamp(1e-5, 1 - 1e-5)
                        model.s1_report_output.bias[0] = torch.logit(probability)
            support = fit_modality_support(fitting)
            del fitting
            if train_config.freeze_image_projection:
                model.image.requires_grad_(False)
                if hasattr(model, "position"):
                    model.position.requires_grad_(False)
            groups, group_manifest = optimizer_groups(model, train_config)
            optimizer = torch.optim.AdamW(groups, weight_decay=train_config.weight_decay)
            sampler = StatefulPatientSampler(roles["train"], train_config.seed + 1)
            write_json(group_manifest, out / "optimizer_groups.json")
            write_json({"model": asdict(model_config), "training": asdict(train_config)}, out / "effective_config.json")
            target_space = {"normalization": "training_CT1_channel_then_set_mean_population_SD",
                            "fit_ids_hash": fingerprint(sorted(split["train"])), "fit_patients": len(split["train"]),
                            "forbidden_overlap": False, "differentiable": True}
            if terminal_objective:
                target_space.update(anchor="fixed_random_projection_of_standardized_set_moments",
                                    anchor_dimension=targets.anchor_dim, anchor_seed=targets.ANCHOR_SEED,
                                    anchor_fit="training_CT1_projected_mean_and_population_SD",
                                    anchor_target_detached=True, anchor_trainable=False)
            if report_enabled:
                target_space["s1_report_concepts"] = {
                    "names": report_contract["names"], "transforms": report_contract["transforms"],
                    "continuous_coordinates": "training_transformed_mean_and_population_SD_clamped_0.05",
                    "binary_coordinate": "unstandardized_binary_logit",
                    "training_valid_counts": targets.report_counts.tolist(),
                    "target_detached": True, "source_contract": report_contract,
                }
            write_json(target_space, out / "target_space.json")
            export_metadata = dict(cohort.metadata)
            readout_contract = {}
            if terminal_objective:
                readout_contract = {"objective": "terminal_state_v1", "pcr_cutoff": "post_neoadjuvant_s1",
                                    "terminal_event_order": 3,
                                    "terminal_boundary": "last_treatment_summary_not_fixed_followup_horizon"}
                export_metadata.update(**readout_contract, scored_query_orders=[3],
                                       query_names=["S0_state_only", "S1_state_only", "S2_state_only",
                                                    "terminal_S3_retrospective"],
                                       intermediate_state_risk_supported=False,
                                       clinical_causal_effects_identified=False)
            if report_enabled:
                readout_contract.update(report_concept_head_trained=train_config.report_concept_weight > 0,
                                        report_concept_supervision="weak_S1_report_targets_not_clinically_adjudicated")
                export_metadata.update(**readout_contract, full_longitudinal_concepts=False,
                                       report_concept_weight=train_config.report_concept_weight)
            step, best_step, best, stale = 0, 0, float("inf"), 0
            history, probes = [], []
            best_validation = None
            initial_validation = None
            if resume:
                checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
                if checkpoint["contract_id"] != contract_id:
                    raise ValueError("Recovery checkpoint contract mismatch")
                model.load_state_dict(checkpoint["model_state"])
                targets.load_state_dict(checkpoint["target_statistics"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                sampler.load_state_dict(checkpoint["sampler"])
                step, best_step = checkpoint["optimizer_steps"], checkpoint["selected_step"]
                best, stale = checkpoint["best_validation_nll"], checkpoint["stale"]
                history, probes = checkpoint["history"], checkpoint["gradient_probes"]
                best_validation = checkpoint["validation_metrics"]
                initial_validation = checkpoint.get("initial_validation_metrics")
                restore_rng(checkpoint["rng"])
            started = time.monotonic()

            def payload(recovery):
                head_status = checkpoint_head_status(model_config, train_config, step)
                metadata = {**export_metadata, "head_training_status": head_status,
                            "report_concept_head_trained": head_status["report_concept_head_trained"],
                            "world_model_trained": head_status["world_model_trained"],
                            "training_recurrence_probability": model.training_recurrence_probability,
                            "outcome_priors": outcome_priors}
                value = {"schema": "modality-event-v2", "model_config": asdict(model_config),
                         "model_state": model.state_dict(), "target_statistics": targets.state_dict(),
                         "target_space": target_space,
                         "contract_id": contract_id, "contract": contract, "fit_ids": split["train"],
                         "encoders": cohort.encoders, "support": support, "metadata": metadata,
                         "optimizer_steps": step, "selected_step": best_step,
                         "best_validation_nll": best, "validation_metrics": best_validation,
                         "initial_validation_metrics": initial_validation, "outcome_priors": outcome_priors,
                         "head_training_status": head_status}
                value.update(readout_contract)
                if report_enabled:
                    value["report_concept_head_trained"] = head_status["report_concept_head_trained"]
                if recovery:
                    value.update(optimizer_state=optimizer.state_dict(), sampler=sampler.state_dict(),
                                 rng=capture_rng(), history=history, gradient_probes=probes, stale=stale)
                return value

            if train_config.validate_initial and not resume:
                set_head_training_status(model, checkpoint_head_status(model_config, train_config, 0))
                _, initial_validation = collect_predictions(
                    model, cohort, roles["validation"], train_config.batch_size, device, targets,
                    report_concept_head_trained=False)
                best_validation = initial_validation
                best = initial_validation["selection_nll"]
                history.append({"optimizer_steps": 0, "initial_baseline": True,
                                "validation": initial_validation, "selected": True})
                atomic_save(payload(False), out / "best.pt")
                atomic_save(payload(True), out / "last.pt")
                write_json(initial_validation, out / "initial_validation_metrics.json")
                write_json(history, out / "history.json")

            budget = min(2, train_config.max_optimizer_steps) if diagnostic else train_config.max_optimizer_steps
            stop_reason = "diagnostic_complete" if diagnostic else "maximum_optimizer_steps"
            while step < budget:
                if stale >= train_config.patience and step >= train_config.minimum_optimizer_steps:
                    stop_reason = "validation_patience" if fixed_split else "inner_validation_patience"
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                means = {}
                update_started = time.monotonic()
                for _ in range(train_config.accumulation_steps):
                    batch = cohort.batch(sampler.next(train_config.batch_size), device)
                    output = model(batch)
                    terms = timeline_loss(output, batch, targets, train_config)
                    if any(not torch.isfinite(value).all() for value in terms.values()):
                        raise FloatingPointError("Nonfinite training loss")
                    (terms["total"] / train_config.accumulation_steps).backward()
                    for key, value in terms.items():
                        means[key] = means.get(key, 0.0) + float(value.detach()) / train_config.accumulation_steps
                health = gradient_health(model)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip, error_if_nonfinite=True))
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite parameter after optimizer update")
                step += 1
                row = {"optimizer_steps": step, "training": means, "gradient_norm_before_clip": gradient_norm,
                       "seconds": time.monotonic() - update_started, "sampler_epochs": sampler.epochs}
                if step == 1 or diagnostic:
                    probes.append({"optimizer_steps": step, **health})
                    write_json(probes, out / "gradient_diagnostics.json")
                evaluate = step % train_config.validation_interval == 0 or step == budget
                if evaluate:
                    verify_source(source_root, source)
                    set_head_training_status(model, checkpoint_head_status(model_config, train_config, step))
                    _, validation = collect_predictions(model, cohort, roles["validation"], train_config.batch_size, device, targets,
                                                        report_concept_head_trained=train_config.report_concept_weight > 0)
                    row["validation"] = validation
                    if validation["selection_nll"] < best - train_config.min_delta:
                        best, best_step, stale = validation["selection_nll"], step, 0
                        best_validation = validation
                        atomic_save(payload(False), out / "best.pt")
                    else:
                        stale += 1
                    row["selected"] = best_step == step
                history.append(row)
                if evaluate or step % train_config.checkpoint_interval == 0 or diagnostic:
                    atomic_save(payload(True), out / "last.pt")
                    write_json(history, out / "history.json")
                write_json({"status": "in_progress", "pid": os.getpid(), "contract_id": contract_id,
                            "optimizer_steps": step, "selected_step": best_step, "elapsed_seconds": time.monotonic() - started,
                            "last_training_loss": means["total"]}, out / "status.json")
                print(json.dumps({"optimizer_steps": step, "loss": means["total"],
                                  "seconds": row["seconds"], "selected_step": best_step}), flush=True)
            selected = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(selected["model_state"])
            targets.load_state_dict(selected["target_statistics"])
            selected_status = checkpoint_head_status(model_config, train_config, selected["optimizer_steps"])
            set_head_training_status(model, selected_status)
            consistency = query_consistency(model, cohort.batch(roles["train"][:2], device))
            validation_rows, validation = collect_predictions(model, cohort, roles["validation"], train_config.batch_size, device, targets,
                                                              report_concept_head_trained=selected_status["report_concept_head_trained"])
            write_json(validation_rows, out / "validation_predictions.json")
            metrics = {"status": "pass", "diagnostic": diagnostic, "optimizer_steps": step,
                       "selected_step": selected["selected_step"], "validation": validation,
                       "selected_optimizer_steps": selected["optimizer_steps"],
                       "initial_validation": initial_validation, "head_training_status": selected_status,
                       "selection_rule": ("minimum_validation_patient_mean_query_BCE" if fixed_split
                                          else "minimum_inner_validation_patient_mean_query_BCE"),
                       "query_consistency": consistency, "stop_reason": stop_reason,
                       "elapsed_seconds": time.monotonic() - started, "parameter_count": sum(p.numel() for p in model.parameters()),
                       "original_holdout65_accessed": fixed_split, "source_mode": cohort.metadata.get("source_mode"),
                       "blocked_arms": {"C_calendar_drift": "No verified calendar event dates",
                                        "D_CT1_assimilation": "Calendar arm unavailable; CT1 remains forecast supervision only"}}
            if terminal_objective:
                metrics.update(**readout_contract, selection_rule="minimum_validation_terminal_BCE",
                               prediction_semantics="retrospective_terminal_recorded_status",
                               blocked_stages={"B_measured_concepts": "No verified longitudinal concept labels in the locked cohort",
                                               "C_causal_strategies": "Insufficient strategy overlap, verified chronology, followup, and confounder information"})
                del metrics["blocked_arms"]
                if report_enabled:
                    metrics["report_concept_head_trained"] = selected_status["report_concept_head_trained"]
                    metrics["blocked_stages"]["B_measured_concepts"] = (
                        "Weak S1 report targets only; clinically adjudicated and full longitudinal concept evidence remains unavailable")
            if fixed_split:
                metrics.update(protocol=protocol, split_patients=contract["split_patients"],
                               historical_holdouts_repartitioned=True, test_used_for_selection=False)
            if train_config.collect_full_train_metrics:
                _, training_metrics = collect_predictions(model, cohort, roles["train"], train_config.batch_size, device, targets,
                                                          report_concept_head_trained=selected_status["report_concept_head_trained"])
                metrics["train"] = training_metrics
                write_json(training_metrics, out / "train_metrics.json")
            if not diagnostic and train_config.evaluate_test:
                evaluation_prefix = "test" if fixed_split else "outer"
                predictions_path = out / f"{evaluation_prefix}_predictions.json"
                evaluation_path = out / f"{evaluation_prefix}_metrics.json"
                if evaluation_path.exists():
                    evaluation_payload = json.loads(evaluation_path.read_text())
                    if (evaluation_payload["selected_step"] != selected["selected_step"]
                            or evaluation_payload["contract_id"] != contract_id):
                        raise ValueError(f"{evaluation_role} evaluation is already bound to a different selected checkpoint")
                    evaluation_metrics = evaluation_payload["metrics"]
                else:
                    evaluation_rows, evaluation_metrics = collect_predictions(
                        model, cohort, roles[evaluation_role], train_config.batch_size, device, targets,
                        report_concept_head_trained=selected_status["report_concept_head_trained"])
                    write_json(evaluation_rows, predictions_path)
                    write_json({"selected_step": selected["selected_step"], "contract_id": contract_id,
                                "metrics": evaluation_metrics, "used_for_selection": False}, evaluation_path)
                metrics[evaluation_role] = evaluation_metrics
            elif not train_config.evaluate_test:
                metrics[evaluation_role] = None
                metrics["test_evaluation_skipped"] = True
                metrics["test_skip_reason"] = "validation_only_selection_protocol"
            selected["validation_metrics"] = validation
            selected["total_optimizer_steps"] = step
            selected["head_training_status"] = selected_status
            selected["metadata"]["head_training_status"] = selected_status
            selected["metadata"]["world_model_trained"] = selected_status["world_model_trained"]
            selected["metadata"]["report_concept_head_trained"] = selected_status["report_concept_head_trained"]
            atomic_save(selected, out / "inference.pt")
            verify_source(source_root, source)
            write_json(metrics, out / "metrics.json")
            write_json({"status": "pass", "pid": os.getpid(), "contract_id": contract_id,
                        "optimizer_steps": step, "selected_step": selected["selected_step"],
                        "diagnostic": diagnostic, "completed_at_unix": time.time()}, out / "status.json")
            return metrics
        except BaseException as error:
            write_json({"status": "fail", "pid": os.getpid(), "contract_id": contract_id,
                        "error_type": type(error).__name__, "error": str(error), "failed_at_unix": time.time()}, out / "status.json")
            raise
