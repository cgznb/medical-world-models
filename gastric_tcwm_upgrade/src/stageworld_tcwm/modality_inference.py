"""Strict modality-event-v2 inference and auditable query semantics."""
from __future__ import annotations

from pathlib import Path

import torch

from .modality_support import modality_support_flags
from .timeline_model import EVENT_FIELDS, FORBIDDEN_FIELDS, TERMINAL_EVENT_ORDER


FORBIDDEN_INPUTS = frozenset({
    "treatment", "drugs", "regimens", "named_mentions", "drug_embedding",
    "text_embedding", "unseen_treatment_names_count", "dose", "cycles", "AD",
})


class ModalityPredictor:
    def __init__(self, model, support, metadata, target_statistics=None):
        self.model = model.eval()
        self.support = support
        self.metadata = dict(metadata)
        self.report_statistics = None
        if model.cfg.s1_report_concepts:
            names = ("report_mean", "report_scale", "report_counts", "report_fitted")
            if not isinstance(target_statistics, dict) or any(name not in target_statistics for name in names):
                raise ValueError("S1 report prediction requires fitted training-only target statistics")
            statistics = {name: target_statistics[name].detach().cpu().clone()
                          for name in names if isinstance(target_statistics[name], torch.Tensor)}
            if (set(statistics) != set(names)
                    or any(statistics[name].shape != (4,) for name in names[:3])
                    or statistics["report_fitted"].shape != torch.Size([])
                    or statistics["report_fitted"].dtype != torch.bool
                    or not bool(statistics["report_fitted"])
                    or not torch.isfinite(statistics["report_mean"]).all()
                    or not torch.isfinite(statistics["report_scale"]).all()
                    or not (statistics["report_scale"] > 0).all()
                    or statistics["report_counts"].dtype != torch.long
                    or (statistics["report_counts"] < 0).any()):
                raise ValueError("Invalid S1 report target statistics")
            self.report_statistics = statistics

    @classmethod
    def load(cls, path: str | Path, device="cpu"):
        from .timeline_config import TimelineConfig
        from .timeline_model import TimelineModel

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "modality-event-v2":
            raise ValueError("Require modality-event-v2; legacy weights need a separate explicit conversion")
        encoders = payload.get("encoders", {})
        if any(key in encoders for key in ("treatment_support", "tab_mean", "tab_scale")):
            raise ValueError("Legacy named support and 361-dimensional statistics are incompatible")
        fit_ids = payload.get("fit_ids", [])
        if not fit_ids or len(fit_ids) != len(set(fit_ids)):
            raise ValueError("Inference export must record unique training fit IDs")
        if set(encoders.get("fit_ids", [])) != set(fit_ids):
            raise ValueError("Clinical encoders and model training patients differ")
        config = TimelineConfig(**payload["model_config"])
        model = TimelineModel(config)
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device)
        metadata = dict(payload["metadata"])
        if "head_training_status" in payload:
            status = payload["head_training_status"]
            step = payload.get("optimizer_steps")
            if (not isinstance(status, dict) or type(step) is not int or step < 0
                    or status.get("optimizer_steps") != step
                    or status != metadata.get("head_training_status")):
                raise ValueError("Selected checkpoint step and prediction-head status disagree")
            trained_flags = ("world_model_trained", "terminal_head_trained", "pcr_head_trained",
                             "forecast_head_trained", "s1_alignment_trained", "report_concept_head_trained")
            if (any(type(status.get(name)) is not bool for name in trained_flags)
                    or (step == 0 and any(status[name] for name in trained_flags))
                    or metadata.get("report_concept_head_trained") is not status["report_concept_head_trained"]
                    or ("report_concept_head_trained" in payload
                        and payload["report_concept_head_trained"] is not status["report_concept_head_trained"])):
                raise ValueError("Prediction-head training flags disagree with the selected checkpoint")
            training = payload.get("contract", {}).get("training")
            if training is not None:
                from .timeline_training import TimelineTrainConfig, checkpoint_head_status

                expected = checkpoint_head_status(config, TimelineTrainConfig(**training).validate(), step)
                if status != expected:
                    raise ValueError("Prediction-head training flags disagree with the training contract")
            metadata["head_training_status"] = dict(status)
        return cls(model, payload["support"], metadata, payload.get("target_statistics"))

    def _prepare_batch(self, batch):
        forbidden = set(batch) & (FORBIDDEN_INPUTS | FORBIDDEN_FIELDS)
        if forbidden:
            raise ValueError("Drug, regimen, dose and legacy inputs are not accepted")
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        present_radio = (batch["modality_known"][..., 1].bool()
                         | batch["modality_value"][..., 1].bool()
                         | batch["modality_applicable"][..., 1].bool())
        if bool((present_radio & batch["event_mask"].bool()).any()):
            raise ValueError("Radiotherapy has no verified source or trained scenario support")
        return batch

    def _report_predictions(self, output):
        logits = output["s1_concept_logits"]
        stats = {name: value.to(logits.device) for name, value in self.report_statistics.items()}
        transformed = logits * stats["report_scale"] + stats["report_mean"]
        transformed[:, 0] = logits[:, 0].sigmoid()
        decoded = transformed.clone()
        decoded[:, 1] *= 100
        decoded[:, 2:] = torch.expm1(transformed[:, 2:])
        outside = ~torch.isfinite(decoded) | ~torch.isfinite(logits)
        outside[:, 1] |= (transformed[:, 1] < 0) | (transformed[:, 1] > 1)
        outside[:, 2:] |= transformed[:, 2:] < 0
        eligible = output["s1_concept_mask"][:, None].expand_as(logits).clone()
        eligible[:, :2] &= output["pcr_mask"][:, None]
        eligible &= stats["report_counts"] > 0
        trained = self.metadata.get("report_concept_head_trained") is True
        if not trained:
            eligible.zero_()
        valid = eligible & ~outside
        values = decoded.masked_fill(~valid, float("nan"))
        coordinates = transformed.masked_fill(~eligible, float("nan"))
        names = output["s1_concept_names"]
        return {
            "s1_report_predictions": {name: values[:, index] for index, name in enumerate(names)},
            "s1_report_transformed_predictions": {name: coordinates[:, index] for index, name in enumerate(names)},
            "s1_report_valid": {name: valid[:, index] for index, name in enumerate(names)},
            "s1_report_out_of_domain": {name: (outside & eligible)[:, index] for index, name in enumerate(names)},
        }, {
            "target_names": list(names), "target_transforms": list(output["s1_concept_target_transforms"]),
            "output_units": ["probability", "percent", "specimen_node_count_scale", "cm"],
            "target_stage": 1, "available_stage": 2,
            "semantics": "future_resection_specimen_pathology_corresponding_to_S1_disease",
            "architecture": "auxiliary_latent_state_readout_not_concept_bottleneck",
            "inverse_transform_interpretation": "point_prediction_not_original_scale_conditional_mean",
            "training_target_mean": stats["report_mean"].tolist(),
            "training_target_scale": stats["report_scale"].tolist(),
            "training_target_counts": stats["report_counts"].tolist(),
            "head_trained": trained, "clinically_adjudicated": False,
            "S2_patient_residual_burden": False, "longitudinal_semantics_validated": False,
        }

    @torch.inference_mode()
    def predict(self, batch):
        batch = self._prepare_batch(batch)
        output = self.model(batch)
        flags = modality_support_flags(batch, self.support)
        probabilities = output["logits"].sigmoid()
        valid = output["query_mask"].bool()
        probabilities = probabilities.masked_fill(~valid, float("nan"))
        event_mask = batch["event_mask"].bool()
        hypothetical = ((batch["role"] == 3) & event_mask).any(1)
        retrospective = ((batch["role"] == 2) & event_mask).any(1)
        result = {
            "risk": probabilities,
            "query_mask": valid,
            "query_order": batch["query_order"],
            "support": flags,
            "metadata": {
                "schema": "modality-event-v2",
                "endpoint": "recorded_recurrence_binary_status",
                "time_basis": self.model.cfg.time_basis,
                "source_mode": self.metadata.get("source_mode", "unspecified"),
                "prospective_supported": False,
                "clinical_validation": False,
                "causal_effects_identified": False,
                "calendar_interpolation_supported": self.model.cfg.time_basis == "calendar_days",
                "radiotherapy_supported": False,
                "contains_hypothetical_history": hypothetical.tolist(),
                "contains_retrospective_history": retrospective.tolist(),
                "query_names": self.metadata.get("query_names", []),
                "objective": self.model.cfg.objective,
                "readout": ("factual_complete_terminal_only" if self.model.cfg.objective == "terminal_state_v1"
                            else "legacy_multistage"),
            },
        }
        status = self.metadata.get("head_training_status")
        if status is not None:
            result["metadata"]["head_training_status"] = dict(status)
            result["metadata"]["world_model_trained"] = bool(status["world_model_trained"])
            if not status["world_model_trained"]:
                result["metadata"]["readout"] = "initial_checkpoint_factual_terminal_only"
                result["metadata"]["strategy_rollout_available"] = False
            if "pcr_logits" in output:
                pcr_valid = output.get("pcr_mask", torch.ones(len(probabilities), device=probabilities.device, dtype=torch.bool)).clone()
                pcr_valid &= bool(status["pcr_head_trained"])
                result["pcr_probability"] = output["pcr_logits"].sigmoid().masked_fill(~pcr_valid, float("nan"))
                result["pcr_valid"] = pcr_valid
            if "forecast" in output:
                forecast_valid = output["forecast_mask"].clone() & bool(status["forecast_head_trained"])
                result["forecast"] = output["forecast"].masked_fill(~forecast_valid[:, None, None], float("nan"))
                result["forecast_valid"] = forecast_valid
        if self.model.cfg.s1_report_concepts:
            predictions, metadata = self._report_predictions(output)
            result.update(predictions)
            result["metadata"]["s1_report_concepts"] = metadata
        return result

    @torch.inference_mode()
    def predict_strategy(self, history_batch, future_events, *, strategy_name, final_event_order=TERMINAL_EVENT_ORDER):
        """Score an explicitly supplied continuation, never an implicit observed suffix."""
        from .modality_support import phase_action_support_flags

        if self.model.cfg.objective != "terminal_state_v1":
            raise ValueError("Explicit terminal strategy rollout requires terminal_state_v1")
        status = self.metadata.get("head_training_status")
        if status is not None and not status.get("world_model_trained", False):
            raise ValueError("Strategy rollout is unavailable for a step-0 checkpoint with an untrained world model")
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("An explicit strategy_name is required")
        if isinstance(final_event_order, bool) or final_event_order != TERMINAL_EVENT_ORDER:
            raise ValueError("This model's terminal boundary is the third modeled event")
        if not isinstance(future_events, dict) or set(future_events) != set(EVENT_FIELDS):
            raise ValueError("future_events must explicitly contain only all event fields")
        history = self._prepare_batch(history_batch)
        future = self._prepare_batch(future_events)
        prefix = history["event_mask"].shape[1]
        if prefix not in (0, 1, 2):
            raise ValueError("Supply a physically truncated S0/S1/S2 history, without observed future events")
        if future["event_mask"].shape != (len(history["ct0"]), TERMINAL_EVENT_ORDER - prefix):
            raise ValueError("Supply every future event through the fixed terminal boundary")
        if not history["event_mask"].all() or not future["event_mask"].all():
            raise ValueError("Explicit paths require every stage record; confirmed absence belongs in modality status")
        if not ((history["role"] == 0) | (history["role"] == 2)).all():
            raise ValueError("The starting history must contain factual delivered or retrospective events")
        if not ((future["role"] == 0) | (future["role"] == 2) | (future["role"] == 3)).all():
            raise ValueError("Future events must specify actions, not unresolved plan records")
        future = dict(future, role=torch.full_like(future["role"], 3))
        path = {name: torch.cat((history[name], future[name]), 1) for name in EVENT_FIELDS}
        self.model.validate_terminal_events(path)
        state = self.model.initialize(history)
        for index in range(TERMINAL_EVENT_ORDER):
            state = self.model.apply_event(state, {name: path[name][:, index] for name in EVENT_FIELDS})
        support = phase_action_support_flags(path, self.support)
        unsupported = support["unsupported_events"][:, prefix:].any(1)
        sparse = support["sparse_events"][:, prefix:].any(1)
        valid = ~(unsupported | sparse)
        if not support["support_audited"]:
            valid = torch.zeros_like(valid)
        probability = self.model.outcome(state).sigmoid().masked_fill(~valid, float("nan"))
        return {
            "risk": probability[:, None], "query_mask": valid[:, None],
            "query_order": torch.full((len(probability), 1), TERMINAL_EVENT_ORDER,
                                      device=probability.device, dtype=torch.long),
            "support": {**modality_support_flags(path, self.support), **support},
            "metadata": {
                "schema": "modality-event-v2", "objective": self.model.cfg.objective,
                "endpoint": "recorded_recurrence_binary_status", "time_basis": "ordinal_stage",
                "inference_kind": "explicit_strategy_terminal_rollout", "strategy_name": strategy_name,
                "start_event_order": prefix, "terminal_event_order": TERMINAL_EVENT_ORDER,
                "future_event_source": "explicit_argument", "ct1_assimilated": False,
                "source_mode": self.metadata.get("source_mode", "unspecified"),
                "contains_hypothetical_history": state.hypothetical.tolist(),
                "support_interpretation": "training_phase_action_counts_not_conditional_causal_overlap",
                "unsupported_future_action": unsupported.tolist(), "sparse_future_action": sparse.tolist(),
                "clinical_recommendation": False, "clinical_validation": False,
                "causal_effects_identified": False, "prospective_supported": False,
                "calendar_interpolation_supported": False,
            },
        }
