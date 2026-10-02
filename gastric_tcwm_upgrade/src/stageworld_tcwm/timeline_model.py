"""A single patient state and shared risk head over legal modality histories."""
from dataclasses import fields
import torch
from torch import nn
from .backbone import CrossBlock, SetReadout, SpatialTransition
from .clinical import ClinicalAnchor
from .continuous_drift import ContinuousDrift
from .event_encoder import EventEncoder
from .event_jump import EventJump
from .patient_state import PatientState
from .timeline_config import TimelineConfig


EVENT_FIELDS = ("modality_value", "modality_known", "modality_applicable", "event_mask",
                "phase", "operation", "role", "event_order", "time_features",
                "occurred_at", "available_at", "event_id")
FORBIDDEN_FIELDS = frozenset(("treatment", "drugs", "regimens", "named_mentions", "drug_embedding",
                              "text_embedding", "unseen_treatment_names_count", "drug_support",
                              "treatment_names", "dose", "dosage", "cycles", "tab_mean", "tab_scale"))
TERMINAL_EVENT_ORDER = 3


class ClinicalFields(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.value = nn.Linear(1, h)
        self.field = nn.Parameter(torch.randn(1, 32, h) * .02)
        self.missing = nn.Embedding(2, h)
        depth = 1 if cfg.capacity_profile == "compact_v1" else 2
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            h, 4, 4*h, cfg.dropout, "gelu", batch_first=True, norm_first=True) for _ in range(depth)])
        self.pool = SetReadout(h, 4, depth, cfg.dropout)

    def forward(self, values, known):
        tokens = self.value(values[..., None]) + self.field + self.missing((~known).long())
        for block in self.blocks:
            tokens = block(tokens)
        return self.pool(tokens)


class SharedRiskHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.pool = SetReadout(h, 4, cfg.readout_blocks, cfg.dropout)
        self.output = nn.Sequential(nn.LayerNorm(4*h), nn.Linear(4*h, h), nn.GELU(),
                                    nn.Dropout(cfg.dropout), nn.Linear(h, 64), nn.GELU(), nn.Linear(64, 1))
        self.terminal_clinical_anchor = cfg.terminal_clinical_anchor
        self.terminal_residual_scale = cfg.terminal_residual_scale
        if self.terminal_clinical_anchor:
            self.clinical_anchor = ClinicalAnchor()
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

    def clinical_anchor_logits(self, clinical):
        if not self.terminal_clinical_anchor:
            raise ValueError("This terminal head has no fitted clinical anchor")
        safe = torch.where(torch.isfinite(clinical), clinical, self.clinical_anchor.mean)
        return self.clinical_anchor(safe)

    def forward(self, state):
        pooled = self.pool(torch.cat((state.z, state.memory, state.clinical), 1)).mean(1)
        value = self.output(torch.cat((pooled, state.z.mean(1), state.memory.mean(1),
                                      state.clinical.mean(1)), -1)).squeeze(-1).float()
        if self.terminal_clinical_anchor:
            if state.raw_clinical is None:
                raise ValueError("The terminal clinical anchor requires retained baseline clinical values")
            value = self.clinical_anchor_logits(state.raw_clinical) + self.terminal_residual_scale * value
        return value


class TimelineModel(nn.Module):
    def __init__(self, cfg: TimelineConfig):
        super().__init__()
        self.cfg = cfg.validate()
        h = cfg.hidden
        self.image = nn.Sequential(nn.Linear(cfg.image_dim, h), nn.LayerNorm(h))
        coord = torch.stack(torch.meshgrid(*([torch.linspace(-1, 1, 3)]*3), indexing="ij")).flatten(1).T[None]
        self.register_buffer("coordinates", coord)
        self.position = nn.Linear(3, h, bias=False)
        self.missing_ct = nn.Parameter(torch.randn(1, 27, h) * .02)
        self.clinical_encoder = ClinicalFields(cfg)
        self.init_blocks = nn.ModuleList([SpatialTransition(h, 4, cfg.dropout) for _ in range(cfg.init_blocks)])
        pool_depth = 1 if cfg.capacity_profile == "compact_v1" else 2
        self.initial_memory = SetReadout(h, 4, pool_depth, cfg.dropout)
        self.event_encoder = EventEncoder(cfg)
        self.event_jump = EventJump(cfg)
        self.drift = ContinuousDrift(cfg)
        self.observation_blocks = nn.ModuleList([CrossBlock(h, cfg.dropout) for _ in range(cfg.observation_blocks)])
        self.outcome = SharedRiskHead(cfg)
        self.decoder = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2*h), nn.GELU(), nn.Linear(2*h, cfg.image_dim))
        self.pcr_pool = SetReadout(h, 4, pool_depth, cfg.dropout)
        self.pcr_output = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.register_buffer("image_mean", torch.zeros(cfg.image_dim))
        self.register_buffer("image_scale", torch.ones(cfg.image_dim))
        self.register_buffer("clinical_mean", torch.zeros(32))
        self.register_buffer("clinical_scale", torch.ones(32))
        self.register_buffer("statistics_fitted", torch.tensor(False))
        self.register_buffer("schema_enabled", torch.tensor(cfg.schema_enabled, dtype=torch.bool))
        if cfg.objective == "terminal_state_v1" and (cfg.capacity_profile == "compact_v1" or cfg.terminal_clinical_anchor):
            self.register_buffer("outcome_priors_fitted", torch.tensor(False))
        if cfg.s1_report_concepts:
            from .report_concept_data import S1_CONCEPT_NAMES, S1_CONCEPT_TRANSFORMS
            if len(S1_CONCEPT_NAMES) != 4 or len(S1_CONCEPT_TRANSFORMS) != 4:
                raise ValueError("The S1 report pilot requires exactly four fixed target coordinates")
            self.s1_concept_names = tuple(S1_CONCEPT_NAMES)
            self.s1_concept_transforms = tuple(S1_CONCEPT_TRANSFORMS)
            # Append optional parameters after all A modules to preserve their seeded initialization.
            self.s1_report_pool = SetReadout(h, 4, cfg.readout_blocks, cfg.dropout)
            self.s1_report_output = nn.Linear(h, 4)

    @torch.no_grad()
    def fit_statistics(self, training_batch):
        clinical = training_batch["clinical"].float()
        known = torch.isfinite(clinical)
        safe = torch.where(known, clinical, torch.zeros_like(clinical))
        mean = safe.sum(0) / known.sum(0).clamp_min(1)
        var = torch.where(known, (safe-mean).square(), torch.zeros_like(safe)).sum(0) / known.sum(0).clamp_min(1)
        scale = var.sqrt().clamp_min(.05)
        if self.cfg.clinical_normalization == "continuous_only":
            binary = ((clinical == 0) | (clinical == 1) | ~known).all(0)
            low = torch.where(known, clinical, float("inf")).amin(0)
            high = torch.where(known, clinical, float("-inf")).amax(0)
            continuous = ~binary & (low < high)
            mean = torch.where(continuous, mean, torch.zeros_like(mean))
            scale = torch.where(continuous, scale, torch.ones_like(scale))
        self.clinical_mean.copy_(mean)
        self.clinical_scale.copy_(scale)
        images = training_batch["ct0"].float()
        valid = training_batch.get("image_valid")
        if valid is not None:
            images = images[valid[:, 0].bool()]
        if len(images):
            if not torch.isfinite(images).all():
                raise ValueError("Available CT0 features must be finite")
            self.image_mean.copy_(images.mean((0, 1)))
            self.image_scale.copy_(images.std((0, 1), unbiased=False).clamp_min(.05))
        self.statistics_fitted.fill_(True)

    @staticmethod
    def _factual_s1_pcr_mask(batch, unique, enabled):
        mask = torch.zeros(len(unique), device=unique.device, dtype=torch.bool)
        if unique.shape[1]:
            present = (batch["modality_value"][:, 0].bool() & batch["modality_known"][:, 0].bool()
                       & batch["modality_applicable"][:, 0].bool() & enabled)
            present = present.clone()
            present[:, 6] = False
            mask = unique[:, 0] & present.any(1) & ((batch["role"][:, 0] == 0) | (batch["role"][:, 0] == 2))
        return mask

    @torch.no_grad()
    def fit_outcome_priors(self, training_batch):
        """Fit terminal prognosis and post-NAC pCR priors using training labels only."""
        if not hasattr(self, "outcome_priors_fitted"):
            return {"initialized": False}
        if not bool(self.statistics_fitted):
            raise ValueError("Fit training input statistics before outcome priors")
        if bool(self.outcome_priors_fitted):
            raise ValueError("Outcome priors are already fitted; do not refit a trained model")
        self.validate_terminal_events(training_batch)
        unique = self._unique_mask(training_batch)
        complete = unique[:, :3].all(1) if unique.shape[1] >= 3 else torch.zeros(
            len(unique), dtype=torch.bool, device=unique.device)
        factual = ~(((training_batch["role"] == 1) | (training_batch["role"] == 3)) & unique).any(1)
        valid = training_batch["binary_valid"].bool() & complete & factual
        if not valid.any():
            raise ValueError("Terminal outcome priors require observed factual training outcomes")

        def prior(labels, mask):
            observed = labels[mask].float()
            if not torch.isfinite(observed).all() or not ((observed == 0) | (observed == 1)).all():
                raise ValueError("Outcome priors require finite observed binary labels")
            probability = observed.mean().clamp(1e-4, 1-1e-4) if len(observed) else observed.new_tensor(.5)
            return probability, torch.logit(probability)

        terminal_probability, terminal_logit = prior(training_batch["binary"], valid)
        pcr_valid = self._factual_s1_pcr_mask(training_batch, unique, self.schema_enabled)
        pcr_valid &= training_batch["pcr_valid"].bool()
        pcr_probability, pcr_logit = prior(training_batch["pcr"], pcr_valid)
        self.outcome.output[-1].weight.zero_()
        self.outcome.output[-1].bias.zero_()
        if self.cfg.terminal_clinical_anchor:
            clinical = training_batch["clinical"].float()
            known = torch.isfinite(clinical) & valid[:, None]
            safe = torch.where(known, clinical, torch.zeros_like(clinical))
            impute = safe.sum(0) / known.sum(0).clamp_min(1)
            self.outcome.clinical_anchor.fit(torch.where(torch.isfinite(clinical), clinical, impute),
                                             training_batch["binary"], valid)
        else:
            self.outcome.output[-1].bias.copy_(terminal_logit.reshape(1))
        self.pcr_output[-1].weight.zero_()
        self.pcr_output[-1].bias.copy_(pcr_logit.reshape(1))
        self.outcome_priors_fitted.fill_(True)
        return {"initialized": True, "terminal_prior_patients": int(valid.sum()),
                "terminal_prior_probability": float(terminal_probability),
                "terminal_clinical_anchor_fitted": self.cfg.terminal_clinical_anchor,
                "pcr_prior_patients": int(pcr_valid.sum()),
                "pcr_prior_probability": float(pcr_probability) if pcr_valid.any() else None,
                "pcr_cutoff": "post_neoadjuvant_s1"}

    def encode_image(self, values):
        return self.image((values.float()-self.image_mean)/self.image_scale) + self.position(self.coordinates)

    def initialize(self, batch):
        if not bool(self.statistics_fitted):
            raise ValueError("Fit baseline statistics on the training fold before initializing a patient")
        ct0, clinical = batch["ct0"], batch["clinical"].float()
        if ct0.ndim != 3 or ct0.shape[1:] != (27, self.cfg.image_dim) or clinical.shape != (len(ct0), 32):
            raise ValueError("Initialization requires CT0[B,27,image_dim] and legal baseline clinical[B,32]")
        known = torch.isfinite(clinical)
        normalized = torch.where(known, (clinical-self.clinical_mean)/self.clinical_scale, torch.zeros_like(clinical))
        condition = self.clinical_encoder(normalized, known)
        valid = batch.get("image_valid", torch.ones((len(ct0), 2), device=ct0.device, dtype=torch.bool))[:, 0].bool()
        image = self.encode_image(torch.where(valid[:, None, None], ct0, torch.zeros_like(ct0)))
        image = torch.where(valid[:, None, None], image, self.missing_ct)
        tokens = torch.cat((condition, image), 1)
        for block in self.init_blocks:
            tokens = block(tokens)
        z = tokens[:, 4:]
        memory = self.initial_memory(torch.cat((z, condition), 1))
        b, device = len(ct0), ct0.device
        if self.cfg.time_basis == "calendar_days":
            if "baseline_time" not in batch:
                raise ValueError("Calendar initialization requires verified baseline_time")
            time = torch.as_tensor(batch["baseline_time"], device=device, dtype=torch.float32).expand(b).clone()
            if not torch.isfinite(time).all():
                raise ValueError("Calendar baseline_time cannot be unknown")
        else:
            time = torch.full((b,), float("nan"), device=device)
        return PatientState(z=z, memory=memory, clinical=condition,
                            phase=torch.zeros(b, device=device, dtype=torch.long),
                            active_value=torch.zeros((b, 7), device=device, dtype=torch.bool),
                            active_known=torch.zeros((b, 7), device=device, dtype=torch.bool),
                            planned_value=torch.zeros((b, 7), device=device, dtype=torch.bool),
                            planned_known=torch.zeros((b, 7), device=device, dtype=torch.bool),
                            time=time, baseline_time=time.clone(), history=z.new_zeros((b, 0, self.cfg.hidden)),
                            history_mask=torch.zeros((b, 0), device=device, dtype=torch.bool),
                            event_ids=torch.empty((b, 0), device=device, dtype=torch.long),
                            event_times=torch.empty((b, 0), device=device, dtype=torch.float32),
                            retrospective=torch.zeros(b, device=device, dtype=torch.bool),
                            hypothetical=torch.zeros(b, device=device, dtype=torch.bool), time_basis=self.cfg.time_basis,
                            raw_clinical=clinical.clone() if self.cfg.terminal_clinical_anchor else None)

    def advance(self, state, target_time):
        if state.time_basis == "ordinal_stage":
            target = torch.as_tensor(target_time, device=state.z.device)
            if torch.isfinite(target).all() and (target == 0).all():
                return state.snapshot()
            raise ValueError("Ordinal models have no day-scaled advance; only advance(0) is an identity")
        return self.drift(state, target_time, self.event_encoder.modality_embedding)

    @staticmethod
    def _unique_mask(batch):
        mask = batch["event_mask"].bool().clone()
        ids = batch["event_id"]
        for index in range(mask.shape[1]):
            if index:
                duplicate = ((ids[:, :index] == ids[:, index, None]) & mask[:, :index]).any(1)
                mask[:, index] &= ~duplicate
        return mask

    def _event_present(self, state, event):
        present = event["event_mask"].bool()
        if state.event_ids.shape[1]:
            duplicate = ((state.event_ids == event["event_id"][:, None]) & state.history_mask).any(1)
            present = present & ~duplicate
        return present

    def _mechanism_batch(self, batch):
        if self.cfg.objective != "terminal_state_v1":
            return batch
        # Provenance is retained in PatientState, but cannot select an untrained dynamics branch.
        roles = batch["role"]
        return dict(batch, role=torch.where((roles == 2) | (roles == 3), torch.zeros_like(roles), roles))

    @staticmethod
    def validate_terminal_events(batch):
        mask = batch["event_mask"].bool()
        if mask.ndim != 2:
            raise ValueError("Terminal events require a [B,L] event mask")
        if mask.shape[1] > TERMINAL_EVENT_ORDER and mask[:, TERMINAL_EVENT_ORDER:].any():
            raise ValueError("terminal_state_v1 ends at the third modeled event")
        for index in range(min(mask.shape[1], TERMINAL_EVENT_ORDER)):
            active = mask[:, index]
            if index and (active & ~mask[:, :index].all(1)).any():
                raise ValueError("Terminal event histories must contain a complete ordered prefix")
            operation = (3, 0, 3)[index]
            if ((active & (batch["event_order"][:, index] != index + 1)).any()
                    or (active & (batch["phase"][:, index] != index + 1)).any()
                    or (active & (batch["operation"][:, index] != operation)).any()):
                raise ValueError("Terminal stage semantics require neoadjuvant summary, surgery, postoperative summary")
            allowed = torch.zeros(7, device=mask.device, dtype=torch.bool)
            allowed[list(((0, 2, 3, 4, 5), (6,), (0,))[index])] = True
            if (batch["modality_applicable"][:, index].bool() & ~allowed & active[:, None]).any():
                raise ValueError("Modality applicability is outside the terminal stage scope")
        roles = batch["role"][mask]
        if ((roles < 0) | (roles > 3)).any():
            raise ValueError("Unknown terminal event provenance")
        if not torch.equal(TimelineModel._unique_mask(batch), mask):
            raise ValueError("Terminal stage records require distinct event IDs")

    def _apply_encoded(self, state, event, local, history_vector, present, *, base=None, prefix_tokens=None, prefix_padding=None):
        plan = (event["role"] == 1) | (event["operation"] == 4)
        observation = event["operation"] == 5
        actual = present & ~plan & ~observation
        # A procedure needs a positively documented modality; absent/unknown surgery is not an operation.
        known_present = event["modality_known"].bool() & event["modality_value"].bool()
        known_present &= event["modality_applicable"].bool() & self.schema_enabled
        actual &= (event["operation"] != 0) | known_present.any(1)
        source = state if base is None else base
        if prefix_tokens is None:
            event_tokens = torch.cat((local, history_vector[:, None]), 1)
            condition = torch.cat((source.clinical, source.memory, event_tokens), 1)
            padding = None
        else:
            event_tokens = prefix_tokens
            condition = torch.cat((source.clinical, source.memory, prefix_tokens), 1)
            padding = torch.cat((torch.zeros((len(state.z), 8), device=state.z.device, dtype=torch.bool), prefix_padding), 1)
        candidate = self.event_jump(source, condition, history_vector, padding)
        z = torch.where(actual[:, None, None], candidate, state.z)
        candidate_memory = self.event_jump.update_memory(source.memory, z, event_tokens, prefix_padding)
        memory = torch.where(present[:, None, None], candidate_memory, state.memory)
        active_value, active_known = state.active_value.clone(), state.active_known.clone()
        persistent = known_present.clone()
        persistent[:, 6] = False
        start = actual & (event["operation"] == 1)
        stop = actual & (event["operation"] == 2)
        active_known = active_known | (persistent & (start | stop)[:, None])
        active_value = torch.where(persistent & start[:, None], torch.ones_like(active_value), active_value)
        active_value = torch.where(persistent & stop[:, None], torch.zeros_like(active_value), active_value)
        planned_scope = present[:, None] & plan[:, None] & event["modality_applicable"].bool() & self.schema_enabled
        planned_known = torch.where(planned_scope, event["modality_known"].bool(), state.planned_known)
        planned_value = torch.where(planned_scope, event["modality_value"].bool(), state.planned_value)
        return state.updated(z=z, memory=memory, phase=torch.where(actual, event["phase"], state.phase),
                             active_value=active_value, active_known=active_known,
                             planned_value=planned_value, planned_known=planned_known,
                             history=torch.cat((state.history, local.mean(1)[:, None]), 1),
                             history_mask=torch.cat((state.history_mask, present[:, None]), 1),
                             event_ids=torch.cat((state.event_ids, event["event_id"][:, None]), 1),
                             event_times=torch.cat((state.event_times, event["available_at"][:, None]), 1),
                             retrospective=state.retrospective | (present & (event["role"] == 2)),
                             hypothetical=state.hypothetical | (present & (event["role"] == 3)))

    def apply_event(self, state, event):
        present = self._event_present(state, event)
        if not present.any():
            return state.snapshot()
        if state.time_basis == "calendar_days":
            occurred, available = event["occurred_at"], event["available_at"]
            if not torch.isfinite(occurred[present]).all() or not torch.isfinite(available[present]).all():
                raise ValueError("Calendar events require verified occurrence and availability times")
            planned = (event["role"] == 1) | (event["operation"] == 4) | (event["role"] == 3)
            if (present & ~planned & (occurred > available)).any():
                raise ValueError("A delivered or retrospective event cannot become available before it occurred")
            state = self.advance(state, torch.where(present, available, state.time))
        local_batch = {name: event[name][:, None] for name in EVENT_FIELDS}
        if state.time_basis == "calendar_days":
            local_batch["time_features"] = self._calendar_features(local_batch, state.baseline_time)
        local_batch["event_mask"] = present[:, None]
        local = self.event_encoder.encode_local(self._mechanism_batch(local_batch))[:, 0]
        vectors = torch.cat((state.history, local.mean(1)[:, None]), 1)
        mask = torch.cat((state.history_mask, present[:, None]), 1)
        history = self.event_encoder.encode_history(vectors, mask)[:, -1]
        return self._apply_encoded(state, event, local, history, present)

    @staticmethod
    def _select_state(checkpoints, indices):
        base = checkpoints[0]
        b = len(base.z)
        values = {}
        for field in fields(base):
            value = getattr(base, field.name)
            if not isinstance(value, torch.Tensor):
                continue
            if field.name in ("history", "history_mask", "event_ids", "event_times"):
                # These variable-length caches are irrelevant to a read-only risk query.
                values[field.name] = value.clone()
            else:
                stacked = torch.stack([getattr(checkpoint, field.name) for checkpoint in checkpoints], 1)
                values[field.name] = stacked[torch.arange(b, device=indices.device), indices].clone()
        return base.updated(**values)

    def query_many(self, checkpoints, query_order, query_time=None):
        if self.cfg.objective == "terminal_state_v1":
            raise ValueError("The terminal head cannot score raw intermediate states; supply an explicit strategy rollout")
        if isinstance(checkpoints, PatientState):
            checkpoints = [checkpoints]
        orders = torch.as_tensor(query_order, device=checkpoints[0].z.device)
        if orders.ndim != 2 or orders.shape[0] != len(checkpoints[0].z):
            raise ValueError("query_order must have shape [B,Q]")
        if orders.is_floating_point() and not (orders == orders.long()).all():
            raise ValueError("query_order counts consumed events and must contain integers")
        orders = orders.long()
        if ((orders < 0) | (orders >= len(checkpoints))).any():
            raise ValueError("A query references an unavailable checkpoint")
        if query_time is not None and self.cfg.time_basis != "calendar_days":
            raise ValueError("Ordinal queries do not support calendar interpolation")
        if query_time is not None:
            query_time = torch.as_tensor(query_time, device=orders.device, dtype=torch.float32)
            if query_time.shape != orders.shape:
                raise ValueError("query_time must have shape [B,Q]")
        results = []
        for column in range(orders.shape[1]):
            state = self._select_state(checkpoints, orders[:, column])
            if query_time is not None:
                target = query_time[:, column]
                # No checkpoint can be used beyond the next available event without ingesting it.
                for checkpoint_index, checkpoint in enumerate(checkpoints[1:], 1):
                    if not checkpoint.history_mask.shape[1]:
                        continue
                    real = checkpoint.history_mask[:, -1]
                    available = checkpoint.event_times[:, -1]
                    skipped = (orders[:, column] < checkpoint_index) & real & (target >= available)
                    if skipped.any():
                        raise ValueError("Calendar query skipped an event already available at that time")
                state = self.advance(state, target)
            results.append(self.outcome(state))
        return torch.stack(results, 1) if results else checkpoints[0].z.new_zeros((len(orders), 0))

    def assimilate(self, state, ct1, available):
        if not self.cfg.assimilate_ct1:
            return state
        safe_ct = torch.where(available[:, None, None], ct1, torch.zeros_like(ct1))
        observed = self.encode_image(safe_ct)
        z = state.z
        for block in self.observation_blocks:
            z = block(z, observed)
        return state.updated(z=torch.where(available[:, None, None], z, state.z))

    @staticmethod
    def _calendar_features(batch, baseline_time):
        mask = batch["event_mask"].bool()
        occurred = torch.where(mask, batch["occurred_at"].float(), baseline_time[:, None])
        available = torch.where(mask, batch["available_at"].float(), baseline_time[:, None])
        delay = available - occurred
        since_baseline = available - baseline_time[:, None]
        return torch.stack((since_baseline/30., (occurred-baseline_time[:, None])/30.,
                            delay/30., torch.log1p(delay.abs()/30.),
                            (delay > 0).float(), torch.zeros_like(delay)), -1)

    @staticmethod
    def _validate_calendar_scan(batch, scan_index, scan_valid, unique):
        scan_time = batch["scan_time"]
        available = batch["available_at"]
        positions = torch.arange(unique.shape[1], device=unique.device)[None]
        consumed = positions < scan_index[:, None]
        future_consumed = unique & consumed & (available > scan_time[:, None])
        skipped_available = unique & ~consumed & (available <= scan_time[:, None])
        if ((future_consumed | skipped_available) & scan_valid[:, None]).any():
            raise ValueError("scan_time and scan_event_index describe inconsistent information prefixes")

    def forward(self, batch):
        forbidden = FORBIDDEN_FIELDS.intersection(batch)
        if forbidden:
            raise ValueError(f"Legacy or named-treatment fields cannot enter modality-event-v2: {sorted(forbidden)}")
        if batch.get("time_basis", self.cfg.time_basis) != self.cfg.time_basis:
            raise ValueError("Batch and model time_basis differ")
        if batch.get("schema", self.cfg.schema) != self.cfg.schema:
            raise ValueError("Batch and model schemas differ")
        orders, mask = batch["query_order"], batch["query_mask"].bool()
        if orders.ndim != 2 or orders.shape != mask.shape:
            raise ValueError("Query order and mask must have matching [B,Q] shapes")
        selected = orders[mask]
        length = batch["event_mask"].shape[1]
        if (not torch.isfinite(selected).all() or (selected != selected.long()).any()
                or ((selected < 0) | (selected > length)).any()):
            raise ValueError("Queries must count available event slots in [0,L]")
        if length:
            safe = torch.where(mask, orders, torch.zeros_like(orders)).long()
            real_event = batch["event_mask"].gather(1, (safe-1).clamp_min(0))
            if (mask & (safe > 0) & ~real_event.bool()).any():
                raise ValueError("A missing event cannot establish a queried landmark")
        unique = self._unique_mask(batch)
        terminal = self.cfg.objective == "terminal_state_v1"
        if terminal:
            self.validate_terminal_events(batch)
        encoded_batch = dict(batch, event_mask=unique)
        state = self.initialize(batch)
        if self.cfg.time_basis == "calendar_days":
            if not torch.isfinite(batch["occurred_at"][unique]).all() or not torch.isfinite(batch["available_at"][unique]).all():
                raise ValueError("Calendar events require verified occurrence and availability times")
            planned = (batch["role"] == 1) | (batch["operation"] == 4) | (batch["role"] == 3)
            if (unique & ~planned & (batch["occurred_at"] > batch["available_at"])).any():
                raise ValueError("A delivered or retrospective event cannot become available before it occurred")
            encoded_batch["time_features"] = self._calendar_features(encoded_batch, state.time)
        local, history = self.event_encoder(self._mechanism_batch(encoded_batch))
        baseline = state
        checkpoints = [state]
        pcr = state.z.new_zeros(len(state.z)) if terminal else self.pcr_output(
            self.pcr_pool(torch.cat((state.z, state.clinical), 1)).mean(1)).squeeze(-1)
        b, length = unique.shape
        scan_index = batch.get("scan_event_index", torch.ones(b, device=state.z.device, dtype=torch.long)).long()
        if terminal and (scan_index != 1).any():
            raise ValueError("terminal_state_v1 requires the post-neoadjuvant S1 scan_event_index == 1")
        if not terminal and ((scan_index < 0) | (scan_index > length)).any():
            raise ValueError("scan_event_index must reference a pre-assimilation checkpoint")
        scan_z = state.z
        image_valid = batch.get("image_valid", torch.ones((b, 2), device=state.z.device, dtype=torch.bool))[:, 1].bool()
        scan_valid = batch.get("scan_mask", image_valid).bool()
        if terminal:
            scan_valid = scan_valid & (unique[:, 0] if length else torch.zeros_like(scan_valid))
            if length:
                scan_valid = scan_valid & ((batch["role"][:, 0] == 0) | (batch["role"][:, 0] == 2))
        if self.cfg.assimilate_ct1 and ((scan_index < 1) & scan_valid).any():
            raise ValueError("CT1 assimilation requires a post-baseline scan checkpoint")
        if self.cfg.time_basis == "calendar_days":
            if "scan_time" not in batch or not torch.isfinite(batch["scan_time"][scan_valid]).all():
                raise ValueError("Calendar scan supervision requires verified scan_time")
            self._validate_calendar_scan(batch, scan_index, scan_valid, unique)
        if (scan_index == 0).any():
            selected = scan_index == 0
            before = state
            if self.cfg.time_basis == "calendar_days":
                before = self.advance(state, torch.where(selected & scan_valid, batch["scan_time"], state.time))
            scan_z = torch.where(selected[:, None, None], before.z, scan_z)
            state = self.assimilate(before, batch["ct1"], selected & scan_valid) if self.cfg.assimilate_ct1 else state
            checkpoints[0] = state
        drift_regularization = state.z.new_zeros(())
        for index in range(length):
            event = {name: batch[name][:, index] for name in EVENT_FIELDS}
            present = unique[:, index]
            if self.cfg.time_basis == "calendar_days":
                target = torch.where(present, event["available_at"], state.time)
                advanced = self.advance(state, target)
                drift_regularization = drift_regularization + (advanced.z-state.z).square().mean()
                state = advanced
            kwargs = {}
            if self.cfg.ablation == "static":
                prefix = local[:, :index+1].flatten(1, 2)
                padding = ~unique[:, :index+1, None].expand(-1, -1, 2).flatten(1)
                kwargs = dict(base=baseline, prefix_tokens=prefix, prefix_padding=padding)
            state = self._apply_encoded(state, event, local[:, index], history[:, index], present, **kwargs)
            if terminal and index == 0:
                pcr = self.pcr_output(self.pcr_pool(
                    torch.cat((state.z, state.memory, state.clinical), 1)).mean(1)).squeeze(-1)
            selected = scan_index == index+1
            before = state
            if selected.any() and self.cfg.time_basis == "calendar_days":
                before = self.advance(state, torch.where(selected & scan_valid, batch["scan_time"], state.time))
            scan_z = torch.where(selected[:, None, None], before.z, scan_z)
            if self.cfg.assimilate_ct1 and selected.any():
                state = self.assimilate(before, batch["ct1"], selected & scan_valid)
            checkpoints.append(state)
        query_order = batch["query_order"]
        query_mask = batch["query_mask"].bool()
        safe_order = torch.where(query_mask, query_order, torch.zeros_like(query_order)).long()
        query_time = batch.get("query_time")
        if query_time is not None:
            checkpoint_time = torch.stack([checkpoint.time for checkpoint in checkpoints], 1)
            query_time = torch.where(query_mask, query_time,
                                     checkpoint_time[torch.arange(b, device=state.z.device)[:, None], safe_order])
        forecast = self.decoder(scan_z) * self.image_scale + self.image_mean
        row = torch.arange(b, device=state.z.device)[:, None]
        retrospective = torch.stack([checkpoint.retrospective for checkpoint in checkpoints], 1)[row, safe_order]
        hypothetical = torch.stack([checkpoint.hypothetical for checkpoint in checkpoints], 1)[row, safe_order]
        extra = {}
        if terminal:
            if query_time is not None:
                raise ValueError("Ordinal terminal predictions do not accept query_time")
            complete = (unique[:, :3].all(1) if length >= 3 else
                        torch.zeros(b, device=state.z.device, dtype=torch.bool))
            factual = ~((batch["role"] == 1) | (batch["role"] == 3)).logical_and(unique).any(1)
            terminal_mask = query_mask & (safe_order == TERMINAL_EVENT_ORDER) & (complete & factual)[:, None]
            terminal_logits = self.outcome(checkpoints[3]) if length >= 3 else state.z.new_zeros(b)
            logits = torch.where(terminal_mask, terminal_logits[:, None], torch.zeros_like(query_order, dtype=state.z.dtype))
            query_mask = terminal_mask
            pcr_mask = self._factual_s1_pcr_mask(batch, unique, self.schema_enabled)
            extra = {"objective": self.cfg.objective, "terminal_mask": terminal_mask,
                     "terminal_event_order": TERMINAL_EVENT_ORDER, "pcr_mask": pcr_mask,
                     "pcr_cutoff": "post_neoadjuvant_s1", "s1_anchor": scan_z.mean(1)}
        else:
            logits = self.query_many(checkpoints, safe_order, query_time)
        if self.cfg.s1_report_concepts:
            concept_logits = state.z.new_zeros((b, 4))
            concept_mask = torch.zeros(b, device=state.z.device, dtype=torch.bool)
            if length:
                s1 = checkpoints[1]
                concept_logits = self.s1_report_output(self.s1_report_pool(
                    torch.cat((s1.z, s1.memory, s1.clinical), 1)).mean(1)).float()
                concept_mask = unique[:, 0] & ((batch["role"][:, 0] == 0) | (batch["role"][:, 0] == 2))
            extra.update(s1_concept_logits=concept_logits, s1_concept_mask=concept_mask,
                         s1_concept_names=self.s1_concept_names,
                         s1_concept_target_transforms=self.s1_concept_transforms,
                         s1_concept_target_stage=1, s1_concept_available_stage=2)
        return {"logits": logits, "query_mask": query_mask, "forecast": forecast,
                "forecast_mask": scan_valid, "pcr_logits": pcr.float(),
                "drift_regularization": drift_regularization / max(length, 1),
                "checkpoint_states": checkpoints, "scan_state": scan_z,
                "retrospective": retrospective, "hypothetical": hypothetical, **extra}
