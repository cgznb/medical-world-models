"""Explicit configurations; binary labels never silently become survival labels."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import math

@dataclass
class ModelConfig:
    image_dim: int = 768
    hidden: int = 128
    latent_dim: int = 32
    world_blocks: int = 4
    surgery_blocks: int = 4
    update_blocks: int = 2
    readout_blocks: int = 2
    readout_slots: int = 8
    dropout: float = 0.1
    endpoint: str = "binary"
    causes: int = 1
    # Months since the explicitly declared endpoint origin, not CT spacing.
    bin_edges: tuple = (0., 3., 6., 12., 18., 24., 36., 48., 60.)
    prior: str = "gaussian"
    flow_blocks: int = 4
    flow_train_steps: int = 4
    flow_eval_steps: int = 24
    postoperative_dim: int = 0
    observation_update: str = "latent"
    clinical_anchor: bool = False
    residual_scale: float = 1.0
    architecture: str = "token_world"
    ct_rank: int = 16
    predictive_transition: bool = True
    readout_kind: str = "attention"

    def validate(self):
        if self.hidden < 16 or self.hidden % 8 or self.latent_dim < 4 or self.latent_dim % 4:
            raise ValueError("hidden must be divisible by 8, latent_dim by 4")
        if self.image_dim < 1 or self.postoperative_dim < 0:
            raise ValueError("Invalid modality dimensions")
        if self.endpoint not in ("binary", "survival") or self.prior not in ("gaussian", "flow"):
            raise ValueError("Unknown endpoint/prior")
        if self.causes not in (1, 2):
            raise ValueError("This implementation supports recurrence or recurrence/death")
        if self.endpoint == "binary" and self.causes != 1:
            raise ValueError("Binary recorded status is not a competing-risk endpoint")
        if self.observation_update not in ("latent", "residual"):
            raise ValueError("Unknown observation update")
        if self.architecture not in ("token_world", "predictive_ct"):
            raise ValueError("Unknown model architecture")
        if self.readout_kind not in ("attention", "pooled"):
            raise ValueError("Unknown readout kind")
        if self.readout_kind == "pooled" and (self.architecture != "token_world" or self.endpoint != "binary"):
            raise ValueError("Pooled readout currently requires a binary token-world model")
        if self.ct_rank < 1 or (self.architecture == "predictive_ct" and self.ct_rank > self.image_dim):
            raise ValueError("CT rank must be between 1 and image_dim")
        if self.architecture == "predictive_ct" and (self.endpoint != "binary" or
                self.prior != "gaussian" or self.postoperative_dim):
            raise ValueError("Predictive CT currently supports binary Gaussian models without postoperative features")
        if self.clinical_anchor and self.endpoint != "binary":
            raise ValueError("Clinical logistic anchors require a binary endpoint")
        if not 0 < self.residual_scale <= 1:
            raise ValueError("Residual scale must be in (0, 1]")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")
        if len(self.bin_edges) < 2 or self.bin_edges[0] != 0 or any(b <= a for a,b in zip(self.bin_edges, self.bin_edges[1:])):
            raise ValueError("Strictly increasing finite bins beginning at zero are required")
        import math
        if not all(math.isfinite(x) for x in self.bin_edges):
            raise ValueError("Nonfinite bin")
        if min(self.world_blocks, self.surgery_blocks, self.update_blocks,
               self.readout_blocks, self.readout_slots, self.flow_blocks,
               self.flow_train_steps, self.flow_eval_steps) < 1:
            raise ValueError("Depths, slots and integration steps must be positive")
        return self

@dataclass
class TrainConfig:
    seed: int = 17
    epochs: int = 150
    warmup_epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    patience: int = 25
    min_delta: float = 1e-4
    samples_train: int = 4
    samples_eval: int = 16
    ct_weight: float = 0.1
    kl_weight: float = 0.02
    kl_balance: float = 0.8
    free_nats: float = 0.5
    pcr_weight: float = 0.2
    flow_weight: float = 1.0
    gradient_clip: float = 1.0
    observation_dropout: float = 0.1
    amp: bool = False
    device: str = "cpu"
    max_optimizer_steps: int | None = None
    warmup_optimizer_steps: int | None = None
    prior_weight: float = 0.0
    prior_ct_weight: float = 0.0
    readout_l2: float = 0.0
    readout_learning_rate: float | None = None
    checkpoint_selection: str = "validation_nll"
    stage_weights: tuple = (1., 1., 1.)
    include_initial_baseline: bool = False
    mc_seed_policy: str = "batch_start"
    mc_antithetic: bool = False
    validation_interval_steps: int | None = None
    max_supervised_steps: int | None = None
    gradient_probe_interval: int | None = None
    observation_recon_weight: float = 0.0

    def validate(self):
        if min(self.epochs, self.batch_size, self.patience, self.samples_train, self.samples_eval) < 1:
            raise ValueError("Invalid training counts")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("Require 0 <= warmup_epochs < epochs")
        if self.max_optimizer_steps is not None and self.max_optimizer_steps < 1:
            raise ValueError("Optimizer step budget must be positive")
        if self.warmup_optimizer_steps is not None and self.warmup_optimizer_steps < 0:
            raise ValueError("Warmup optimizer steps must be nonnegative")
        if (self.max_optimizer_steps is not None and self.warmup_optimizer_steps is not None
                and self.warmup_optimizer_steps >= self.max_optimizer_steps):
            raise ValueError("Optimizer budget must leave supervised updates after warmup")
        if not 0 <= self.kl_balance <= 1 or not 0 <= self.observation_dropout < 1:
            raise ValueError("Invalid KL/dropout configuration")
        if min(self.learning_rate, self.gradient_clip) <= 0:
            raise ValueError("Learning rate and gradient clip must be positive")
        if self.readout_learning_rate is not None and self.readout_learning_rate <= 0:
            raise ValueError("Readout learning rate must be positive")
        if self.checkpoint_selection not in ("validation_nll", "fixed_budget"):
            raise ValueError("Unknown checkpoint selection policy")
        if self.mc_seed_policy not in ("batch_start", "case_key"):
            raise ValueError("Unknown MC seed policy")
        if (len(self.stage_weights) != 3 or
                any(not math.isfinite(w) or w < 0 for w in self.stage_weights)):
            raise ValueError("stage_weights must contain three finite nonnegative weights")
        self.stage_weights = tuple(float(w) for w in self.stage_weights)
        for name in ("validation_interval_steps", "max_supervised_steps", "gradient_probe_interval"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.checkpoint_selection == "fixed_budget" and self.include_initial_baseline:
            raise ValueError("Initial baseline selection requires validation_nll selection")
        if min(self.weight_decay, self.free_nats, self.ct_weight, self.kl_weight, self.pcr_weight,
               self.flow_weight, self.prior_weight, self.prior_ct_weight, self.readout_l2,
               self.observation_recon_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        return self

def load_config(path):
    value = json.loads(Path(path).read_text())
    m = dict(value.get("model", {}))
    if "bin_edges" in m:
        m["bin_edges"] = tuple(m["bin_edges"])
    return ModelConfig(**m).validate(), TrainConfig(**value.get("train", {})).validate()

def config_dict(model, train):
    return {"model": asdict(model), "train": asdict(train)}
