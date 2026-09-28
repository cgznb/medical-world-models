"""Strict versioned configuration; small tests use the same architecture classes."""
from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import json
import hashlib
import math
import yaml

@dataclass
class EncoderConfig:
    phase_channels: int = 8
    stem_width: int = 48
    widths: tuple = (96, 192, 384)
    depths: tuple = (2, 2, 4)
    heads: tuple = (3, 6, 12)
    window: tuple = (2, 4, 4)
    token_grid: tuple = (2, 4, 4)
    dim: int = 192
    query_heads: int = 6
    anatomy_tokens: int = 4
    disease_tokens: int = 8
    predictor_depth: int = 4
    phase_mixer_depth: int = 2
    drop_path: float = 0.0
    checkpoint_blocks: bool = True
    mask_ratio: float = 0.5
    phase_drop_probability: float = 0.25

@dataclass
class NetworkConfig:
    # MONAI is the actual pinned dependency, never silently replaced by native.
    backend: str = "monai"
    time_basis: str = "calendar_days"  # calendar_days | stage_index
    use_future: bool = True
    channels: tuple = (128, 256, 384)
    num_res_blocks: int = 2
    attention_heads: int = 8
    history_depth: int = 3
    semantic_depth: int = 4
    readout_depth: int = 3
    max_visits: int = 8
    dropout: float = 0.1  # readout only; vector field is deterministic
    coupling: bool = True
    readout_source: str = "joint"  # joint | reencode
    checkpoint_blocks: bool = True
    pillar_dim: int = 1152
    future_residual_limit: float = 3.0
    clinical_prior: bool = True

@dataclass
class LossConfig:
    reconstruction: float = 1.0
    phase_difference: float = 0.1
    masked_jepa: float = 1.0
    variance: float = 0.1
    covariance: float = 0.01
    anatomy: float = 0.01
    pillar: float = 0.1
    dense_teacher: float = 0.0
    segmentation: float = 0.0
    kinetics: float = 0.0
    biomarkers: float = 0.0
    fm_image: float = 1.0
    fm_state: float = 1.0
    repa: float = 0.05
    real_pcr: float = 1.0
    marginal_pcr: float = 1.0
    observed_pcr: float = 0.25
    grounding: float = 0.1
    energy: float = 0.05
    prediction_grounding: float = 0.02
    residual_l2: float = 0.01
    # Explicit ablation, NOT the default probabilistic objective.
    per_sample_bce: bool = False

@dataclass
class TrainConfig:
    seed: int = 20260928
    device: str = "cuda"
    precision: str = "bf16"
    threads: int = 4
    batch_size: int = 1
    accumulation: int = 4
    lr: float = 1e-4
    joint_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup: int = 100
    ema_decay: float = 0.995
    representation_steps: int = 10000
    flow_steps: int = 30000
    readout_steps: int = 3000
    joint_steps: int = 5000
    validation_every: int = 100
    checkpoint_every: int = 100
    log_every: int = 10
    validation_cases: int = 32
    reverse_probability: float = 0.25
    joint_image_scope: str = "decoder"  # all | decoder | frozen
    strict_determinism: bool = False
    validate_samples: int = 4
    selection_generation_weight: float = 0.05
    allow_synthetic: bool = False

@dataclass
class SamplingConfig:
    train_samples: int = 2
    train_steps: int = 20
    inference_samples: int = 8
    inference_steps: int = 20
    method: str = "heun"

@dataclass
class Config:
    schema: str = "responsewm_v1"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def validate(self):
        e, n, t, s = self.encoder, self.network, self.training, self.sampling
        if self.schema != "responsewm_v1":
            raise ValueError("Unsupported configuration schema")
        dimensions = (*e.widths,*e.depths,*e.heads,*e.window,*e.token_grid,*n.channels,
                      e.stem_width,e.dim,e.query_heads,e.anatomy_tokens,e.disease_tokens,
                      e.predictor_depth,e.phase_mixer_depth,n.attention_heads,n.num_res_blocks,n.pillar_dim)
        if any(not isinstance(v,int) or isinstance(v,bool) or v < 1 for v in dimensions):
            raise ValueError("Architecture sizes must be positive integers")
        if e.phase_channels != 8 or len(e.widths) < 2:
            raise ValueError("24-channel, three-phase VQ contract required")
        if len(e.widths) != len(e.depths) or len(e.widths) != len(e.heads):
            raise ValueError("Encoder stages do not match")
        if any(w % h for w, h in zip(e.widths, e.heads)) or e.dim % e.query_heads or e.stem_width % e.query_heads:
            raise ValueError("Invalid attention dimensions")
        if len(e.window) != 3 or len(e.token_grid) != 3:
            raise ValueError("3D grids required")
        if len(n.channels) < 2 or any(c % n.attention_heads for c in n.channels):
            raise ValueError("Invalid image backbone")
        if n.time_basis not in {"calendar_days", "stage_index"}:
            raise ValueError("Unknown time basis")
        if n.backend not in {"monai", "native"} or n.readout_source not in {"joint", "reencode"}:
            raise ValueError("Unknown backbone/readout")
        if min(n.history_depth, n.semantic_depth, n.readout_depth) < 2:
            raise ValueError("Use multi-block history/semantic/readout stacks")
        if n.semantic_depth % 2 or n.max_visits < 2:
            raise ValueError("Even semantic depth and >=2 visits required")
        if t.precision not in {"fp32", "bf16"} or (t.device == "cpu" and t.precision != "fp32"):
            raise ValueError("Use fp32 on CPU, fp32/bf16 on CUDA")
        if t.joint_image_scope not in {"all", "decoder", "frozen"}:
            raise ValueError("Unknown joint image tuning scope")
        if not 0 <= t.reverse_probability <= 1 or not 0 < t.ema_decay < 1:
            raise ValueError("Invalid reverse probability / EMA")
        if not 0 < e.mask_ratio < 1 or not 0 <= e.phase_drop_probability <= 1 or not 0 <= e.drop_path < 1:
            raise ValueError("Invalid encoder regularization")
        if not 0 <= n.dropout < 1 or n.future_residual_limit <= 0:
            raise ValueError("Invalid readout regularization")
        positive = (t.lr, t.joint_lr, t.grad_clip, t.batch_size, t.accumulation,
                    t.validation_every, t.checkpoint_every, t.log_every, t.threads,
                    t.validation_cases, s.train_steps, s.inference_steps, s.inference_samples,
                    t.validate_samples)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Positive finite training/sampling values required")
        if s.train_samples < 2 or s.method not in {"euler", "heun"}:
            raise ValueError("Training distribution scores need K>=2 and a supported ODE solver")
        if t.weight_decay < 0 or t.warmup < 0 or t.selection_generation_weight < 0:
            raise ValueError("Negative optimizer/selection hyperparameter")
        if any(v < 0 for k, v in asdict(t).items() if k.endswith("_steps")):
            raise ValueError("Negative training budget")
        if any(not isinstance(v, bool) and (not math.isfinite(v) or v < 0) for v in asdict(self.loss).values()):
            raise ValueError("Invalid loss weights")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def from_dict(value):
    def build(cls, data):
        extra = set(data) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(extra)}")
        return cls(**data)
    v = dict(value)
    for key, cls in (("encoder", EncoderConfig), ("network", NetworkConfig), ("loss", LossConfig),
                     ("training", TrainConfig), ("sampling", SamplingConfig)):
        v[key] = build(cls, v.get(key, {}))
    cfg = build(Config, v)
    for obj, names in ((cfg.encoder, ("widths", "depths", "heads", "window", "token_grid")),
                       (cfg.network, ("channels",))):
        for name in names:
            setattr(obj, name, tuple(getattr(obj, name)))
    return cfg.validate()


def load_config(path):
    return from_dict(yaml.safe_load(Path(path).read_text()))
