"""Strict, independent configuration for modality-event-v2 models."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class TimelineConfig:
    architecture: str = "modality_timeline_v2"
    schema: str = "modality-event-v2"
    image_dim: int = 768
    clinical_dim: int = 32
    hidden: int = 128
    init_blocks: int = 2
    field_blocks: int = 2
    history_blocks: int = 3
    event_blocks: int = 3
    drift_blocks: int = 3
    observation_blocks: int = 2
    readout_blocks: int = 2
    dropout: float = .1
    time_basis: str = "ordinal_stage"
    ablation: str = "dynamic"
    objective: str = "legacy_multistage"
    capacity_profile: str = "protocol_v1"
    terminal_clinical_anchor: bool = False
    terminal_residual_scale: float = .25
    clinical_normalization: str = "standard"
    s1_report_concepts: bool = False
    assimilate_ct1: bool = False
    ode_step_days: float = 1.
    schema_enabled: tuple = (True, False, True, True, True, True, True)

    def validate(self):
        if self.schema != "modality-event-v2" or self.architecture != "modality_timeline_v2":
            raise ValueError("Timeline models require the modality-event-v2 schema")
        if self.hidden not in (64, 128) or self.clinical_dim != 32:
            raise ValueError("The protocol requires H64/H128 and clinical32")
        if self.capacity_profile not in ("protocol_v1", "compact_v1"):
            raise ValueError("Unknown capacity_profile")
        compact = self.capacity_profile == "compact_v1"
        if compact and self.hidden != 64:
            raise ValueError("compact_v1 requires hidden=64")
        for name, expected in (("init_blocks", 2), ("field_blocks", 2), ("history_blocks", 3),
                               ("event_blocks", 3), ("drift_blocks", 3),
                               ("observation_blocks", 2), ("readout_blocks", 2)):
            expected = 1 if compact else expected
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"{name} must equal the protocol depth {expected}")
        if isinstance(self.image_dim, bool) or not isinstance(self.image_dim, int) or self.image_dim < 1:
            raise ValueError("image_dim must be a positive integer")
        if self.time_basis not in ("ordinal_stage", "calendar_days"):
            raise ValueError("Unknown time_basis")
        if self.ablation not in ("dynamic", "static"):
            raise ValueError("ablation must be dynamic or static")
        if self.objective not in ("legacy_multistage", "terminal_state_v1"):
            raise ValueError("Unknown timeline objective")
        if not isinstance(self.terminal_clinical_anchor, bool):
            raise ValueError("terminal_clinical_anchor must be boolean")
        if self.terminal_clinical_anchor and self.objective != "terminal_state_v1":
            raise ValueError("The terminal clinical anchor requires terminal_state_v1")
        if (isinstance(self.terminal_residual_scale, bool)
                or not isinstance(self.terminal_residual_scale, (float, int))
                or not math.isfinite(self.terminal_residual_scale) or self.terminal_residual_scale <= 0):
            raise ValueError("terminal_residual_scale must be finite and positive")
        if self.clinical_normalization not in ("standard", "continuous_only"):
            raise ValueError("Unknown clinical_normalization")
        if not isinstance(self.s1_report_concepts, bool):
            raise ValueError("s1_report_concepts must be boolean")
        if self.s1_report_concepts and self.objective != "terminal_state_v1":
            raise ValueError("S1 report concepts require terminal_state_v1")
        if self.objective == "terminal_state_v1" and (
                self.ablation != "dynamic" or self.time_basis != "ordinal_stage" or self.assimilate_ct1):
            raise ValueError("terminal_state_v1 requires dynamic ordinal transitions without CT1 assimilation")
        if self.ablation == "static" and (self.time_basis != "ordinal_stage" or self.assimilate_ct1):
            raise ValueError("The static information comparator uses ordinal prefixes without CT1 assimilation")
        if not isinstance(self.assimilate_ct1, bool):
            raise ValueError("assimilate_ct1 must be boolean")
        if not isinstance(self.dropout, (float, int)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite in [0, 1)")
        if not isinstance(self.ode_step_days, (float, int)) or not math.isfinite(self.ode_step_days) or self.ode_step_days <= 0:
            raise ValueError("ode_step_days must be finite and positive")
        if len(self.schema_enabled) != 7 or any(type(value) is not bool for value in self.schema_enabled):
            raise ValueError("schema_enabled must contain seven booleans")
        if self.schema_enabled[1]:
            raise ValueError("Radiotherapy remains disabled pending an audited source and training population")
        self.schema_enabled = tuple(self.schema_enabled)
        return self

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Timeline configuration must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown timeline configuration fields: {sorted(unknown)}")
        return cls(**values).validate()

    def to_dict(self):
        return asdict(self.validate())


def load_timeline_config(path):
    return TimelineConfig.from_dict(json.loads(Path(path).read_text()))
