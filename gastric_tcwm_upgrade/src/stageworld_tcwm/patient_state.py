"""Explicit persistent patient state; queries operate on snapshots."""
from dataclasses import dataclass, fields, replace
import torch


@dataclass(frozen=True)
class PatientState:
    z: torch.Tensor
    memory: torch.Tensor
    clinical: torch.Tensor
    phase: torch.Tensor
    active_value: torch.Tensor
    active_known: torch.Tensor
    planned_value: torch.Tensor
    planned_known: torch.Tensor
    time: torch.Tensor
    baseline_time: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    event_ids: torch.Tensor
    event_times: torch.Tensor
    retrospective: torch.Tensor
    hypothetical: torch.Tensor
    time_basis: str
    raw_clinical: torch.Tensor | None = None

    @property
    def Z(self):
        return self.z

    @property
    def M(self):
        return self.memory

    @property
    def C(self):
        return self.clinical

    @property
    def m(self):
        return self.memory

    @property
    def c(self):
        return self.clinical

    def updated(self, **values):
        return replace(self, **values)

    def snapshot(self):
        return replace(self, **{field.name: getattr(self, field.name).clone()
                               for field in fields(self)
                               if isinstance(getattr(self, field.name), torch.Tensor)})
