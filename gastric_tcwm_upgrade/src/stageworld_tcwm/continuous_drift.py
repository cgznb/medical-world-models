"""Calendar-only FP32 ODE; ordinal stages never become elapsed days."""
import torch
from torch import nn
from .spatial_dynamics import SpatialDynamicsBlock


class ContinuousDrift(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.step_days = cfg.ode_step_days
        self.blocks = nn.ModuleList([SpatialDynamicsBlock(h, 0.) for _ in range(cfg.drift_blocks)])
        self.active_status = nn.Embedding(3, h)
        self.active_scope = nn.Parameter(torch.randn(1, 1, h) * .02)
        self.phase = nn.Embedding(5, h)
        self.time_projection = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h))
        self.rate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.candidate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.register_buffer("schema_enabled", torch.tensor(cfg.schema_enabled, dtype=torch.bool))

    def derivative(self, state, z, time_days, modality_embedding):
        status = torch.where(state.active_known, state.active_value.long() + 1,
                             torch.zeros_like(state.active_value, dtype=torch.long))
        active = modality_embedding.weight[None] + self.active_status(status) + self.active_scope
        active_padding = (~self.schema_enabled).expand(len(z), -1).clone()
        active_padding[:, 6] = True
        relative_days = time_days - state.baseline_time
        time = self.time_projection(torch.stack((relative_days / 30., torch.log1p(relative_days.abs() / 30.)), -1))
        condition = torch.cat((state.clinical, state.memory, active, self.phase(state.phase)[:, None], time[:, None]), 1)
        padding = torch.cat((torch.zeros((len(z), 8), dtype=torch.bool, device=z.device),
                             active_padding, torch.zeros((len(z), 2), dtype=torch.bool, device=z.device)), 1)
        transformed = z
        for block in self.blocks:
            transformed = block(transformed, condition, padding)
        return self.rate(transformed).sigmoid() * (self.candidate(transformed).tanh() - z)

    def forward(self, state, target_time, modality_embedding):
        if state.time_basis != "calendar_days":
            raise ValueError("Continuous drift requires verified calendar_days; ordinal stages have no day interpolation")
        target_time = torch.as_tensor(target_time, dtype=torch.float32, device=state.z.device).expand_as(state.time)
        if not torch.isfinite(target_time).all() or not torch.isfinite(state.time).all():
            raise ValueError("Calendar propagation requires known baseline and query times")
        elapsed = target_time - state.time
        if (elapsed < 0).any():
            raise ValueError("Queries cannot propagate a checkpoint backwards in time")
        if not (elapsed > 0).any():
            return state.snapshot()
        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise ImportError("Calendar models require the official torchdiffeq solver") from exc
        with torch.autocast(device_type=state.z.device.type, enabled=False):
            source = state.updated(z=state.z.float(), clinical=state.clinical.float(), memory=state.memory.float())
            def rhs(tau, z):
                days = source.time + tau * elapsed
                return (elapsed / 30.)[:, None, None] * self.derivative(source, z, days, modality_embedding)
            solution = odeint(rhs, source.z, source.z.new_tensor([0., 1.]), method="rk4",
                               options={"step_size": min(1., self.step_days / float(elapsed.max().item()))})[-1]
        return state.updated(z=solution, time=target_time.clone())
