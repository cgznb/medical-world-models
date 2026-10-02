"""An event-shared spatial jump and memory update, without regimen inputs."""
import torch
from torch import nn
from .backbone import CrossBlock
from .spatial_dynamics import SpatialDynamicsBlock


class EventJump(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.blocks = nn.ModuleList([SpatialDynamicsBlock(h, cfg.dropout) for _ in range(cfg.event_blocks)])
        self.delta = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        nn.init.normal_(self.delta[-1].weight, std=.001)
        nn.init.zeros_(self.delta[-1].bias)
        self.gate = nn.Linear(2*h, h)
        memory_depth = 1 if cfg.capacity_profile == "compact_v1" else 2
        self.memory_blocks = nn.ModuleList([CrossBlock(h, cfg.dropout) for _ in range(memory_depth)])

    def forward(self, state, condition, history, condition_padding=None):
        transformed = state.z
        for block in self.blocks:
            transformed = block(transformed, condition, condition_padding)
        gate = self.gate(torch.cat((state.z.mean(1), history), -1)).sigmoid()[:, None]
        return state.z + gate * self.delta(transformed)

    def update_memory(self, memory, state, event_tokens, padding=None):
        observations = torch.cat((state.mean(1, keepdim=True), event_tokens), 1)
        if padding is not None:
            padding = torch.cat((torch.zeros((len(state), 1), dtype=torch.bool, device=state.device), padding), 1)
        for block in self.memory_blocks:
            memory = block(memory, observations, key_padding_mask=padding)
        return memory
