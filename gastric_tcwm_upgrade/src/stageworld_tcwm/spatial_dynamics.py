"""Shared four-branch spatial block for events and deterministic drift."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialDynamicsBlock(nn.Module):
    def __init__(self, hidden, dropout=0.):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(5)])
        self.self_attention = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.local = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv3d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden)
        self.mix = nn.Conv3d(hidden, hidden, 1)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4*hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, state, condition, condition_padding=None):
        normalized = self.norms[0](state)
        state = state + self.drop(self.self_attention(normalized, normalized, normalized, need_weights=False)[0])
        condition = self.norms[2](condition)
        state = state + self.drop(self.cross_attention(self.norms[1](state), condition, condition,
                                key_padding_mask=condition_padding, need_weights=False)[0])
        volume = self.norms[3](state).transpose(1, 2).reshape(-1, state.shape[-1], 3, 3, 3)
        local = self.mix(F.gelu(self.local(volume) + self.context(volume)))
        state = state + self.drop(local.flatten(2).transpose(1, 2))
        return state + self.drop(self.ffn(self.norms[4](state)))
