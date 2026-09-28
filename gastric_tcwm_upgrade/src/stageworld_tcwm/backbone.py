"""Spatial backbone adapted from the user's Generated651 commit aec8cbe.

SpatialTransition, ConditionTokens and TwoWayFusion retain that architecture.
TwoWayFusion derives from CLARITY (MIT); see licenses/CLARITY_LICENSE.txt.
Changes: configurable dropout, mask-safe observation handling, no endpoint imports.
No CT0/CT1 voxel correspondence is assumed by the observation update.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ConditionTokens(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.clinical = nn.Linear(32, hidden)
        self.treatment = nn.Linear(82, hidden)
        self.time = nn.Sequential(nn.Linear(65, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.types = nn.Parameter(torch.randn(1,6,hidden) * .02)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, -math.log(10000), 32)))

    def forward(self, x):
        if x.shape[-1] != 361:
            raise ValueError("Expected clinical32 + four descriptor groups328 + interval1")
        phase = x[:,-1:] * self.frequencies
        time = self.time(torch.cat((x[:,-1:], phase.sin(), phase.cos()), -1))
        return torch.cat((self.clinical(x[:,:32])[:,None],
                          self.treatment(x[:,32:360].reshape(-1,4,82)), time[:,None]), 1) + self.types

class SpatialTransition(nn.Module):
    def __init__(self, hidden, condition_count=6, dropout=.1):
        super().__init__()
        self.condition_count = condition_count
        self.transformer = nn.TransformerEncoderLayer(hidden, 4, 4*hidden, dropout,
                    "gelu", batch_first=True, norm_first=True)
        self.local = nn.Conv3d(hidden,hidden,3,padding=1,groups=hidden)
        self.context = nn.Conv3d(hidden,hidden,3,padding=2,dilation=2,groups=hidden)
        self.mix = nn.Conv3d(hidden,hidden,1)
        self.film = nn.Linear(hidden,2*hidden)
        self.norm = nn.GroupNorm(8,hidden)

    def forward(self, tokens):
        tokens = self.transformer(tokens)
        c, image = tokens[:,:self.condition_count], tokens[:,self.condition_count:]
        if image.shape[1] != 27:
            raise ValueError("The inherited spatial transition requires a 3x3x3 grid")
        volume = image.transpose(1,2).reshape(-1,image.shape[-1],3,3,3)
        scale,shift = self.film(c.mean(1)).chunk(2,-1)
        normed = self.norm(volume) * (1 + .1*scale[:,:,None,None,None])
        normed = normed + .1*shift[:,:,None,None,None]
        volume = volume + self.mix(F.gelu(self.local(normed) + self.context(normed)))
        return torch.cat((c,volume.flatten(2).transpose(1,2)),1)

class TwoWayFusion(nn.Module):
    """CLARITY-derived sequential pre-norm bidirectional feature fusion."""
    def __init__(self, hidden, dropout=.1):
        super().__init__()
        self.attention = nn.ModuleList([nn.MultiheadAttention(hidden,4,dropout=dropout,batch_first=True) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(4)])
        self.feedforward = nn.ModuleList([nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),
                 nn.Dropout(dropout),nn.Linear(4*hidden,hidden)) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
        self.final = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
    def forward(self, baseline, future):
        baseline = baseline + self.dropout(self.attention[0](self.norms[0](baseline),future,future,need_weights=False)[0])
        future = future + self.dropout(self.attention[1](self.norms[1](future),baseline,baseline,need_weights=False)[0])
        baseline = baseline + self.dropout(self.feedforward[0](self.norms[2](baseline)))
        future = future + self.dropout(self.feedforward[1](self.norms[3](future)))
        return self.final[0](baseline), self.final[1](future)

class CrossBlock(nn.Module):
    """Mask-safe pre-norm observation/query attention with a four-times-width FFN."""
    def __init__(self, hidden, dropout=.1):
        super().__init__()
        self.qnorm, self.knorm, self.fnorm = (nn.LayerNorm(hidden) for _ in range(3))
        self.attn = nn.MultiheadAttention(hidden,4,dropout=dropout,batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*hidden,hidden))
        self.drop = nn.Dropout(dropout)
    def forward(self, queries, observations, key_padding_mask=None):
        normalized = self.knorm(observations)
        queries = queries + self.drop(self.attn(self.qnorm(queries), normalized, normalized,
                    key_padding_mask=key_padding_mask, need_weights=False)[0])
        return queries + self.drop(self.ffn(self.fnorm(queries)))

class SetReadout(nn.Module):
    """Learned seed pooling, inspired by Set Transformer's PMA; not anatomical slots."""
    def __init__(self, hidden, slots=8, blocks=2, dropout=.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1,slots,hidden)*.02)
        self.blocks = nn.ModuleList([CrossBlock(hidden,dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
    def forward(self, tokens):
        result = self.seeds.expand(tokens.shape[0],-1,-1)
        for block in self.blocks:
            result = block(result,tokens)
        return self.norm(result)

class SurgeryTransition(nn.Module):
    """Original present-only residual transition; no invented surgery time or CT2."""
    def __init__(self, hidden, blocks=4, dropout=.1):
        super().__init__()
        self.clinical = nn.Linear(32,hidden)
        self.event = nn.Embedding(4,hidden)
        self.types = nn.Parameter(torch.randn(1,2,hidden)*.02)
        self.blocks = nn.ModuleList([SpatialTransition(hidden,2,dropout) for _ in range(blocks)])
        self.delta = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
    def forward(self, s1, clinical, status):
        present = status == 1
        if not present.any():
            return s1
        cond = torch.stack((self.clinical(clinical[present]),self.event(status[present])),1) + self.types
        tok = torch.cat((cond,s1[present]),1)
        for block in self.blocks:
            tok = block(tok)
        output = s1.clone()
        output[present] = s1[present] + self.delta(tok[:,2:])
        return output
