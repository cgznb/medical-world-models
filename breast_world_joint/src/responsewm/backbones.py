"""Two explicit image backends and the coupled semantic velocity network.

Native retains the V2 image parameter layout. MONAI calls its actual 1.5.1
modules with an explicit bottleneck boundary, rather than mutable forward hooks.
The semantic branch adapts adaLN-Zero conditioning (DiT) to state tokens.
The bidirectional bridge is new task-specific code, not a released pretrained model.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .legacy.layers import TimeEmbedding, SwinBlock3D, maybe_checkpoint
from .legacy.utils import group_count

class FiLMResBlock3D(nn.Module):
    def __init__(self, cin, cout, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(cin, min(32, cin//2)), cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(cout, min(32, cout//2)), cout)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2*cout))
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(c).to(h.dtype).chunk(2, -1)
        h = self.norm2(h)*(1+scale[..., None, None, None]) + shift[..., None, None, None]
        return self.skip(x) + self.conv2(F.silu(h))

class SpatialCrossAttention(nn.Module):
    def __init__(self, channels, context_dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.context = nn.Linear(context_dim, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0)
        self.output = nn.Linear(channels, channels)
    def forward(self, x, context):
        seq = x.flatten(2).transpose(1, 2)
        c = self.context(context)
        h = self.attention(self.norm(seq), c, c, need_weights=False)[0]
        return (seq + self.output(h)).transpose(1, 2).reshape_as(x)

class NativeImageBackbone(nn.Module):
    """Full V2 residual/Swin/cross-attention U-Net, not a smoke-only mock."""
    def __init__(self, cfg, context_dim):
        super().__init__()
        self.cfg = cfg
        widths = cfg.channels
        td = widths[0]*4
        self.time = TimeEmbedding(td)
        self.pooled_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, td))
        self.input = nn.Conv3d(48, widths[0], 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        previous = widths[0]
        for level, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(FiLMResBlock3D(previous, width, td))
                previous = width
            self.down.append(blocks)
            if level < len(widths)-1:
                self.downsample.append(nn.Conv3d(width, width, 3, stride=2, padding=1))
        self.mid1 = FiLMResBlock3D(widths[-1], widths[-1], td)
        self.mid_self = nn.Sequential(SwinBlock3D(widths[-1], cfg.attention_heads, (2,4,4)),
                                     SwinBlock3D(widths[-1], cfg.attention_heads, (2,4,4), shifted=True))
        self.mid_cross = SpatialCrossAttention(widths[-1], context_dim, cfg.attention_heads)
        self.mid2 = FiLMResBlock3D(widths[-1], widths[-1], td)
        self.up = nn.ModuleList()
        previous = widths[-1]
        for width in reversed(widths[:-1]):
            blocks = nn.ModuleList([FiLMResBlock3D(previous+width, width, td)])
            blocks.extend(FiLMResBlock3D(width, width, td) for _ in range(cfg.num_res_blocks-1))
            self.up.append(blocks)
            previous = width
        self.out_norm = nn.GroupNorm(group_count(widths[0], min(32, widths[0]//2)), widths[0])
        self.out = nn.Conv3d(widths[0], 48, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode(self, joint, tau, context):
        c = self.time(tau) + self.pooled_context(context.mean(1))
        x, skips = self.input(joint), []
        for level, blocks in enumerate(self.down):
            for block in blocks:
                x = block(x, c)
            skips.append(x)
            if level < len(self.downsample):
                x = self.downsample[level](x)
        x = self.mid2(self.mid_cross(self.mid_self(self.mid1(x, c)), context), c)
        return x, (c, skips)

    def decode(self, x, cache, context):
        c, skips = cache
        for blocks, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat((x, skip), 1)
            for block in blocks:
                x = block(x, c)
        return self.out(F.silu(self.out_norm(x)))

    def forward(self, joint, tau, context):
        h, cache = self.encode(joint, tau, context)
        return self.decode(h, cache, context)

    def tune_decoder(self):
        self.requires_grad_(False)
        for module in (self.up, self.mid1, self.mid2, self.mid_cross, self.mid_self, self.out_norm, self.out):
            module.requires_grad_(True)

class MonaiImageBackbone(nn.Module):
    """Parameter-compatible with V2 MonaiVelocityUNet.network (MONAI 1.5.1).

    The split forward is adapted from MONAI's Apache-2.0 forward implementation.
    Class conditioning and ControlNet residuals are intentionally not exposed.
    Test test_monai_split_parity checks native MONAI output equivalence when installed.
    """
    def __init__(self, cfg, context_dim):
        super().__init__()
        try:
            import monai
            from monai.networks.nets import DiffusionModelUNet
        except ImportError as exc:
            raise ImportError("Install monai==1.5.1 or explicitly choose network.backend=native") from exc
        if monai.__version__ != "1.5.1":
            raise RuntimeError("Pinned MONAI 1.5.1 required; audit split-forward before upgrading")
        self.network = DiffusionModelUNet(
            spatial_dims=3, in_channels=48, out_channels=48, channels=tuple(cfg.channels),
            num_res_blocks=cfg.num_res_blocks,
            attention_levels=tuple(i == len(cfg.channels)-1 for i in range(len(cfg.channels))),
            num_head_channels=tuple(c//cfg.attention_heads if i == len(cfg.channels)-1 else 0
                                    for i, c in enumerate(cfg.channels)),
            norm_num_groups=group_count(cfg.channels[0]), with_conditioning=True,
            cross_attention_dim=context_dim, transformer_num_layers=1,
            upcast_attention=True, use_flash_attention=False)
        self.cfg = cfg

    def encode(self, joint, tau, context):
        from monai.networks.nets.diffusion_model_unet import get_timestep_embedding
        n = self.network
        emb = n.time_embed(get_timestep_embedding(tau, n.block_out_channels[0]).to(joint.dtype))
        h = n.conv_in(joint)
        skips = [h]
        for block in n.down_blocks:
            h, residuals = block(hidden_states=h, temb=emb, context=context)
            skips.extend(residuals)
        h = n.middle_block(hidden_states=h, temb=emb, context=context)
        return h, (emb, skips)

    def decode(self, h, cache, context):
        emb, skips = cache
        for block in self.network.up_blocks:
            idx = -len(block.resnets)
            residuals, skips = skips[idx:], skips[:idx]
            h = block(hidden_states=h, res_hidden_states_list=residuals, temb=emb, context=context)
        return self.network.out(h)

    def forward(self, joint, tau, context):
        h, cache = self.encode(joint, tau, context)
        return self.decode(h, cache, context)

    def tune_decoder(self):
        self.requires_grad_(False)
        for module in (self.network.middle_block, self.network.up_blocks, self.network.out):
            module.requires_grad_(True)

class StateDiTBlock(nn.Module):
    """adaLN-Zero self attention + conditioned cross attention + gated FFN."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.memory_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(approximate="tanh"), nn.Linear(4*dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9*dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    def forward(self, s, condition, memory):
        values = self.modulation(condition).chunk(9, -1)
        a, b, g, ac, bc, gc, af, bf, gf = [v[:, None] for v in values]
        q = self.norm_self(s)*(1+b)+a
        s = s + g*self.self_attn(q, q, q, need_weights=False)[0]
        q = self.norm_cross(s)*(1+bc)+ac
        m = self.memory_norm(memory)
        s = s + gc*self.cross_attn(q, m, m, need_weights=False)[0]
        return s + gf*self.ff(self.norm_ff(s)*(1+bf)+af)

class BidirectionalBridge(nn.Module):
    """Both updates use the same pre-update pair; no hooks or mutable caches."""
    def __init__(self, channels, dim, heads):
        super().__init__()
        self.image_in = nn.Conv3d(channels, dim, 1)
        self.image_out = nn.Conv3d(dim, channels, 1)
        self.image_norm = nn.LayerNorm(dim)
        self.state_norm = nn.LayerNorm(dim)
        self.to_image = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.to_state = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.image_gate = nn.Parameter(torch.zeros(()))
        self.state_gate = nn.Parameter(torch.zeros(()))
    def forward(self, h, s):
        spatial = self.image_in(h)
        z = self.image_norm(spatial.flatten(2).transpose(1, 2))
        sn = self.state_norm(s)
        dz = self.to_image(z, sn, sn, need_weights=False)[0]
        ds = self.to_state(sn, z, z, need_weights=False)[0]
        dh = self.image_out(dz.transpose(1, 2).reshape_as(spatial))
        return h + self.image_gate.tanh()*dh, s + self.state_gate.tanh()*ds

class CoupledVelocity(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e, n = cfg.encoder, cfg.network
        self.cfg = cfg
        self.image = (MonaiImageBackbone if n.backend == "monai" else NativeImageBackbone)(n, e.dim)
        self.semantic_in = nn.Linear(e.dim, e.dim)
        self.token_position = nn.Parameter(torch.randn(1, 2*e.disease_tokens, e.dim)*0.02)
        self.time = TimeEmbedding(e.dim)
        self.context_condition = nn.Sequential(nn.LayerNorm(e.dim), nn.Linear(e.dim, e.dim))
        self.blocks = nn.ModuleList([StateDiTBlock(e.dim, e.query_heads) for _ in range(n.semantic_depth)])
        # Multiple residual exchanges at the bottleneck, not just output concatenation.
        self.bridges = nn.ModuleList([BidirectionalBridge(n.channels[-1], e.dim, e.query_heads)
                                      for _ in range(n.semantic_depth//2)])
        self.semantic_out = nn.Sequential(nn.LayerNorm(e.dim), nn.Linear(e.dim, e.dim))
        nn.init.zeros_(self.semantic_out[-1].weight)
        nn.init.zeros_(self.semantic_out[-1].bias)
        self.spatial_projection = nn.Conv3d(n.channels[-1], e.dim, 3, padding=1)

    def _forward(self, z, s, tau, context):
        h, cache = self.image.encode(z, tau, context)
        s = self.semantic_in(s) + self.token_position
        c = self.time(tau) + self.context_condition(context.mean(1))
        for index, block in enumerate(self.blocks):
            s = block(s, c, context)
            if index % 2 == 1 and self.cfg.network.coupling:
                h, s = self.bridges[index//2](h, s)
        vz = self.image.decode(h, cache, context)
        vs = self.semantic_out(s)
        grid = self.cfg.encoder.token_grid
        dense = F.adaptive_avg_pool3d(self.spatial_projection(h), grid).flatten(2).transpose(1, 2)
        return vz, vs, dense

    def forward(self, z, s, tau, context):
        if z.ndim != 5 or z.shape[1] != 48:
            raise ValueError("Expected joint image [B,48,D,H,W]")
        if s.shape != (len(z), 2*self.cfg.encoder.disease_tokens, self.cfg.encoder.dim):
            raise ValueError("Joint semantic shape mismatch")
        # Checkpoint the full coupled evaluation, without forward hooks or mutable state.
        if self.cfg.network.checkpoint_blocks and torch.is_grad_enabled() and (
            self.training or z.requires_grad or s.requires_grad
        ):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward, z, s, tau, context, use_reentrant=False, preserve_rng_state=True)
        return self._forward(z, s, tau, context)
