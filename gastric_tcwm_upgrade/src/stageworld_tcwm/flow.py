"""Experimental conditional latent flow prior with four-token adaLN dynamics.

The affine path follows Flow Matching; adaLN conditioning follows the DiT design.
This is an independent small-latent implementation, not copied upstream weights.
See SOURCE_AUDIT.md: do not call its Gaussian auxiliary KL a KL against the flow.
The ODE time t in [0,1] is NOT clinical time.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ConditionalFlowBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden,elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden,elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden,4,batch_first=True)
        self.cross = nn.MultiheadAttention(hidden,4,batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden,4*hidden),nn.GELU(),nn.Linear(4*hidden,hidden))
        self.modulation = nn.Sequential(nn.SiLU(),nn.Linear(hidden,7*hidden))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    def forward(self,x,condition,context):
        shift,scale,gate,shift2,scale2,gate2,cross_gate = self.modulation(condition).chunk(7,-1)
        q = self.norm1(x)*(1+scale[:,None])+shift[:,None]
        x = x + gate[:,None]*self.attn(q,q,q,need_weights=False)[0]
        x = x + cross_gate[:,None]*self.cross(q,context,context,need_weights=False)[0]
        x = x + gate2[:,None]*self.ff(self.norm2(x)*(1+scale2[:,None])+shift2[:,None])
        return x

class ConditionalFlowPrior(nn.Module):
    def __init__(self, latent_dim, hidden, blocks):
        super().__init__()
        self.latent_dim = latent_dim
        self.input = nn.Linear(latent_dim//4,hidden)
        self.position = nn.Parameter(torch.randn(1,4,hidden)*.02)
        self.time = nn.Sequential(nn.Linear(65,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.context_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([ConditionalFlowBlock(hidden) for _ in range(blocks)])
        self.output = nn.Linear(hidden,latent_dim//4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("freq",torch.exp(torch.linspace(0,-math.log(10000),32)))
    def velocity(self,z,t,context):
        phase = t[:,None]*self.freq[None]*1000
        c = self.time(torch.cat((t[:,None],phase.sin(),phase.cos()),-1))
        context = self.context_norm(context)
        c = c + context.mean(1)
        x = self.input(z.reshape(-1,4,self.latent_dim//4)) + self.position
        for block in self.blocks:
            x = block(x,c,context)
        return self.output(x).flatten(1)
    def loss(self,target,context,generator=None):
        # Target posterior is detached; no CT1 observation enters the flow context.
        target = target.detach()
        source = torch.randn(target.shape,device=target.device,generator=generator)
        t = torch.rand(len(target),device=target.device,generator=generator)
        position = (1-t[:,None])*source+t[:,None]*target
        return (self.velocity(position,t,context)-(target-source)).square().mean(-1)
    def sample(self,epsilon,context,steps):
        if steps < 1:
            raise ValueError("Integration steps must be positive")
        b,m,zdim = epsilon.shape
        z = epsilon.reshape(b*m,zdim)
        context = context[:,None].expand(-1,m,-1,-1).reshape(b*m,*context.shape[1:])
        # Differentiable Heun solver; inference decorated at API level, not here.
        dt = 1./steps
        for i in range(steps):
            t = torch.full((b*m,),i*dt,device=z.device,dtype=z.dtype)
            first = self.velocity(z,t,context)
            second = self.velocity(z+dt*first,t+dt,context)
            z = z + .5*dt*(first+second)
        return z.reshape(b,m,zdim)
