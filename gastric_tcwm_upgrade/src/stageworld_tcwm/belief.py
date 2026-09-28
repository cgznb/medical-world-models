"""Gaussian prior/posterior filtering. Dreamer-inspired, not a Dreamer reproduction."""
import torch
from torch import nn
from .backbone import CrossBlock, SetReadout

class GaussianParams(nn.Module):
    def __init__(self, hidden, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,2*hidden),nn.GELU(),nn.Linear(2*hidden,2*latent_dim))
    def forward(self, vector):
        mean, logvar = self.net(vector).chunk(2,-1)
        return mean, logvar.clamp(-6.,2.)

class ObservationPosterior(nn.Module):
    """Observation information enters q(z), never the S0 prior branch."""
    def __init__(self, hidden, latent_dim, blocks, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([CrossBlock(hidden,dropout) for _ in range(blocks)])
        self.pool = SetReadout(hidden,4,1,dropout)
        self.params = GaussianParams(hidden,latent_dim)
    def forward(self, predicted, observed):
        state = predicted
        for block in self.blocks:
            state = block(state,observed)
        return self.params(self.pool(state).mean(1))

class StochasticInjector(nn.Module):
    def __init__(self, hidden, latent_dim):
        super().__init__()
        self.affine = nn.Sequential(nn.Linear(latent_dim,hidden),nn.GELU(),nn.Linear(hidden,2*hidden))
        self.norm = nn.LayerNorm(hidden)
    def forward(self, state, z):
        # state [B,27,H]; z [B,M,Z] -> [B,M,27,H]
        scale, shift = self.affine(z).chunk(2,-1)
        return state[:,None] + .1*(self.norm(state)[:,None]*scale[:,:,None] + shift[:,:,None])

def sample_gaussian(mean, logvar, epsilon):
    return mean[:,None] + (.5*logvar).exp()[:,None]*epsilon

def diagonal_kl(q_mean,q_logvar,p_mean,p_logvar):
    # float32 log-density arithmetic, including under autocast.
    q_mean,q_logvar,p_mean,p_logvar = [x.float() for x in (q_mean,q_logvar,p_mean,p_logvar)]
    return .5*(p_logvar-q_logvar + (q_logvar.exp()+(q_mean-p_mean).square())/p_logvar.exp()-1).sum(-1)

def balanced_kl(q_mean,q_logvar,p_mean,p_logvar,balance=.8,free_nats=.5):
    dynamics = diagonal_kl(q_mean.detach(),q_logvar.detach(),p_mean,p_logvar)
    representation = diagonal_kl(q_mean,q_logvar,p_mean.detach(),p_logvar.detach())
    return balance*dynamics.clamp_min(free_nats)+(1-balance)*representation.clamp_min(free_nats)
