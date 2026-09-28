"""Exact piecewise-exponential competing risks and mixture likelihoods.

Rates are conditional on the current history (including event-free entry).
Integrate from the actual landmark entry; do not average logits or hazards.
"""
import math
import torch
from torch.nn import functional as F

def check_edges(edges):
    if edges.ndim != 1 or len(edges)<2 or edges[0] != 0 or not (edges[1:]>edges[:-1]).all():
        raise ValueError("Invalid time bins")

def exposure(time,entry,edges):
    # All three are in months since the same clinical time origin.
    return (torch.minimum(time[...,None],edges[1:])-
            torch.maximum(entry[...,None],edges[:-1])).clamp_min(0)

def mixture_binary_nll(logits,labels):
    # [B,M], [B] -> [B]. Average likelihood, NOT per-sample NLL.
    labels = labels[:,None].expand_as(logits)
    loglik = -F.binary_cross_entropy_with_logits(logits.float(),labels.float(),reduction="none")
    return -(torch.logsumexp(loglik,-1)-math.log(logits.shape[-1]))

def mixture_survival_nll(rates,time,event,entry,edges):
    """[B,M,J,C] rates; event 0=censoring, 1/2=cause; [B] times.

    Bins use [left,right), with the last right endpoint included. Events on
    internal boundaries use the bin on their right. No silent extrapolation.
    """
    rates = rates.float()
    check_edges(edges)
    if rates.ndim != 4 or rates.shape[2] != len(edges)-1:
        raise ValueError("Invalid rate tensor")
    if (rates <= 0).any() or not torch.isfinite(rates).all():
        raise ValueError("Rates must be finite and positive")
    if (event < 0).any() or (event > rates.shape[-1]).any():
        raise ValueError("Unsupported competing event")
    if (time > edges[-1]).any() or (time <= entry).any() or (entry < 0).any():
        raise ValueError("Invalid risk set or time beyond configured horizon")
    exp_time = exposure(time,entry,edges)
    cumulative = (rates.sum(-1)*exp_time[:,None]).sum(-1)
    bins = torch.bucketize(time.contiguous(),edges[1:-1],right=True)
    batch = torch.arange(len(time),device=time.device)
    chosen = rates[batch,:,bins,:]
    cause_idx = (event-1).clamp_min(0)[:,None,None].expand(-1,rates.shape[1],1)
    cause = chosen.gather(-1,cause_idx).squeeze(-1)
    loglik = -cumulative+(event>0)[:,None]*cause.log()
    return -(torch.logsumexp(loglik,1)-math.log(rates.shape[1]))

def competing_curves(rates,horizons,entry,edges):
    """Return per-component conditional survival [B,M,Q] and CIF [B,M,Q,C].

    horizons may be [Q] (absolute since origin) or [B,Q]. Predictions before
    entry or beyond fitted bins are errors, not extrapolated output.
    """
    rates = rates.float()
    if horizons.ndim == 1:
        horizons = horizons[None].expand(rates.shape[0],-1)
    if horizons.shape[0] != len(rates) or (horizons < entry[:,None]).any() or (horizons > edges[-1]).any():
        raise ValueError("Query must lie between current entry and maximum modeled time")
    b,m,j,c = rates.shape
    lengths = exposure(horizons,entry[:,None],edges) # B,Q,J
    total = rates.sum(-1)  # B,M,J
    increments = total[:,:,None,:]*lengths[:,None] # B,M,Q,J
    before = torch.cat((torch.zeros_like(increments[...,:1]),increments[...,:-1].cumsum(-1)),-1)
    probability = (-before).exp()*(-torch.expm1(-increments))
    ratio = rates/total[...,None]
    cif = (probability[...,None]*ratio[:,:,None]).sum(-2)
    survival = (-increments.sum(-1)).exp()
    return survival,cif
