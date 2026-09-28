import math
import pytest
import torch
from stageworld_tcwm.survival import mixture_binary_nll,mixture_survival_nll,competing_curves


def test_binary_is_mixture_likelihood_not_average_loss():
    logits=torch.tensor([[-6.,6.]])
    loss=mixture_binary_nll(logits,torch.ones(1))
    assert loss.item()==pytest.approx(math.log(2),abs=1e-6)
    assert loss.item()<torch.nn.functional.softplus(-logits).mean().item()

@pytest.mark.parametrize('event',[0,1,2])
def test_constant_hazard_analytic_likelihood(event):
    rates=torch.tensor([[[[.1,.2],[.1,.2]]]])
    nll=mixture_survival_nll(rates,torch.tensor([5.]),torch.tensor([event]),torch.tensor([2.]),torch.tensor([0.,3.,10.]))
    expected=.3*3-(math.log([1.,.1,.2][event]) if event else 0)
    assert nll.item()==pytest.approx(expected,abs=1e-6)


def test_internal_boundary_uses_right_bin():
    rates=torch.tensor([[[[.1],[.8]]]])
    loss=mixture_survival_nll(rates,torch.tensor([3.]),torch.tensor([1]),torch.tensor([0.]),torch.tensor([0.,3.,10.]))
    assert loss.item()==pytest.approx(.3-math.log(.8),abs=1e-6)


def test_curves_mass_and_monotonicity():
    rates=torch.rand(3,4,3,2)*.1+.01
    edges=torch.tensor([0.,3.,6.,12.]);entry=torch.tensor([0.,1.,2.])
    surv,cif=competing_curves(rates,torch.tensor([2.,3.,6.,9.,12.]),entry,edges)
    torch.testing.assert_close(surv+cif.sum(-1),torch.ones_like(surv))
    assert (torch.diff(surv,dim=-1)<=1e-6).all()
    assert (torch.diff(cif,dim=-2)>=-1e-6).all()


def test_competing_cif_is_not_one_minus_all_cause_survival():
    rates=torch.tensor([[[[.1,.2]]]])
    survival,cif=competing_curves(rates,torch.tensor([5.]),torch.tensor([2.]),torch.tensor([0.,10.]))
    assert survival.item()==pytest.approx(math.exp(-.9),abs=1e-6)
    assert cif[0,0,0,0].item()==pytest.approx((1-math.exp(-.9))/3,abs=1e-6)


def test_conditioned_entry_zero_risk_at_entry():
    rates=torch.ones(2,3,2,1)*.02
    entry=torch.tensor([1.,4.]);horizons=entry[:,None]
    s,c=competing_curves(rates,horizons,entry,torch.tensor([0.,3.,10.]))
    torch.testing.assert_close(s,torch.ones_like(s));torch.testing.assert_close(c,torch.zeros_like(c))

@pytest.mark.parametrize('time,entry,event',[(11.,0.,0),(2.,3.,1),(3.,3.,0),(3.,-1.,1),(3.,0.,2)])
def test_invalid_likelihood_contract(time,entry,event):
    with pytest.raises(ValueError):
        mixture_survival_nll(torch.ones(1,2,1,1)*.1,torch.tensor([time]),torch.tensor([event]),torch.tensor([entry]),torch.tensor([0.,10.]))

@pytest.mark.parametrize('query',[[-1.],[11.]])
def test_no_query_extrapolation(query):
    with pytest.raises(ValueError):
        competing_curves(torch.ones(1,2,1,1),torch.tensor(query),torch.tensor([0.]),torch.tensor([0.,10.]))


def test_mixture_survival_not_mean_rate():
    rates=torch.tensor([[[[.01]],[[.2]]]])
    s,_=competing_curves(rates,torch.tensor([5.]),torch.tensor([0.]),torch.tensor([0.,10.]))
    assert s.mean().item()!=pytest.approx(math.exp(-.105*5),abs=1e-3)


def test_likelihood_backward_is_finite():
    raw=torch.randn(4,3,2,2,requires_grad=True)
    loss=mixture_survival_nll(torch.nn.functional.softplus(raw),torch.tensor([1.,3.,4.,6.]),torch.tensor([0,1,2,1]),torch.zeros(4),torch.tensor([0.,3.,6.])).mean()
    loss.backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
