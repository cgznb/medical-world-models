from dataclasses import replace
import copy
import pytest
import torch
from stageworld_tcwm.model import TreatmentBeliefWorld
from stageworld_tcwm.data import Cohort,split_indices,make_split
from stageworld_tcwm.support import fit_support,support_flags


def run(model,batch,stage=2,aux=False):
    return model(batch,samples=3,seed=19,max_stage=stage,compute_aux=aux)['predictions']

@pytest.mark.parametrize('prior',['gaussian','flow'])
def test_future_ct1_cannot_change_s0(config,cohort,prior):
    model=TreatmentBeliefWorld(replace(config,prior=prior)).eval()
    batch=cohort.batch(torch.arange(3));model.fit_statistics(batch)
    before=run(model,batch)
    batch['ct1']=torch.randn_like(batch['ct1'])*100
    after=run(model,batch)
    torch.testing.assert_close(before[:,0],after[:,0],rtol=0,atol=0)
    assert not torch.allclose(before[:,1],after[:,1],atol=1e-7,rtol=0)


def test_s0_inference_does_not_read_ct1_key(config,cohort):
    model=TreatmentBeliefWorld(config).eval();batch=cohort.batch(torch.arange(2))
    batch.pop('ct1');run(model,batch,stage=0)


def test_no_new_information_no_artificial_risk_change(config,cohort):
    model=TreatmentBeliefWorld(config).eval();batch=cohort.batch(torch.arange(3))
    output=run(model,batch)
    torch.testing.assert_close(output[:,1],output[:,2],rtol=0,atol=0)

@pytest.mark.parametrize('availability',[2,3])
def test_observation_availability_mask(config,cohort,availability):
    model=TreatmentBeliefWorld(config).eval();batch=cohort.batch(torch.arange(3))
    batch['ct1_available_stage'].fill_(availability)
    output=run(model,batch)
    torch.testing.assert_close(output[:,0],output[:,1],rtol=0,atol=0)
    if availability==3:
        torch.testing.assert_close(output[:,0],output[:,2],rtol=0,atol=0)


def test_missing_ct1_exact_prior_fallback(config,cohort):
    model=TreatmentBeliefWorld(config).eval();batch=cohort.batch(torch.arange(2))
    batch['image_valid'][:,1]=False
    output=run(model,batch)
    torch.testing.assert_close(output[:,0],output[:,1],rtol=0,atol=0)


def test_treatment_is_not_erased(config,cohort):
    model=TreatmentBeliefWorld(config).eval();batch=cohort.batch(torch.arange(3))
    before=run(model,batch,0)
    batch['treatment']=1-batch['treatment']
    after=run(model,batch,0)
    assert not torch.allclose(before,after,atol=1e-7,rtol=0)


def test_postoperative_only_changes_s2(config,cohort):
    model=TreatmentBeliefWorld(replace(config,postoperative_dim=8)).eval()
    torch.nn.init.normal_(model.post_delta.weight,std=.1)
    batch=cohort.batch(torch.arange(3));batch['post']=torch.randn(3,4,8);batch['post_mask']=torch.ones(3,4,dtype=torch.bool)
    before=run(model,batch);batch['post']=torch.randn(3,4,8)*50
    after=run(model,batch)
    torch.testing.assert_close(before[:,:2],after[:,:2],rtol=0,atol=0)
    assert not torch.allclose(before[:,2],after[:,2],rtol=0,atol=1e-7)
    batch['post_mask'].fill_(False)
    assert torch.isfinite(run(model,batch)).all()


def test_normalization_uses_only_given_training_patients(config,cohort):
    a=TreatmentBeliefWorld(config);b=TreatmentBeliefWorld(config)
    train=torch.arange(20);a.fit_statistics(cohort.batch(train))
    cohort.tensors['ct0'][20:]+=1000;cohort.tensors['clinical'][20:]+=1000
    b.fit_statistics(cohort.batch(train))
    for k in ('tab_mean','tab_scale','image_mean','image_scale'):
        torch.testing.assert_close(getattr(a,k),getattr(b,k),rtol=0,atol=0)

@pytest.mark.parametrize('bad',['overlap','missing','duplicate'])
def test_patient_split_rejects_bad_membership(cohort,bad):
    split=make_split(cohort.ids,cohort.tensors['binary'].numpy())
    if bad=='overlap':split['test'][0]=split['train'][0]
    if bad=='missing':split['test'].pop()
    if bad=='duplicate':split['train'].append(split['train'][0])
    with pytest.raises(ValueError):split_indices(cohort,split)


def test_no_surgery_is_outside_training_support(cohort):
    batch=cohort.batch(torch.arange(len(cohort)))
    support=fit_support(batch)
    query=cohort.batch(torch.arange(2));query['surgery'].zero_()
    assert all(x['warnings'] for x in support_flags(support,query))


def test_survival_cannot_be_invented_from_partial_fields(cohort):
    cohort.tensors['time']=torch.ones(len(cohort))
    with pytest.raises(ValueError):cohort.validate()


def test_event_precedes_landmark_is_rejected():
    from stageworld_tcwm.synthetic import synthetic_cohort
    cohort=synthetic_cohort(32,16,survival=True)
    cohort.tensors['entry'][0,1]=cohort.tensors['time'][0]+1
    with pytest.raises(ValueError):cohort.validate()

@pytest.mark.parametrize('field,value',[('hidden',30),('latent_dim',7),('world_blocks',0),('endpoint','unknown')])
def test_configuration_contract(config,field,value):
    with pytest.raises(ValueError):replace(config,**{field:value}).validate()


def test_flow_training_and_state_gradients(config,cohort):
    model=TreatmentBeliefWorld(replace(config,prior='flow'))
    batch=cohort.batch(torch.arange(3))
    output=model(batch,samples=2,compute_aux=True)
    loss=output['predictions'].square().mean()+output['flow_loss'].mean()
    loss.backward()
    grads=[p.grad for p in model.flow.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads)>0


def test_pcr_cannot_read_surgery_or_post_ct1(config,cohort):
    model=TreatmentBeliefWorld(config).eval();batch=cohort.batch(torch.arange(3))
    before=model(batch,samples=2,seed=17)['pcr_logits']
    batch['surgery'].zero_();batch['ct1']*=100
    after=model(batch,samples=2,seed=17)['pcr_logits']
    torch.testing.assert_close(before,after,rtol=0,atol=0)
