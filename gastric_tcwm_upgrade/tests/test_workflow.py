from dataclasses import replace
from pathlib import Path
import copy
import json
import pytest
import torch
from stageworld_tcwm.config import TrainConfig
from stageworld_tcwm.data import atomic_save,make_split,split_indices
from stageworld_tcwm.synthetic import synthetic_cohort
from stageworld_tcwm.training import train,warmstart_spatial
from stageworld_tcwm.model import TreatmentBeliefWorld
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.evaluation import collect_predictions,evaluate_predictions,CensoringKM


def query_from(cohort):
    names={'clinical','treatment','interval_days','surgery','ct0','ct1','image_valid','ct1_available_stage','entry','post','post_mask'}
    return {'schema':'tcwm-query-v1','plan_source':'hypothetical','interval_source':'specified_query',
            'tensors':{k:v[:2].clone() for k,v in cohort.tensors.items() if k in names}}


def setup_case(tmp_path,config,survival=False,prior='gaussian',postoperative_dim=0):
    data=synthetic_cohort(n=40,image_dim=config.image_dim,seed=13,survival=survival,causes=2 if survival else 1,postoperative_dim=postoperative_dim)
    path=tmp_path/'data.pt';data.save(path)
    split=make_split(data.ids,data.tensors['binary'].numpy(),17)
    cfg=replace(config,endpoint='survival' if survival else 'binary',causes=2 if survival else 1,prior=prior,postoperative_dim=postoperative_dim)
    tc=TrainConfig(epochs=2,warmup_epochs=0,batch_size=16,samples_train=2,samples_eval=3,observation_dropout=0,patience=2)
    return data,path,split,cfg,tc

@pytest.mark.parametrize('survival,prior,post_dim',[(False,'gaussian',0),(False,'flow',0),(True,'gaussian',0),(True,'flow',0),(False,'gaussian',8)])
def test_end_to_end_train_export_evaluate_predict(tmp_path,config,survival,prior,post_dim):
    data,path,split,cfg,tc=setup_case(tmp_path,config,survival,prior,post_dim)
    report=train(path,split,cfg,tc,tmp_path/'run')
    assert report['test_evaluated'] is False
    predictor=Predictor(tmp_path/'run/inference.pt')
    query=query_from(data)
    for stage in (0,1,2):
        result=predictor.predict(query,stage,samples=3,horizons=[12.,24.] if survival else None,allow_extrapolation=True)
        assert result['causal_effects_identified'] is False
        p=result['recurrence_probability'] if survival else result['recorded_status_probability']
        assert torch.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        if survival:assert (p[:,1]>=p[:,0]).all()
    rows=split_indices(data,split)
    pred=collect_predictions(predictor.model,data,rows['test'],samples=3)
    stats=evaluate_predictions(pred,data,rows['test'],cfg,rows['train'])
    assert stats['selection_nll'] is not None
    before=torch.load(tmp_path/'run/inference.pt',weights_only=True)['model_state']
    train(path,split,cfg,tc,tmp_path/'run',resume=True)
    after=torch.load(tmp_path/'run/inference.pt',weights_only=True)['model_state']
    for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)


def test_predictor_rejects_labels_horizons_and_unknown_treatment(tmp_path,config):
    data,path,split,cfg,tc=setup_case(tmp_path,config)
    train(path,split,cfg,tc,tmp_path/'run');predictor=Predictor(tmp_path/'run/inference.pt')
    query=query_from(data)
    with pytest.raises(ValueError):predictor.predict(query,0,horizons=[12.],allow_extrapolation=True)
    query['tensors']['binary']=torch.zeros(2)
    with pytest.raises(ValueError):predictor.predict(query,0,allow_extrapolation=True)
    query['tensors'].pop('binary');query['tensors']['surgery'].zero_()
    with pytest.raises(ValueError):predictor.predict(query,0)


def test_predictor_rejects_future_plan_availability(tmp_path,config):
    data,path,split,cfg,tc=setup_case(tmp_path,config)
    train(path,split,cfg,tc,tmp_path/'run');p=Predictor(tmp_path/'run/inference.pt')
    q=query_from(data);q['plan_source']='documented_plan';q['tensors']['plan_available_stage']=torch.ones(2,dtype=torch.long)
    with pytest.raises(ValueError):p.predict(q,0,allow_extrapolation=True)
    q['tensors']['plan_available_stage'].zero_();q['interval_source']='retrospective_actual'
    with pytest.raises(ValueError):p.predict(q,0,allow_extrapolation=True)


def test_existing_run_requires_exact_contract(tmp_path,config):
    data,path,split,cfg,tc=setup_case(tmp_path,config)
    train(path,split,cfg,tc,tmp_path/'run')
    with pytest.raises(ValueError):train(path,split,cfg,tc,tmp_path/'run')
    with pytest.raises(ValueError):train(path,split,cfg,replace(tc,learning_rate=1e-3),tmp_path/'run',resume=True)


def test_epoch_interruption_resume_replays_exactly(tmp_path,config,monkeypatch):
    from stageworld_tcwm import training
    data,path,split,cfg,tc=setup_case(tmp_path,config)
    cfg=replace(cfg,dropout=.1);tc=replace(tc,observation_dropout=.25)
    train(path,split,cfg,tc,tmp_path/'full')
    real_save=training.atomic_save
    def interrupt(obj,path):
        real_save(obj,path)
        if Path(path).name=='last.pt' and obj['epoch']==0:
            raise InterruptedError('simulated interruption after epoch checkpoint')
    monkeypatch.setattr(training,'atomic_save',interrupt)
    with pytest.raises(InterruptedError):train(path,split,cfg,tc,tmp_path/'resume')
    monkeypatch.setattr(training,'atomic_save',real_save)
    train(path,split,cfg,tc,tmp_path/'resume',resume=True)
    a=torch.load(tmp_path/'full/last.pt',weights_only=True)
    b=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    for k in a['model_state']:torch.testing.assert_close(a['model_state'][k],b['model_state'][k],rtol=0,atol=0)
    assert a['history']==b['history']


def test_survival_rejects_binary_only_cohort(tmp_path,config):
    data,path,split,cfg,tc=setup_case(tmp_path,config)
    with pytest.raises(ValueError):train(path,split,replace(cfg,endpoint='survival'),tc,tmp_path/'run')


def test_legacy_fold_binding_is_checked(tmp_path,config):
    data,path,split,cfg,tc=setup_case(tmp_path,config)
    data.encoders={'fit_ids':split['test']};data.save(path)
    with pytest.raises(ValueError):train(path,split,cfg,tc,tmp_path/'run')


def test_warmstart_cannot_import_other_patients(tmp_path,config):
    p=tmp_path/'old.pt';atomic_save({'contract':{'train_ids':['wrong']},'model_state':{}},p)
    with pytest.raises(ValueError):warmstart_spatial(TreatmentBeliefWorld(config),p,['correct'])


def test_censoring_km_ties_and_left_limit():
    km=CensoringKM([1.,2.,2.,3.],[1,0,1,0])
    assert km.at(torch.tensor([2.]).numpy(),left=True).item()==pytest.approx(1.)
    assert km.at(torch.tensor([2.]).numpy()).item()==pytest.approx(.5)
