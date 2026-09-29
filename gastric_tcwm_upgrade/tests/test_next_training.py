from dataclasses import replace
import json
import importlib.util
from pathlib import Path

import pytest
import torch

from stageworld_tcwm.config import TrainConfig
from stageworld_tcwm.data import Cohort,file_sha256,make_split,fingerprint,write_json
from stageworld_tcwm.inference import Predictor
from stageworld_tcwm.losses import endpoint_loss,patient_weighted_stage_mean,total_loss
from stageworld_tcwm.model import TreatmentBeliefWorld
from stageworld_tcwm.synthetic import synthetic_cohort
from stageworld_tcwm.training import train,optimizer_parameter_groups


def test_stage_weight_normalization_preserves_eligibility_and_gradient():
    values = torch.tensor([[1.,3.,3.],[7.,5.,5.],[2.,9.,9.]],requires_grad=True)
    valid = torch.tensor([[True,True,True],[False,True,True],[False,False,True]])
    original = valid.clone()
    result = patient_weighted_stage_mean(values,valid,(.5,.5,0.))
    assert result.item() == pytest.approx(3.5)
    expected = patient_weighted_stage_mean(values[:,:2],valid[:,:2],(.5,.5))
    actual_grad = torch.autograd.grad(result,values,retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected,values)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=0,atol=0)
    torch.testing.assert_close(valid,original)
    assert not actual_grad[:,2].any()


def test_survival_entry_stages_remain_distinct(config):
    cfg = replace(config,endpoint="survival")
    prediction = torch.ones(1,3,2,len(cfg.bin_edges)-1,1)*.1
    batch = {"prefix_valid":torch.ones(1,3,dtype=torch.bool),"time":torch.tensor([20.]),
             "event":torch.tensor([1]),"entry":torch.tensor([[0.,3.,12.]])}
    first = endpoint_loss({"predictions":prediction},batch,cfg,(0.,1.,0.))
    later = endpoint_loss({"predictions":prediction},batch,cfg,(0.,0.,1.))
    assert first.item() != pytest.approx(later.item())


def setup_case(tmp_path,config,**kwargs):
    cohort = synthetic_cohort(n=40,image_dim=config.image_dim,seed=13)
    path = tmp_path/"cohort.pt"
    cohort.save(path)
    split = make_split(cohort.ids,cohort.tensors["binary"].numpy(),17)
    cfg = TrainConfig(epochs=4,warmup_epochs=0,batch_size=8,samples_train=2,samples_eval=2,
                      observation_dropout=.2,**kwargs)
    return path,split,cfg


def test_baseline_can_win_and_resume_preserves_last_neural_state(config,tmp_path):
    cfg = replace(config,clinical_anchor=True,observation_update="residual")
    path,split,tc = setup_case(tmp_path,cfg,include_initial_baseline=True,min_delta=100.,
                              max_supervised_steps=3,validation_interval_steps=1,patience=5)
    report = train(path,split,cfg,tc,tmp_path/"run")
    best = torch.load(tmp_path/"run/best.pt",weights_only=True)
    last = torch.load(tmp_path/"run/last.pt",weights_only=True)
    bundle = torch.load(tmp_path/"run/inference.pt",weights_only=True)
    assert report["selected_kind"] == bundle["selected_kind"] == "clinical_baseline"
    assert report["selected_supervised_steps"] == best["supervised_steps"] == 0
    assert last["supervised_steps"] == report["supervised_steps"] == 3
    assert not torch.equal(best["model_state"]["outcome.output.4.bias"],last["model_state"]["outcome.output.4.bias"])
    assert train(path,split,cfg,tc,tmp_path/"run",resume=True) == report


def test_baseline_post_warmup_and_step_early_stopping(config,tmp_path):
    cfg = replace(config,clinical_anchor=True,observation_update="residual")
    path,split,tc = setup_case(tmp_path,cfg,include_initial_baseline=True,min_delta=100.,
                              warmup_optimizer_steps=2,validation_interval_steps=1,patience=2)
    report = train(path,split,cfg,tc,tmp_path/"run")
    head0 = torch.load(tmp_path/"run/baseline_head_start.pt",weights_only=True)
    assert head0["optimizer_steps"] == 2 and head0["supervised_steps"] == 0
    assert report["optimizer_steps"] == 4 and report["supervised_steps"] == 2
    assert report["stop_reason"] == "early_stopping"
    assert report["early_stopping_unit"] == "validation_cycle"


def test_mid_epoch_validation_resume_replays_exactly(config,tmp_path,monkeypatch):
    from stageworld_tcwm import training
    config = replace(config,dropout=.1)
    path,split,tc = setup_case(tmp_path,config,max_supervised_steps=5,validation_interval_steps=2,
                              patience=10,gradient_probe_interval=2)
    train(path,split,config,tc,tmp_path/"full")
    save = training.atomic_save
    def interrupt(value,path):
        save(value,path)
        if Path(path).name == "last.pt" and value["optimizer_steps"] == 2:
            raise InterruptedError("mid-epoch checkpoint")
    monkeypatch.setattr(training,"atomic_save",interrupt)
    with pytest.raises(InterruptedError):
        train(path,split,config,tc,tmp_path/"resume")
    monkeypatch.setattr(training,"atomic_save",save)
    train(path,split,config,tc,tmp_path/"resume",resume=True)
    full = torch.load(tmp_path/"full/last.pt",weights_only=True)
    resumed = torch.load(tmp_path/"resume/last.pt",weights_only=True)
    assert full["history"] == resumed["history"]
    assert full["gradient_probes"] == resumed["gradient_probes"]
    assert full["task_counts"] == resumed["task_counts"]
    for name in full["model_state"]:
        torch.testing.assert_close(full["model_state"][name],resumed["model_state"][name],rtol=0,atol=0)


def test_optimizer_groups_cover_all_parameters_once(config):
    model = TreatmentBeliefWorld(config)
    groups,manifest = optimizer_parameter_groups(model,TrainConfig(learning_rate=2e-5,readout_learning_rate=1e-4))
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert [group["lr"] for group in groups] == [2e-5,1e-4]
    assert all(name.startswith(("outcome.","pcr_output.")) for name in manifest[1]["parameter_names"])


def test_zero_stage_weights_leave_endpoint_optimizer_inactive(config,cohort):
    model = TreatmentBeliefWorld(config)
    batch = cohort.batch(torch.arange(8))
    model.fit_statistics(batch)
    tc = TrainConfig(epochs=2,warmup_epochs=0,stage_weights=(0.,0.,0.),pcr_weight=0.)
    loss,metrics = total_loss(model(batch,samples=2),batch,model,tc)
    loss.backward()
    assert metrics["endpoint_active"] == 0
    assert all(p.grad is None for name,p in model.named_parameters() if name.startswith("outcome."))


@pytest.mark.parametrize("field,value",[("stage_weights",[-1,1,1]),("stage_weights",[1,1]),
    ("stage_weights",[1,float("nan"),1]),("validation_interval_steps",0),("max_supervised_steps",1.5),
    ("gradient_probe_interval",-1),("mc_seed_policy","unknown")])
def test_new_config_rejects_invalid_values(field,value):
    with pytest.raises(ValueError):
        TrainConfig(**{field:value}).validate()


def test_legacy_contract_default_migration_is_explicit_and_strict(config,tmp_path):
    path,split,tc = setup_case(tmp_path,config,max_optimizer_steps=2)
    cohort = Cohort.load(path)
    cohort.encoders["fit_ids"] = split["train"]
    cohort.metadata.update({"outer_evaluation_ids":split["test"],"excluded_ids":[],"inner_fold":0})
    cohort.save(path)
    report = train(path,split,config,tc,tmp_path/"run")
    contract_path = tmp_path/"run/contract.json"
    old = json.loads(contract_path.read_text())
    added = ("stage_weights","include_initial_baseline","mc_seed_policy","mc_antithetic",
             "validation_interval_steps","max_supervised_steps","gradient_probe_interval","observation_recon_weight")
    for name in added:
        old["contract"]["train"].pop(name)
    old["id"] = fingerprint(old["contract"])
    write_json(old,contract_path)
    for name in ("last.pt","best.pt"):
        payload = torch.load(tmp_path/"run"/name,weights_only=True)
        payload["contract_id"] = old["id"]
        torch.save(payload,tmp_path/"run"/name)
    resumed = train(path,split,config,tc,tmp_path/"run",resume=True)
    assert resumed["optimizer_steps"] == report["optimizer_steps"]
    preserved = json.loads(contract_path.read_text())
    assert preserved == old
    assert fingerprint(preserved["contract"]) == preserved["id"]
    effective = json.loads((tmp_path/"run/effective_config.json").read_text())
    assert all(name in effective["train"] for name in added)
    spec = importlib.util.spec_from_file_location("next_verify_ct_cv",
        Path(__file__).resolve().parents[1]/"scripts/evaluate_ct_cv.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    predictor = Predictor(tmp_path/"run/inference.pt")
    fold = {"cohort":cohort,"split":split,"hashes":{"cohort":file_sha256(path),"split":fingerprint(split)}}
    assert evaluation.verify_run(predictor,tmp_path/"run/inference.pt",fold) == old["contract"]
    with pytest.raises(ValueError,match="identical"):
        train(path,split,config,replace(tc,stage_weights=(.5,.5,0.)),tmp_path/"run",resume=True)
