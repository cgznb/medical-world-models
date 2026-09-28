from dataclasses import replace
from pathlib import Path
import pytest
import torch

from stageworld_tcwm.config import TrainConfig
from stageworld_tcwm.data import make_split
from stageworld_tcwm.losses import total_loss,warmup_state
from stageworld_tcwm.model import TreatmentBeliefWorld
from stageworld_tcwm.synthetic import synthetic_cohort
from stageworld_tcwm.training import train


def fitted_model(config,cohort):
    model = TreatmentBeliefWorld(config)
    batch = cohort.batch(torch.arange(8))
    model.fit_statistics(batch)
    return model,batch


def test_warmup_does_not_decay_or_initialize_inactive_heads(config,cohort):
    model,batch = fitted_model(config,cohort)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    inactive = {name:parameter for name,parameter in model.named_parameters()
                if name.startswith(("outcome.","pcr_pool.","pcr_output.","surgery.","surgery_context."))}
    before = {name:parameter.detach().clone() for name,parameter in inactive.items()}
    cfg = TrainConfig(epochs=3,warmup_epochs=1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss,metrics = total_loss(model(batch,samples=2),batch,model,cfg,epoch=0)
        loss.backward()
        assert metrics["endpoint_active"]==metrics["pcr_active"]==0
        for parameter in inactive.values():
            assert parameter.grad is None
        optimizer.step()
    for name,parameter in inactive.items():
        assert parameter not in optimizer.state
        torch.testing.assert_close(parameter,before[name],rtol=0,atol=0)
    optimizer.zero_grad(set_to_none=True)
    loss,_ = total_loss(model(batch,samples=2),batch,model,cfg,epoch=1)
    loss.backward()
    optimizer.step()
    for name in ("outcome.output.4.bias","pcr_output.3.bias"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None
        assert int(optimizer.state[parameter]["step"])==1


def test_zero_weight_auxiliary_heads_have_no_optimizer_state(config,cohort):
    model,batch = fitted_model(replace(config,prior="flow"),cohort)
    cfg = TrainConfig(epochs=2,warmup_epochs=0,ct_weight=0,kl_weight=0,pcr_weight=0,flow_weight=0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    names = ("decoder.","pcr_pool.","pcr_output.")
    inactive = {name:parameter for name,parameter in model.named_parameters() if name.startswith(names)}
    before = {name:parameter.detach().clone() for name,parameter in inactive.items()}
    loss,metrics = total_loss(model(batch,samples=2),batch,model,cfg)
    loss.backward()
    optimizer.step()
    assert all(metrics[name+"_active"]==0 for name in ("ct","kl","pcr","flow"))
    for name,parameter in inactive.items():
        assert parameter.grad is None and parameter not in optimizer.state
        torch.testing.assert_close(parameter,before[name],rtol=0,atol=0)


def test_no_observed_enabled_loss_fails_clearly(config,cohort):
    model,batch = fitted_model(config,cohort)
    cfg = TrainConfig(epochs=2,warmup_epochs=1,ct_weight=0,kl_weight=0,flow_weight=0)
    with pytest.raises(ValueError,match="No enabled training loss"):
        total_loss(model(batch,samples=2),batch,model,cfg,epoch=0)


def test_step_warmup_overrides_epoch_setting():
    cfg = TrainConfig(epochs=3,warmup_epochs=2,warmup_optimizer_steps=2)
    assert warmup_state(cfg,epoch=0,optimizer_step=0)==(True,.5)
    assert warmup_state(cfg,epoch=0,optimizer_step=1)==(True,1.)
    assert warmup_state(cfg,epoch=0,optimizer_step=2)==(False,1.)
    with pytest.raises(ValueError,match="optimizer-step count"):
        warmup_state(cfg,epoch=0)


def setup_training(tmp_path,config):
    cohort = synthetic_cohort(n=40,image_dim=config.image_dim,seed=13)
    path = tmp_path/"cohort.pt"
    cohort.save(path)
    split = make_split(cohort.ids,cohort.tensors["binary"].numpy(),17)
    cfg = TrainConfig(epochs=5,warmup_epochs=3,warmup_optimizer_steps=2,
                     max_optimizer_steps=5,batch_size=8,samples_train=2,
                     samples_eval=2,observation_dropout=.25,patience=5)
    return path,split,cfg


def test_step_budget_counters_and_partial_epoch(config,tmp_path):
    path,split,cfg = setup_training(tmp_path,config)
    report = train(path,split,config,cfg,tmp_path/"run")
    checkpoint = torch.load(tmp_path/"run/last.pt",weights_only=True)
    history = checkpoint["history"]
    assert report["optimizer_steps"]==checkpoint["optimizer_steps"]==5
    assert report["supervised_steps"]==checkpoint["supervised_steps"]==3
    assert report["stop_reason"]=="optimizer_step_limit"
    assert history[-1]["epoch_optimizer_steps"]<report["steps_per_full_epoch"]
    assert sum(row["epoch_optimizer_steps"] for row in history)==5
    assert sum(row["epoch_supervised_steps"] for row in history)==3
    selected = torch.load(tmp_path/"run/best.pt",weights_only=True)
    assert report["selected_optimizer_steps"]==selected["optimizer_steps"]
    assert report["selected_supervised_steps"]==selected["supervised_steps"]
    inference = torch.load(tmp_path/"run/inference.pt",weights_only=True)
    assert inference["selected_supervised_steps"]==selected["supervised_steps"]
    head = dict(TreatmentBeliefWorld(config).named_parameters())
    parameter_index = list(head).index("outcome.output.4.bias")
    assert int(checkpoint["optimizer_state"]["state"][parameter_index]["step"])==3
    before = {k:v.clone() for k,v in checkpoint["model_state"].items()}
    train(path,split,config,cfg,tmp_path/"run",resume=True)
    after = torch.load(tmp_path/"run/last.pt",weights_only=True)
    assert after["optimizer_steps"]==5 and after["supervised_steps"]==3
    for name,value in before.items():
        torch.testing.assert_close(value,after["model_state"][name],rtol=0,atol=0)


def test_step_budget_interruption_resume_replays_exactly(config,tmp_path,monkeypatch):
    from stageworld_tcwm import training
    path,split,cfg = setup_training(tmp_path,config)
    config = replace(config,dropout=.1)
    train(path,split,config,cfg,tmp_path/"full")
    save = training.atomic_save
    def interrupt(payload,path):
        save(payload,path)
        if Path(path).name=="last.pt" and payload["epoch"]==0:
            raise InterruptedError("after an epoch checkpoint")
    monkeypatch.setattr(training,"atomic_save",interrupt)
    with pytest.raises(InterruptedError):
        train(path,split,config,cfg,tmp_path/"resume")
    monkeypatch.setattr(training,"atomic_save",save)
    train(path,split,config,cfg,tmp_path/"resume",resume=True)
    a = torch.load(tmp_path/"full/last.pt",weights_only=True)
    b = torch.load(tmp_path/"resume/last.pt",weights_only=True)
    assert a["history"]==b["history"]
    assert a["optimizer_steps"]==b["optimizer_steps"]==5
    assert a["supervised_steps"]==b["supervised_steps"]==3
    for name in a["model_state"]:
        torch.testing.assert_close(a["model_state"][name],b["model_state"][name],rtol=0,atol=0)


def test_step_budget_must_reach_supervised_training(config,tmp_path):
    path,split,cfg = setup_training(tmp_path,config)
    cfg = replace(cfg,warmup_optimizer_steps=None,warmup_epochs=3,max_optimizer_steps=2)
    with pytest.raises(ValueError,match="post-warmup optimizer step"):
        train(path,split,config,cfg,tmp_path/"run")


def test_resume_rejects_inconsistent_step_counters(config,tmp_path):
    path,split,cfg = setup_training(tmp_path,config)
    train(path,split,config,cfg,tmp_path/"run")
    checkpoint_path = tmp_path/"run/last.pt"
    checkpoint = torch.load(checkpoint_path,weights_only=True)
    checkpoint["supervised_steps"] -= 1
    torch.save(checkpoint,checkpoint_path)
    with pytest.raises(ValueError,match="counters disagree"):
        train(path,split,config,cfg,tmp_path/"run",resume=True)
