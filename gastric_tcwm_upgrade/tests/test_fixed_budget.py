from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch

from stageworld_tcwm.config import TrainConfig
from stageworld_tcwm.data import make_split
from stageworld_tcwm.synthetic import synthetic_cohort
from stageworld_tcwm.training import train


def setup_case(tmp_path,config,max_steps=None):
    cohort = synthetic_cohort(n=40,image_dim=config.image_dim,seed=13)
    path = tmp_path/"cohort.pt"
    cohort.save(path)
    split = make_split(cohort.ids,cohort.tensors["binary"].numpy(),17)
    cfg = TrainConfig(epochs=3,warmup_epochs=0,batch_size=8,samples_train=2,
                      samples_eval=2,patience=1,observation_dropout=.2,
                      max_optimizer_steps=max_steps,checkpoint_selection="fixed_budget")
    return cohort,path,split,cfg


@pytest.mark.parametrize("max_steps",[None,5])
def test_fixed_budget_only_evaluates_at_final_budget(config,tmp_path,monkeypatch,max_steps):
    from stageworld_tcwm import training
    _,path,split,cfg = setup_case(tmp_path,config,max_steps)
    collect = training.collect_predictions
    saves = training.atomic_save
    calls,checkpoints = [],[]
    def record_collection(*args,**kwargs):
        calls.append(len(checkpoints))
        return collect(*args,**kwargs)
    def record_checkpoint(payload,path):
        if Path(path).name=="last.pt":
            checkpoints.append(payload["optimizer_steps"])
        saves(payload,path)
    monkeypatch.setattr(training,"collect_predictions",record_collection)
    monkeypatch.setattr(training,"atomic_save",record_checkpoint)
    report = train(path,split,config,cfg,tmp_path/"run")
    last = torch.load(tmp_path/"run/last.pt",weights_only=True)
    best = torch.load(tmp_path/"run/best.pt",weights_only=True)
    inference = torch.load(tmp_path/"run/inference.pt",weights_only=True)
    assert len(calls)==1 and calls[0]==len(checkpoints)-1
    assert all(not row["validation"]["evaluated"] and row["validation"]["selection_nll"] is None
               for row in last["history"][:-1])
    assert last["history"][-1]["validation"]["evaluated"]
    assert last["stale"]==0
    assert best["epoch"]==last["epoch"]==inference["selected_epoch"]
    assert best["optimizer_steps"]==report["planned_optimizer_steps"]==report["actual_optimizer_steps"]
    assert report["selection_policy"]==inference["selection_policy"]=="fixed_budget"
    assert report["best_validation_nll"] is None and report["final_validation_nll"] is not None
    assert report["budget_fulfilled"] and report["validation_evaluations"]==1
    assert report["validation_used_for_checkpoint_selection"] is False
    assert report["stop_reason"]==("epoch_limit" if max_steps is None else "optimizer_step_limit")
    json.loads((tmp_path/"run/training_report.json").read_text(),parse_constant=lambda value:pytest.fail(value))


def test_fixed_budget_validation_labels_cannot_change_export(config,tmp_path):
    cohort,path,split,cfg = setup_case(tmp_path,config)
    cfg = replace(cfg,epochs=2)
    train(path,split,config,cfg,tmp_path/"original")
    lookup = {patient:index for index,patient in enumerate(cohort.ids)}
    rows = torch.tensor([lookup[patient] for patient in split["validation"]])
    cohort.tensors["binary"][rows] = 1-cohort.tensors["binary"][rows]
    altered = tmp_path/"altered.pt"
    cohort.save(altered)
    train(altered,split,config,cfg,tmp_path/"altered")
    a = torch.load(tmp_path/"original/inference.pt",weights_only=True)
    b = torch.load(tmp_path/"altered/inference.pt",weights_only=True)
    assert a["selected_epoch"]==b["selected_epoch"]==1
    for name in a["model_state"]:
        torch.testing.assert_close(a["model_state"][name],b["model_state"][name],rtol=0,atol=0)


def test_fixed_budget_interruption_resume_preserves_export_and_counts(config,tmp_path,monkeypatch):
    from stageworld_tcwm import training
    _,path,split,cfg = setup_case(tmp_path,config,max_steps=6)
    config = replace(config,dropout=.1)
    train(path,split,config,cfg,tmp_path/"full")
    save = training.atomic_save
    def interrupt(payload,path):
        save(payload,path)
        if Path(path).name=="last.pt" and payload["epoch"]==0:
            raise InterruptedError("fixed-budget checkpoint saved")
    monkeypatch.setattr(training,"atomic_save",interrupt)
    with pytest.raises(InterruptedError):
        train(path,split,config,cfg,tmp_path/"resume")
    checkpoint = torch.load(tmp_path/"resume/last.pt",weights_only=True)
    assert checkpoint["best"] is None and not checkpoint["history"][-1]["validation"]["evaluated"]
    monkeypatch.setattr(training,"atomic_save",save)
    report = train(path,split,config,cfg,tmp_path/"resume",resume=True)
    a = torch.load(tmp_path/"full/last.pt",weights_only=True)
    b = torch.load(tmp_path/"resume/last.pt",weights_only=True)
    assert a["history"]==b["history"]
    assert a["optimizer_steps"]==b["optimizer_steps"]==report["actual_optimizer_steps"]==6
    for name in a["model_state"]:
        torch.testing.assert_close(a["model_state"][name],b["model_state"][name],rtol=0,atol=0)
    def no_more_validation(*args,**kwargs):
        pytest.fail("Completed fixed-budget resume must not rescore validation")
    monkeypatch.setattr(training,"collect_predictions",no_more_validation)
    again = train(path,split,config,cfg,tmp_path/"resume",resume=True)
    assert again==report


def test_fixed_budget_can_export_without_observed_validation_labels(config,tmp_path):
    cohort,path,split,cfg = setup_case(tmp_path,config,max_steps=2)
    lookup = {patient:index for index,patient in enumerate(cohort.ids)}
    rows = torch.tensor([lookup[patient] for patient in split["validation"]])
    cohort.tensors["binary_valid"][rows] = False
    cohort.tensors["prefix_valid"][rows] = False
    cohort.save(path)
    report = train(path,split,config,cfg,tmp_path/"run")
    assert report["final_validation_nll"] is None
    assert report["best_validation_nll"] is None
    assert report["budget_fulfilled"]
