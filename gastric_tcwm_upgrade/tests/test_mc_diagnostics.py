from dataclasses import asdict,replace
import json

import pytest
import torch

from stageworld_tcwm.diagnostics import dependency_probe,mc_stability_probe
from stageworld_tcwm.evaluation import collect_predictions,evaluate_predictions
from stageworld_tcwm.inference import ALLOWED,Predictor
from stageworld_tcwm.model import model_from_config
from stageworld_tcwm.monte_carlo import case_key_epsilon,cohort_case_keys,monte_carlo_standard_error
from stageworld_tcwm.support import fit_support


@pytest.mark.parametrize("antithetic",[False,True])
def test_case_noise_exact_for_batching_order_and_extended_draws(cohort,antithetic):
    rows = torch.arange(19)
    keys = cohort_case_keys(cohort,rows)
    full = case_key_epsilon(keys,8,8,29,antithetic=antithetic)
    for size in (1,8,16):
        split = torch.cat([case_key_epsilon(keys[start:start+size],8,8,29,antithetic=antithetic)
                           for start in range(0,len(keys),size)])
        assert torch.equal(full,split)
    order = torch.randperm(len(rows))
    reordered = case_key_epsilon([keys[int(i)] for i in order],8,8,29,antithetic=antithetic)
    assert torch.equal(reordered,full[order])
    extended = case_key_epsilon(keys,16,8,29,antithetic=antithetic)
    assert torch.equal(full,extended[:,:8])
    if antithetic:
        assert torch.equal(full[:,::2],-full[:,1::2])


@pytest.mark.parametrize("architecture",["token_world","predictive_ct"])
def test_prediction_batching_and_order_invariance(config,cohort,architecture):
    model = model_from_config(replace(config,architecture=architecture,ct_rank=4)).eval()
    model.fit_statistics(cohort.batch(torch.arange(24)))
    rows = torch.arange(19)
    expected = collect_predictions(model,cohort,rows,4,1,29,"case_key",True)
    for size in (8,16):
        actual = collect_predictions(model,cohort,rows,4,size,29,"case_key",True)
        torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-5)
    order = torch.randperm(len(rows))
    actual = collect_predictions(model,cohort,rows[order],4,8,29,"case_key",True)
    torch.testing.assert_close(expected[order],actual,atol=1e-6,rtol=1e-5)


def test_exported_inference_matches_evaluation_and_requires_case_metadata(config,cohort,tmp_path):
    model = model_from_config(config).eval()
    training = cohort.batch(torch.arange(24))
    model.fit_statistics(training)
    path = tmp_path/"inference.pt"
    torch.save({"schema":"tcwm-inference-v1","locked_selection":True,"model_config":asdict(config),
        "model_state":model.state_dict(),"support":fit_support(training),"metadata":cohort.metadata,
        "evaluation_config":{"mc_seed_policy":"case_key","mc_antithetic":True}},path)
    predictor = Predictor(path)
    rows = torch.arange(8)
    batch = cohort.batch(rows)
    query = {"schema":"tcwm-query-v1","plan_source":"retrospective_factual",
        "interval_source":"retrospective_actual","case_keys":cohort_case_keys(cohort,rows),
        "tensors":{key:value for key,value in batch.items() if key in ALLOWED}}
    expected = collect_predictions(model,cohort,rows,8,1,43,"case_key",True).sigmoid().mean(2)
    for stage in (0,1,2):
        actual = predictor.predict(query,stage,samples=8,seed=43,allow_extrapolation=True)
        torch.testing.assert_close(expected[:,stage],actual["recorded_status_probability"],atol=1e-6,rtol=1e-5)
        assert actual["mc_independent_units"] == 4
        assert "case_keys" not in actual
    query.pop("case_keys")
    with pytest.raises(ValueError,match="case key"):
        predictor.predict(query,0,samples=8,seed=43,allow_extrapolation=True)


def test_antithetic_standard_error_uses_pair_means():
    values = torch.tensor([[1.,3.,7.,9.]])
    torch.testing.assert_close(monte_carlo_standard_error(values,True),torch.tensor([3.]))
    assert monte_carlo_standard_error(values[:,:2],True) is None
    with pytest.raises(ValueError,match="even"):
        case_key_epsilon(["anonymous"],3,4,antithetic=True)


def test_health_and_dependency_keep_aggregate_only(config,cohort):
    model = model_from_config(replace(config,observation_update="residual")).eval()
    rows = torch.arange(4)
    model.fit_statistics(cohort.batch(torch.arange(24)))
    prediction,health = collect_predictions(model,cohort,rows,4,2,17,"case_key",True,True)
    assert health["raw_kl"]["mean"] >= 0
    assert health["free_nats_kl"]["mean"] >= health["raw_kl"]["mean"]
    assert "updated_injected_absolute_difference" in health
    assert "prior_active_units" in health
    report = dependency_probe(model,cohort,rows,samples=4,batch_size=2,antithetic=True)
    assert report["prefix_boundary_checks"]["passed"]
    assert report["perturbations"]["CT1"]["stages"]["S0"]["max_absolute_difference"] == 0
    assert all(identifier not in json.dumps(report) for identifier in cohort.ids)
    weighted = evaluate_predictions(prediction,cohort,rows,config,stage_weights=(.5,.5,0.))
    legacy = evaluate_predictions(prediction,cohort,rows,config)
    assert weighted["legacy_three_stage_nll"] == legacy["selection_nll"]
    assert weighted["stages"]["S2"]["n"] == legacy["stages"]["S2"]["n"]


def test_mc_probe_reports_every_seed_and_patient_count(config,cohort):
    model = model_from_config(config).eval()
    model.fit_statistics(cohort.batch(torch.arange(24)))
    report = mc_stability_probe(model,cohort,torch.arange(3),samples=(2,4),seeds=(17,29,43),antithetic=True)
    assert report["patients"] == 3
    assert all(len(row["selection_nll"]) == 3 and row["mc_nll_sd"] >= 0 for row in report["records"])
    assert not report["seed_selection_permitted"]


def test_new_bundle_defaults_reproduce_locked_selection(config,cohort,tmp_path):
    from stageworld_tcwm.cli import main
    from stageworld_tcwm.config import TrainConfig
    from stageworld_tcwm.data import make_split,split_indices,write_json
    from stageworld_tcwm.training import train

    data,split_path,run = tmp_path/"cohort.pt",tmp_path/"split.json",tmp_path/"run"
    cohort.save(data)
    split = make_split(cohort.ids,cohort.tensors["binary"].numpy(),17)
    write_json(split,split_path)
    cfg = TrainConfig(seed=29,epochs=2,warmup_epochs=0,max_optimizer_steps=2,
        batch_size=8,samples_train=2,samples_eval=4,mc_seed_policy="case_key",mc_antithetic=True,
        stage_weights=(.5,.5,0.))
    training = train(data,split,config,cfg,run)
    predictor = Predictor(run/"inference.pt")
    assert predictor.bundle["evaluation_config"]["mc_seed"] == 10029
    rows = split_indices(cohort,split)["validation"]
    batch = cohort.batch(rows)
    query = {"schema":"tcwm-query-v1","plan_source":"retrospective_factual",
        "interval_source":"retrospective_actual","case_keys":cohort_case_keys(cohort,rows),
        "tensors":{key:value for key,value in batch.items() if key in ALLOWED}}
    prediction = predictor.predict(query,1,allow_extrapolation=True)
    expected = collect_predictions(predictor.model,cohort,rows,4,8,10029,"case_key",True)
    assert prediction["samples"] == 4 and prediction["seed"] == 10029
    torch.testing.assert_close(prediction["recorded_status_probability"],expected[:,1].sigmoid().mean(1))
    main(["--threads","1","evaluate","--data",str(data),"--split",str(split_path),"--run",str(run)])
    evaluated = json.loads((run/"evaluation_validation.json").read_text())
    assert evaluated["samples"] == 4 and evaluated["seed"] == 10029
    assert evaluated["selection_nll"] == pytest.approx(training["best_validation_nll"],abs=1e-6)
    explicit = predictor.predict(query,1,samples=2,seed=43,allow_extrapolation=True)
    assert explicit["samples"] == 2 and explicit["seed"] == 43
