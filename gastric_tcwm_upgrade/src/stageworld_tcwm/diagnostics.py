"""Aggregate branch health and fixed-random-number dependency diagnostics."""
import copy

import torch

from .belief import diagonal_kl
from .monte_carlo import cohort_case_keys,evaluation_epsilon


def _patient_absolute_mean(value):
    return value.flatten(1).abs().mean(1)


def branch_health_values(output,batch,free_nats=.5):
    diagnostics = output["diagnostics"]
    values = {}
    predictions = output["predictions"]
    if predictions.ndim == 3:
        probability = predictions.sigmoid().mean(2)
        for stage in range(predictions.shape[1]):
            valid = batch["prefix_valid"][:,stage]
            values[f"S{stage}_probability"] = probability[valid,stage]
            residual = diagnostics["neural_residual"][:,stage].mean(1)
            values[f"S{stage}_neural_residual"] = residual[valid]
        paired = batch["prefix_valid"][:,:2].all(1)
        values["S1_S0_probability_absolute_difference"] = (probability[paired,1]-probability[paired,0]).abs()
    values["anchor_logit"] = diagnostics["anchor_logit"]
    observed = batch["image_valid"][:,1] & (batch["ct1_available_stage"] <= 1)
    pmean,plogvar = output["pmean"],output["plogvar"]
    qmean,qlogvar = output["qmean"][observed],output["qlogvar"][observed]
    values["prior_variance"] = plogvar.exp().flatten()
    values["posterior_variance"] = qlogvar.exp().flatten()
    values["prior_means_for_active_units"] = pmean
    values["posterior_means_for_active_units"] = qmean
    values["posterior_prior_mean_absolute_difference"] = (qmean-pmean[observed]).abs().mean(1)
    raw_kl = diagonal_kl(qmean,qlogvar,pmean[observed],plogvar[observed])
    values["raw_kl"] = raw_kl
    values["free_nats_kl"] = raw_kl.clamp_min(free_nats)
    deterministic = diagnostics["deterministic"][:,None]
    injected,updated,reference = (diagnostics[name] for name in ("injected","updated","reference"))
    values["injected_deterministic_absolute_difference"] = _patient_absolute_mean(injected-deterministic)
    values["updated_injected_absolute_difference"] = _patient_absolute_mean(updated-injected)[observed]
    values["reference_injected_absolute_difference"] = _patient_absolute_mean(reference[:,0]-injected)
    values["S1_S0_reference_absolute_difference"] = _patient_absolute_mean(reference[:,1]-reference[:,0])[observed]
    values["S1_S0_readout_absolute_difference"] = _patient_absolute_mean(
        diagnostics["readout"][:,1]-diagnostics["readout"][:,0])[observed]
    return {name:value.detach().float().cpu() for name,value in values.items()}


def distribution(value):
    value = value.flatten().double()
    if not value.numel():
        return {"n":0,"mean":None,"sd":None,"quantiles":None}
    return {"n":value.numel(),"mean":float(value.mean()),"sd":float(value.std(unbiased=False)),
            "quantiles":dict(zip(("p05","p50","p95"),value.quantile(torch.tensor([.05,.5,.95],dtype=torch.float64)).tolist()))}


def summarize_branch_health(chunks,active_threshold=.01):
    values = {name:torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]}
    report = {name:distribution(value) for name,value in values.items() if not name.endswith("_means_for_active_units")}
    report["active_units_definition"] = "variance_across_patient_means_above_threshold"
    report["active_units_variance_threshold"] = active_threshold
    for branch in ("prior","posterior"):
        means = values[branch+"_means_for_active_units"]
        report[branch+"_active_units"] = int((means.var(0,unbiased=False)>active_threshold).sum()) if len(means)>1 else None
        report[branch+"_active_units_patients"] = len(means)
    report["neural_residual_definition"] = "scaled_neural_logit_increment_patient_MC_mean"
    report["posterior_population"] = "patients_with_CT1_legally_available_at_S1"
    report["causal_effects_identified"] = False
    return report


def mc_stability_probe(model,cohort,rows,*,stage_weights=(1.,1.,1.),samples=(16,32,64),
                       seeds=(1729,2718,3141),batch_size=16,antithetic=False,min_delta=1e-4):
    from .evaluation import collect_predictions,evaluate_predictions
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("MC stability requires at least three distinct prespecified seeds")
    if min_delta <= 0:
        raise ValueError("min_delta must be positive")
    records = []
    for count in samples:
        scores = []
        for seed in seeds:
            pred = collect_predictions(model,cohort,rows,count,batch_size,seed,
                mc_seed_policy="case_key",mc_antithetic=antithetic)
            report = evaluate_predictions(pred,cohort,rows,model.cfg,stage_weights=stage_weights)
            if report["selection_nll"] is None:
                raise ValueError("MC stability requires valid weighted endpoint targets")
            scores.append(report["selection_nll"])
        sd = float(torch.tensor(scores,dtype=torch.float64).std(unbiased=True))
        records.append({"samples":count,"seeds":list(seeds),"selection_nll":scores,
                        "mc_nll_sd":sd,"sd_over_min_delta":sd/min_delta})
    at32 = next((row for row in records if row["samples"]==32),None)
    return {"schema":"tcwm-mc-stability-v1","patients":len(rows),"mc_seed_policy":"case_key",
            "mc_antithetic":antithetic,"stage_weights":list(stage_weights),"min_delta":min_delta,
            "records":records,"recommended_samples_eval":64 if at32 and at32["mc_nll_sd"]>=.5*min_delta else 32,
            "recommendation_rule":"K32_sd_at_least_half_min_delta_requires_relocking_at_K64",
            "seed_selection_permitted":False,"independent_clinical_validation":False}


@torch.inference_mode()
def dependency_probe(model,cohort,rows,*,samples=32,batch_size=16,seed=1729,
                     permutation_seed=2718,stage_weights=(1.,1.,1.),antithetic=False):
    from .evaluation import collect_predictions,evaluate_predictions
    if model.cfg.endpoint != "binary":
        raise ValueError("Dependency probability reports currently require a binary endpoint")
    rows = torch.as_tensor(rows,dtype=torch.long)
    predict = lambda data: collect_predictions(model,data,rows,samples,batch_size,seed,
        mc_seed_policy="case_key",mc_antithetic=antithetic)
    original = predict(cohort)
    original_report = evaluate_predictions(original,cohort,rows,model.cfg,stage_weights=stage_weights)
    before = original.sigmoid().mean(2)
    masks = cohort.tensors["image_valid"][rows]
    report = {"schema":"tcwm-dependency-v1","patients":len(rows),"samples":samples,"seed":seed,
              "permutation_seed":permutation_seed,"mc_seed_policy":"case_key","mc_antithetic":antithetic,
              "interpretation":"input_dependency_not_causal_effect_or_predictive_benefit",
              "causal_effects_identified":False,"perturbations":{}}
    scenarios = (("CT0",("ct0",),masks[:,0]),("CT1",("ct1",),masks[:,1]),
                 ("paired_CT",("ct0","ct1"),masks.all(1)),
                 ("known_scenario",("treatment","interval_days","surgery"),torch.ones(len(rows),dtype=torch.bool)))
    for name,fields,eligible in scenarios:
        selected = rows[eligible]
        if len(selected)<2:
            report["perturbations"][name] = {"supported":False,"eligible_patients":len(selected)}
            continue
        # A random cycle changes every eligible row while preserving joint paired fields.
        order = torch.randperm(len(selected),generator=torch.Generator().manual_seed(permutation_seed))
        donors = selected.clone()
        donors[order] = selected[order.roll(1)]
        modified = copy.copy(cohort)
        modified.tensors = dict(cohort.tensors)
        for field in fields:
            modified.tensors[field] = cohort.tensors[field].clone()
            modified.tensors[field][selected] = cohort.tensors[field][donors]
        altered = predict(modified)
        after = altered.sigmoid().mean(2)
        metrics = evaluate_predictions(altered,cohort,rows,model.cfg,stage_weights=stage_weights)
        stage_report = {}
        for stage in range(3):
            valid = cohort.tensors["prefix_valid"][rows,stage]
            difference = after[valid,stage]-before[valid,stage]
            stage_report[f"S{stage}"] = {"probability_difference":distribution(difference),
                "mean_absolute_difference":float(difference.abs().mean()) if len(difference) else None,
                "max_absolute_difference":float(difference.abs().max()) if len(difference) else None}
        report["perturbations"][name] = {"supported":True,"eligible_patients":len(selected),
            "stages":stage_report,"selection_nll":metrics["selection_nll"],
            "selection_nll_difference":metrics["selection_nll"]-original_report["selection_nll"]
                if metrics["selection_nll"] is not None and original_report["selection_nll"] is not None else None}
    report["prefix_boundary_checks"] = prefix_boundary_probe(model,cohort,rows,samples=samples,
        batch_size=batch_size,seed=seed,antithetic=antithetic)
    return report


@torch.inference_mode()
def prefix_boundary_probe(model,cohort,rows,*,samples=32,batch_size=16,seed=1729,antithetic=False):
    model.eval()
    device = next(model.parameters()).device
    maximum = {"S0_future_target_change":0.,"pCR_S0_future_target_change":0.,
               "S1_S2_without_new_information":0.,"undeclared_future_treatment_change":0.}
    checked_no_new = 0
    for start in range(0,len(rows),batch_size):
        selected = rows[start:start+batch_size]
        batch = cohort.batch(selected,device)
        epsilon = evaluation_epsilon(model,len(selected),samples,seed,policy="case_key",
            case_keys=cohort_case_keys(cohort,selected),antithetic=antithetic)
        before = model(batch,samples=samples,epsilon=epsilon,compute_aux=True)
        changed = dict(batch)
        changed["ct1"] = -batch["ct1"]+7.
        for key in ("binary","pcr"):
            changed[key] = 1-batch[key]
            changed[key+"_valid"] = ~batch[key+"_valid"]
        if "post" in changed:
            changed["post"] = -changed["post"]+11.
        if "time" in changed:
            changed["time"] = changed["time"]+100.
            changed["event"] = torch.zeros_like(changed["event"])
        after = model(changed,samples=samples,epsilon=epsilon,compute_aux=True)
        maximum["S0_future_target_change"] = max(maximum["S0_future_target_change"],float((before["predictions"][:,0]-after["predictions"][:,0]).abs().max()))
        maximum["pCR_S0_future_target_change"] = max(maximum["pCR_S0_future_target_change"],float((before["pcr_logits"]-after["pcr_logits"]).abs().max()))
        no_new = ~(batch["image_valid"][:,1] & (batch["ct1_available_stage"]==2))
        if "post_mask" in batch:
            no_new &= ~batch["post_mask"].any(1)
        if "entry" in batch:
            no_new &= batch["entry"][:,1]==batch["entry"][:,2]
        if no_new.any():
            checked_no_new += int(no_new.sum())
            delta = before["predictions"][no_new,1]-before["predictions"][no_new,2]
            maximum["S1_S2_without_new_information"] = max(maximum["S1_S2_without_new_information"],float(delta.abs().max()))
        changed = dict(batch)
        changed["future_actual_treatment_record"] = torch.full_like(batch["treatment"],100.)
        after = model(changed,samples=samples,epsilon=epsilon,compute_aux=False)
        maximum["undeclared_future_treatment_change"] = max(maximum["undeclared_future_treatment_change"],float((before["predictions"]-after["predictions"]).abs().max()))
    return {"max_absolute_logit_or_rate_difference":maximum,
            "no_new_information_patients":checked_no_new,
            "passed":all(value<=1e-6 for value in maximum.values())}
