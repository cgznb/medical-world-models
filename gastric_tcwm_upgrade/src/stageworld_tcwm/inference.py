"""Explicit, label-free, support-aware scenario inference; never a treatment optimizer."""
from dataclasses import fields
from pathlib import Path
import torch
from .config import ModelConfig
from .model import model_from_config
from .survival import competing_curves
from .support import support_flags
from .monte_carlo import evaluation_epsilon,monte_carlo_standard_error

ALLOWED = {"clinical","treatment","interval_days","surgery","ct0","ct1","image_valid",
           "ct1_available_stage","post","post_mask","entry","plan_available_stage"}

class Predictor:
    def __init__(self,bundle_path,device="cpu"):
        bundle = torch.load(bundle_path,map_location="cpu",weights_only=True)
        if bundle.get("schema")!="tcwm-inference-v1" or not bundle.get("locked_selection"):
            raise ValueError("Require a validation-selected inference bundle")
        self.bundle = bundle
        self.cfg = ModelConfig(**bundle["model_config"]).validate()
        self.device = torch.device(device)
        self.model = model_from_config(self.cfg).to(self.device).eval()
        self.model.load_state_dict(bundle["model_state"],strict=True)
        if not bool(self.model.statistics_fitted):
            raise ValueError("Missing fitted preprocessing")

    @torch.inference_mode()
    def predict(self,query,stage,*,samples=None,seed=None,horizons=None,allow_extrapolation=False,
                mc_seed_policy=None,mc_antithetic=None):
        evaluation = self.bundle.get("evaluation_config",{})
        samples = evaluation.get("samples_eval",32) if samples is None else samples
        seed = evaluation.get("mc_seed",17) if seed is None else seed
        if stage not in (0,1,2) or samples < 2:
            raise ValueError("stage is 0/1/2; at least two Monte Carlo samples are required")
        if query.get("schema")!="tcwm-query-v1":
            raise ValueError("Require tcwm-query-v1")
        source = query.get("plan_source")
        if source not in ("documented_plan","retrospective_factual","hypothetical"):
            raise ValueError("Declare documented_plan, retrospective_factual or hypothetical")
        if query.get("interval_source") not in ("specified_query","retrospective_actual"):
            raise ValueError("Declare whether the CT interval is a specified scenario or retrospective actual")
        if source=="documented_plan" and stage==0 and query["interval_source"]=="retrospective_actual":
            raise ValueError("An eventual actual CT interval is not a baseline-known fact")
        tensors = dict(query["tensors"])
        if set(tensors)-ALLOWED:
            raise ValueError(f"Inference rejects labels, outcomes and unknown fields: {set(tensors)-ALLOWED}")
        needed = {"clinical","treatment","interval_days","surgery","ct0","image_valid"}
        if not needed<=set(tensors):
            raise ValueError(f"Missing inference fields: {needed-set(tensors)}")
        b = len(tensors["clinical"])
        if not b:
            raise ValueError("Empty query")
        shapes = {"clinical":(b,32),"treatment":(b,4,82),"interval_days":(b,),
                  "surgery":(b,),"ct0":(b,27,self.cfg.image_dim),"image_valid":(b,2)}
        for name,shape in shapes.items():
            if tensors[name].shape!=shape:
                raise ValueError(f"{name} has wrong shape")
        if tensors["image_valid"].dtype!=torch.bool or tensors["surgery"].dtype!=torch.long:
            raise ValueError("Invalid mask/event dtypes")
        if not ((tensors["surgery"]>=0)&(tensors["surgery"]<=3)).all() or not (tensors["interval_days"]>0).all():
            raise ValueError("Invalid event/interval")
        if source=="documented_plan":
            known = tensors.get("plan_available_stage")
            if known is None or known.shape!=(b,) or (known<0).any() or (known>stage).any():
                raise ValueError("A documented plan must have been available at the requested stage")
        if stage==0:
            # Do not even inspect CT1/post values in an early prediction request.
            tensors.pop("post",None); tensors.pop("post_mask",None)
            tensors["ct1"] = torch.zeros_like(tensors["ct0"])
        elif "ct1" not in tensors:
            if tensors["image_valid"][:,1].any():
                raise ValueError("CT1 flagged available but not supplied")
            tensors["ct1"] = torch.zeros_like(tensors["ct0"])
        if tensors["ct1"].shape!=tensors["ct0"].shape:
            raise ValueError("CT1 must use the same observation feature encoder")
        if "ct1_available_stage" not in tensors:
            tensors["ct1_available_stage"] = torch.ones(b,dtype=torch.long)
        avail = tensors["ct1_available_stage"]
        if avail.shape!=(b,) or avail.dtype!=torch.long or not ((avail>=1)&(avail<=3)).all():
            raise ValueError("Invalid CT1 information-availability stage")
        if "post" in tensors:
            if stage<2:
                tensors.pop("post"); tensors.pop("post_mask",None)
            elif self.cfg.postoperative_dim==0:
                raise ValueError("This bundle was not trained with postoperative observations")
            elif tensors["post"].ndim!=3 or tensors["post"].shape[-1]!=self.cfg.postoperative_dim or tensors.get("post_mask",torch.empty(0)).shape!=tensors["post"].shape[:2] or tensors.get("post_mask",torch.empty(0)).dtype!=torch.bool:
                raise ValueError("Invalid postoperative observation contract")
        if self.cfg.endpoint=="binary" and horizons is not None:
            raise ValueError("Recorded binary status does not identify a 1/2-year survival curve")
        if self.cfg.endpoint=="survival":
            if horizons is None or "entry" not in tensors:
                raise ValueError("Survival needs explicit query horizons and known landmark entry")
            if tensors["entry"].shape==(b,):
                tensors["entry"] = tensors["entry"][:,None].repeat(1,3)
            if tensors["entry"].shape!=(b,3):
                raise ValueError("entry must be [B] or [B,3] months since endpoint origin")
        for k,v in tensors.items():
            if not isinstance(v,torch.Tensor) or not torch.isfinite(v).all():
                raise ValueError(f"Nonfinite/non-tensor input {k}")
        flags = support_flags(self.bundle["support"],tensors)
        if not allow_extrapolation and any(x["warnings"] for x in flags):
            raise ValueError(f"Unsupported/sparse scenario. Research-only override is explicit: {flags}")
        tensors = {k:v.to(self.device) for k,v in tensors.items()}
        mc_seed_policy = evaluation.get("mc_seed_policy","batch_start") if mc_seed_policy is None else mc_seed_policy
        mc_antithetic = evaluation.get("mc_antithetic",False) if mc_antithetic is None else mc_antithetic
        epsilon = evaluation_epsilon(self.model,b,samples,seed,policy=mc_seed_policy,
            case_keys=query.get("case_keys"),antithetic=mc_antithetic)
        kwargs = {"epsilon":epsilon} if epsilon is not None else {}
        output = self.model(tensors,samples=samples,seed=seed,max_stage=stage,compute_aux=False,**kwargs)["predictions"][:,stage]
        result = {"stage":stage,"plan_source":source,"interpretation":"scenario_conditioned_association",
                  "causal_effects_identified":False,"clinical_validation":False,
                  "support":flags,"samples":samples,"seed":seed,
                  "mc_seed_policy":mc_seed_policy,"mc_antithetic":mc_antithetic,
                  "mc_independent_units":samples//2 if mc_antithetic else samples,
                  "uncertainty_note":"latent spread and Monte Carlo error, not clinical confidence intervals"}
        if self.cfg.endpoint=="binary":
            probabilities = output.sigmoid()
            standard_error = monte_carlo_standard_error(probabilities,mc_antithetic)
            result.update({"recorded_status_probability":probabilities.mean(1).cpu(),
                           "latent_probability_std":probabilities.std(1,unbiased=False).cpu(),
                           "monte_carlo_standard_error":standard_error.cpu() if standard_error is not None else None})
        else:
            horizons = torch.as_tensor(horizons,dtype=torch.float,device=self.device)
            survival,cif = competing_curves(output,horizons,tensors["entry"][:,stage],self.model.outcome.edges)
            standard_error = monte_carlo_standard_error(cif[...,0],mc_antithetic)
            result.update({"horizons_months_since_origin":horizons.cpu(),"survival":survival.mean(1).cpu(),
                           "cumulative_incidence":cif.mean(1).cpu(),
                           "recurrence_probability":cif.mean(1)[...,0].cpu(),
                           "latent_recurrence_std":cif[...,0].std(1,unbiased=False).cpu(),
                           "monte_carlo_standard_error":standard_error.cpu() if standard_error is not None else None,
                           "time_origin":self.bundle["metadata"]["time_origin"]})
        return result


def ensemble_predict(paths,query,stage,**kwargs):
    if len(paths)<2:
        raise ValueError("An ensemble needs at least two independently trained bundles")
    predictions = [Predictor(path).predict(query,stage,**kwargs) for path in paths]
    key = "recurrence_probability" if "recurrence_probability" in predictions[0] else "recorded_status_probability"
    if any(p.get("time_origin")!=predictions[0].get("time_origin") or key not in p for p in predictions):
        raise ValueError("Ensemble endpoint/time-origin mismatch")
    values = torch.stack([p[key] for p in predictions])
    return {"probability":values.mean(0),"between_model_std":values.std(0,unbiased=False),
            "interpretation":"unvalidated ensemble spread, not a calibrated interval","members":len(paths)}


def jsonable(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,dict):
        return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    return value
