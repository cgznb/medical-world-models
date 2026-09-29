"""Factual-scenario supervision only. No outcome labels for altered treatments."""
import torch
from torch.nn import functional as F
from .belief import balanced_kl
from .survival import mixture_binary_nll,mixture_survival_nll


def feature_set_loss(prediction,target,valid,scale=None):
    """Generated651 set objective, normalized by TRAIN-only feature scales.

    64 fixed random projections; token permutation invariant. This is NOT
    anatomical registration, CT-pixel reconstruction, or a latent-set likelihood.
    """
    if not valid.any():
        return prediction.sum()*0.
    p,y = prediction[valid].float(),target[valid].detach().float()
    if scale is not None:
        p,y = p/scale,y/scale
    generator = torch.Generator().manual_seed(1729)
    directions = F.normalize(torch.randn(p.shape[-1],64,generator=generator),dim=0).to(p.device)
    pp,yy = (p@directions).sort(1).values,(y@directions).sort(1).values
    global_loss = F.smooth_l1_loss(p.mean(1),y.mean(1))+(1-F.cosine_similarity(p.mean(1),y.mean(1),dim=-1)).mean()
    spread = F.smooth_l1_loss(p.std(1,unbiased=False),y.std(1,unbiased=False))
    return global_loss+.25*(pp-yy).square().mean()+.1*spread


def patient_weighted_stage_mean(losses,valid,stage_weights=None):
    """Normalize eligible stage weights within patients, then average patients."""
    if losses.ndim != 2 or valid.shape != losses.shape:
        raise ValueError("Stage losses and eligibility must have matching (patient, stage) shapes")
    weights = losses.new_ones(losses.shape[1]) if stage_weights is None else torch.as_tensor(
        stage_weights,device=losses.device,dtype=losses.dtype)
    if weights.shape != (losses.shape[1],) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Stage weights must be finite, nonnegative and match the stage count")
    effective = valid.to(losses.dtype)*weights
    denominator = effective.sum(1)
    selected = denominator > 0
    return ((losses*effective).sum(1)[selected]/denominator[selected]).mean() if selected.any() else losses.sum()*0.


def endpoint_loss(output,batch,cfg,stage_weights=None):
    prediction = output["predictions"]
    b,stages = prediction.shape[:2]
    losses = prediction.new_zeros((b,stages))
    valid = batch["prefix_valid"][:,:stages].clone()
    if cfg.endpoint == "binary":
        valid &= batch["binary_valid"][:,None]
    else:
        if not all(k in batch for k in ("time","event","entry")):
            raise ValueError("Cannot train survival from binary recurrence labels")
    edges = torch.tensor(cfg.bin_edges,device=prediction.device,dtype=torch.float)
    for stage in range(stages):
        selected = valid[:,stage]
        if not selected.any():
            continue
        if cfg.endpoint == "binary":
            values = mixture_binary_nll(prediction[selected,stage],batch["binary"][selected])
        else:
            values = mixture_survival_nll(prediction[selected,stage],batch["time"][selected],
                         batch["event"][selected],batch["entry"][selected,stage],edges)
        losses[selected,stage] = values
    return patient_weighted_stage_mean(losses,valid,stage_weights) if valid.any() else prediction.sum()*0


def warmup_state(train_cfg,epoch=0,optimizer_step=None):
    steps = getattr(train_cfg,"warmup_optimizer_steps",None)
    if steps is not None:
        if optimizer_step is None:
            raise ValueError("Step-based warmup requires the completed optimizer-step count")
        return optimizer_step < steps,min(1.,(optimizer_step+1)/max(1,steps))
    epochs = train_cfg.warmup_epochs
    return epoch < epochs,min(1.,(epoch+1)/max(1,epochs))


def total_loss(output,batch,model,train_cfg,epoch=0,optimizer_step=None,return_terms=False):
    warm,anneal = warmup_state(train_cfg,epoch,optimizer_step)
    paired = batch["image_valid"].all(1)
    pcr_valid = batch["pcr_valid"]
    endpoint_valid = batch["prefix_valid"][:,:output["predictions"].shape[1]]
    if model.cfg.endpoint == "binary":
        endpoint_valid = endpoint_valid & batch["binary_valid"][:,None]
    weighted_endpoint_valid = endpoint_valid & (torch.as_tensor(
        train_cfg.stage_weights,device=endpoint_valid.device)[:endpoint_valid.shape[1]]>0)
    observed = paired & (batch["ct1_available_stage"] <= 1)
    active = {"endpoint":not warm and bool(weighted_endpoint_valid.any()),
              "pcr":not warm and train_cfg.pcr_weight>0 and bool(pcr_valid.any()),
              "ct":train_cfg.ct_weight>0 and bool(paired.any()),
              "kl":train_cfg.kl_weight>0 and bool(paired.any()),
              "flow":train_cfg.flow_weight>0 and "flow_loss" in output and bool(paired.any()),
              "prior":train_cfg.prior_weight>0 and "prior_nll" in output and bool(paired.any()),
              "prior_ct":train_cfg.prior_ct_weight>0 and "prior_features" in output and bool(paired.any()),
              "observation_recon":train_cfg.observation_recon_weight>0 and "updated_features" in output and bool(observed.any())}
    active["readout_l2"] = train_cfg.readout_l2>0 and (active["endpoint"] or active["pcr"])
    zero = output["predictions"].new_zeros(())
    grad_enabled = torch.is_grad_enabled()
    # Disabled terms remain observable metrics but have no autograd connection.
    # A zero multiplier would still give AdamW a gradient and decay the head.
    with torch.set_grad_enabled(grad_enabled and active["endpoint"]):
        endpoint = endpoint_loss(output,batch,model.cfg,train_cfg.stage_weights)
    with torch.set_grad_enabled(grad_enabled and active["ct"]):
        ct = feature_set_loss(output["features"],batch["ct1"],paired,model.image_scale)
    with torch.set_grad_enabled(grad_enabled and active["kl"]):
        values = balanced_kl(output["qmean"],output["qlogvar"],output["pmean"],output["plogvar"],
                             train_cfg.kl_balance,train_cfg.free_nats)
        kl = values[paired].mean() if paired.any() else zero
    with torch.set_grad_enabled(grad_enabled and active["pcr"]):
        pcr = mixture_binary_nll(output["pcr_logits"][pcr_valid],batch["pcr"][pcr_valid]).mean() if pcr_valid.any() else zero
    with torch.set_grad_enabled(grad_enabled and active["flow"]):
        values = output.get("flow_loss")
        flow = values[paired].mean() if values is not None and paired.any() else zero
    with torch.set_grad_enabled(grad_enabled and active["prior"]):
        values = output.get("prior_nll")
        prior = values[paired].mean() if values is not None and paired.any() else zero
    with torch.set_grad_enabled(grad_enabled and active["prior_ct"]):
        prior_ct = feature_set_loss(output["prior_features"],batch["ct1"],paired,model.image_scale) if "prior_features" in output else zero
    with torch.set_grad_enabled(grad_enabled and active["observation_recon"]):
        observation_recon = feature_set_loss(output["updated_features"],batch["ct1"],observed,model.image_scale) if "updated_features" in output else zero
    penalty = zero
    if active["readout_l2"]:
        prefixes = tuple(name for name, enabled in (("outcome.",active["endpoint"]),
                         ("pcr_output.",active["pcr"])) if enabled)
        penalty = sum((p.square().sum() for name,p in model.named_parameters()
                       if name.startswith(prefixes) and p.ndim > 1),zero)
    terms = {"endpoint":endpoint,"pcr":pcr,"ct":ct,"kl":kl,"flow":flow,
             "prior":prior,"prior_ct":prior_ct,"readout_l2":penalty,"observation_recon":observation_recon}
    weights = {"endpoint":1.,"pcr":train_cfg.pcr_weight,"ct":train_cfg.ct_weight,
               "kl":train_cfg.kl_weight*anneal,"flow":train_cfg.flow_weight,
               "prior":train_cfg.prior_weight,"prior_ct":train_cfg.prior_ct_weight,"readout_l2":train_cfg.readout_l2,
               "observation_recon":train_cfg.observation_recon_weight}
    if not any(active.values()):
        raise ValueError("No enabled training loss has observed targets in this batch")
    loss = sum((weights[name]*value for name,value in terms.items() if active[name]),zero)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite joint objective")
    metrics = {k:float(v.detach()) for k,v in {"loss":loss,"endpoint_nll":endpoint,"ct_set":ct,"kl":kl,
               "pcr_nll":pcr,"flow_mse":flow,"prior_nll":prior,"prior_ct_set":prior_ct,"readout_l2":penalty,
               "observation_recon_set":observation_recon}.items()}
    metrics.update({name+"_active":int(enabled) for name,enabled in active.items()})
    metrics.update({"endpoint_targets":int(weighted_endpoint_valid.any(1).sum()),
                    "pcr_targets":int(pcr_valid.sum()),"ct_targets":int(paired.sum()),
                    "observation_recon_targets":int(observed.sum())})
    return (loss,metrics,{name:weights[name]*value for name,value in terms.items() if active[name]}) if return_terms else (loss,metrics)
