"""Patient-separated training with validation selection or a fixed update budget."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .config import ModelConfig,TrainConfig
from .data import Cohort,atomic_save,write_json,fingerprint,file_sha256,split_indices
from .model import model_from_config
from .losses import total_loss,warmup_state
from .evaluation import collect_predictions,evaluate_predictions
from .support import fit_support


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.mha.set_fastpath_enabled(False)


def capture_rng():
    return {"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def warmstart_spatial(model,checkpoint,training_ids):
    """Copy only compatible original spatial weights AFTER checking original fit IDs.

    An inference.pt without audited fit IDs is intentionally rejected.
    No optimizer, normalizer, endpoint head or patient split is imported.
    """
    payload = torch.load(checkpoint,map_location="cpu",weights_only=True)
    bound = payload.get("contract",{}).get("train_ids")
    if bound is None or set(bound)!=set(training_ids):
        raise ValueError("Original checkpoint training patients cannot be verified as identical")
    source = payload["model_state"]
    target = model.state_dict()
    accepted = {}
    for key,value in source.items():
        name = key[len("world."):] if key.startswith("world.") else key
        if (name.startswith("blocks.") or name.startswith("surgery.blocks.")) and name in target and target[name].shape==value.shape:
            accepted[name]=value
    if not accepted:
        raise ValueError("No compatible spatial weights; use H=128 and original depths")
    target.update(accepted)
    model.load_state_dict(target)
    return sorted(accepted)


def train(cohort_path,split,model_cfg,train_cfg,out_dir,resume=False,warmstart=None):
    model_cfg.validate(); train_cfg.validate()
    if model_cfg.architecture == "predictive_ct" and (train_cfg.ct_weight or train_cfg.kl_weight or train_cfg.flow_weight or train_cfg.prior_ct_weight):
        raise ValueError("Predictive CT uses prior_weight latent likelihood; disable legacy CT/KL/flow objectives")
    if train_cfg.prior_weight and (model_cfg.architecture != "predictive_ct" or not model_cfg.predictive_transition):
        raise ValueError("Direct prior supervision requires a learned predictive CT transition")
    cohort = Cohort.load(cohort_path)
    roles = split_indices(cohort,split)
    if cohort.encoders.get("fit_ids") is not None and set(cohort.encoders["fit_ids"]) != set(split["train"]):
        raise ValueError("Legacy encoders were fitted on a different training fold; reconvert the raw cache")
    if cohort.tensors["ct0"].shape[-1] != model_cfg.image_dim:
        raise ValueError("image_dim must match the feature cache")
    if model_cfg.endpoint=="survival":
        if "time" not in cohort.tensors:
            raise ValueError("Survival training requires a verified follow-up sidecar")
        if int(cohort.tensors["event"].max()) > model_cfg.causes:
            raise ValueError("Competing death cannot be silently recoded as censoring")
        if float(cohort.tensors["time"].max()) > model_cfg.bin_edges[-1]:
            raise ValueError("Follow-up exceeds bins; explicitly administratively censor first or extend bins")
    if model_cfg.postoperative_dim and ("post" not in cohort.tensors or cohort.tensors["post"].shape[-1]!=model_cfg.postoperative_dim):
        raise ValueError("Postoperative module requires real matching observation features")
    steps_per_epoch = (len(roles["train"])+train_cfg.batch_size-1)//train_cfg.batch_size
    max_steps = getattr(train_cfg,"max_optimizer_steps",None)
    warmup_steps = getattr(train_cfg,"warmup_optimizer_steps",None)
    warmup_steps = train_cfg.warmup_epochs*steps_per_epoch if warmup_steps is None else warmup_steps
    budget = min(train_cfg.epochs*steps_per_epoch,max_steps) if max_steps is not None else train_cfg.epochs*steps_per_epoch
    fixed_budget = train_cfg.checkpoint_selection == "fixed_budget"
    if budget<=warmup_steps:
        raise ValueError("Training budget must include at least one post-warmup optimizer step")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    contract = {"cohort_sha256":file_sha256(cohort_path),"split":fingerprint(split),
                "model":asdict(model_cfg),"train":asdict(train_cfg),
                "warmstart_sha256":file_sha256(warmstart) if warmstart else None}
    contract_id = fingerprint(contract)
    existing = out_dir/"contract.json"
    if existing.exists():
        import json
        old = json.loads(existing.read_text())
        if old["id"]!=contract_id or not resume:
            raise ValueError("Existing run requires --resume and identical data/split/configuration")
    write_json({"id":contract_id,"contract":contract},existing)
    seed_all(train_cfg.seed)
    device = torch.device(train_cfg.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = model_from_config(model_cfg).to(device)
    model.fit_statistics(cohort.batch(roles["train"],device))
    if model_cfg.clinical_anchor:
        model.fit_clinical_anchors(cohort.batch(roles["train"],device))
    copied = warmstart_spatial(model,warmstart,split["train"]) if warmstart else []
    parameters = list(model.named_parameters())
    if train_cfg.readout_learning_rate is not None:
        head = lambda name: name.startswith(("outcome.", "pcr_output."))
        groups = [{"params": [p for name,p in parameters if not head(name)]},
                  {"params": [p for name,p in parameters if head(name)],
                   "lr": train_cfg.readout_learning_rate}]
    else:
        groups = [p for _,p in parameters]
    optimizer = torch.optim.AdamW(groups,lr=train_cfg.learning_rate,weight_decay=train_cfg.weight_decay)
    support = fit_support(cohort.batch(roles["train"]))
    start,best,stale,history = 0,None if fixed_budget else float("inf"),0,[]
    optimizer_steps,supervised_steps = 0,0
    recovery = out_dir/"last.pt"
    if resume and recovery.exists():
        ckpt = torch.load(recovery,map_location="cpu",weights_only=True)
        if ckpt["contract_id"]!=contract_id:
            raise ValueError("Recovery contract mismatch")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start,best,stale,history = ckpt["epoch"]+1,ckpt["best"],ckpt["stale"],ckpt["history"]
        optimizer_steps = int(ckpt.get("optimizer_steps",start*steps_per_epoch))
        supervised_steps = int(ckpt.get("supervised_steps",max(0,optimizer_steps-warmup_steps)))
        if optimizer_steps<0 or not 0<=supervised_steps<=optimizer_steps or optimizer_steps>budget:
            raise ValueError("Recovery optimizer-step counters conflict with the training budget")
        if history and "optimizer_steps" in history[-1]:
            if (history[-1]["optimizer_steps"]!=optimizer_steps or
                    history[-1]["supervised_steps"]!=supervised_steps or
                    sum(row["epoch_optimizer_steps"] for row in history)!=optimizer_steps or
                    sum(row["epoch_supervised_steps"] for row in history)!=supervised_steps):
                raise ValueError("Recovery optimizer-step counters disagree with checkpoint history")
        restore_rng(ckpt["rng"])
    for epoch in range(start,train_cfg.epochs):
        if (not fixed_budget and stale>=train_cfg.patience) or optimizer_steps>=budget:
            break
        model.train()
        order_generator = torch.Generator().manual_seed(train_cfg.seed*100003+epoch)
        order = roles["train"][torch.randperm(len(roles["train"]),generator=order_generator)]
        training_metrics = []
        epoch_start_steps,epoch_start_supervised = optimizer_steps,supervised_steps
        for rows in order.split(train_cfg.batch_size):
            if optimizer_steps>=budget:
                break
            batch = cohort.batch(rows,device)
            # Training-only CT1 observation dropout; the target remains available.
            if train_cfg.observation_dropout:
                drop = torch.rand(len(rows),device=device)<train_cfg.observation_dropout
                batch["ct1_available_stage"] = torch.where(drop,torch.full_like(batch["ct1_available_stage"],3),batch["ct1_available_stage"])
            optimizer.zero_grad(set_to_none=True)
            warm,_ = warmup_state(train_cfg,epoch,optimizer_steps)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=train_cfg.amp and device.type=="cuda"):
                output = model(batch,train_cfg.samples_train,compute_aux=True,force_gaussian=warm)
                loss,metrics = total_loss(output,batch,model,train_cfg,epoch,optimizer_step=optimizer_steps)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),train_cfg.gradient_clip,error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            supervised_steps += int(bool(metrics["endpoint_active"] or metrics["pcr_active"]))
            metrics["gradient_norm_before_clip"] = float(norm)
            training_metrics.append((len(rows),metrics))
        if not fixed_budget or optimizer_steps>=budget:
            predictions = collect_predictions(model,cohort,roles["validation"],train_cfg.samples_eval,
                                train_cfg.batch_size,train_cfg.seed+10000)
            validation = evaluate_predictions(predictions,cohort,roles["validation"],model_cfg,roles["train"])
            validation.update({"evaluated":True,"used_for_checkpoint_selection":not fixed_budget})
        else:
            validation = {"selection_nll":None,"evaluated":False,"used_for_checkpoint_selection":False}
        score = validation["selection_nll"]
        if (score is None and not fixed_budget) or (score is not None and not np.isfinite(score)):
            raise ValueError("No finite validation endpoint likelihood; inspect label support")
        supervised_epoch = supervised_steps>epoch_start_supervised
        selected = supervised_epoch and (fixed_budget or score<best-train_cfg.min_delta)
        if fixed_budget:
            best,stale = None,0
        elif selected:
            best,stale=score,0
        elif supervised_epoch:
            stale += 1
        averages = {k:sum(n*d[k] for n,d in training_metrics)/sum(n for n,_ in training_metrics)
                    for k in training_metrics[0][1]}
        history.append({"epoch":epoch,"selected":selected,"training":averages,"validation":validation,
                        "optimizer_steps":optimizer_steps,"supervised_steps":supervised_steps,
                        "epoch_optimizer_steps":optimizer_steps-epoch_start_steps,
                        "epoch_supervised_steps":supervised_steps-epoch_start_supervised,
                        "training_patients_seen":sum(n for n,_ in training_metrics)})
        payload = {"contract_id":contract_id,"model_config":asdict(model_cfg),"model_state":model.state_dict(),
                   "optimizer_state":optimizer.state_dict(),"epoch":epoch,"best":best,"stale":stale,
                   "history":history,"rng":capture_rng(),"fit_ids":split["train"],"support":support,
                   "optimizer_steps":optimizer_steps,"supervised_steps":supervised_steps,
                   "selection_policy":train_cfg.checkpoint_selection}
        if selected:
            atomic_save({k:v for k,v in payload.items() if k not in ("optimizer_state","rng","history")},out_dir/"best.pt")
        atomic_save(payload,recovery)
        write_json(history,out_dir/"history.json")
        score_text = "not_evaluated" if score is None else f"{score:.6f}"
        print(f"epoch={epoch} optimizer_steps={optimizer_steps} supervised_steps={supervised_steps} val_nll={score_text} selected={selected}",flush=True)
    best_path = out_dir/"best.pt"
    if not best_path.exists():
        raise RuntimeError("No post-warmup checkpoint selected")
    selected = torch.load(best_path,map_location="cpu",weights_only=True)
    if fixed_budget and (optimizer_steps!=budget or selected["optimizer_steps"]!=budget):
        raise RuntimeError("Fixed-budget export requires a selected checkpoint at the complete planned budget")
    bundle = {"schema":"tcwm-inference-v1","model_config":selected["model_config"],
              "model_state":selected["model_state"],"support":support,"metadata":cohort.metadata,
              "contract_id":contract_id,"selected_epoch":selected["epoch"],"locked_selection":True,
              "selection_policy":train_cfg.checkpoint_selection,
              "selected_optimizer_steps":selected.get("optimizer_steps",(selected["epoch"]+1)*steps_per_epoch),
              "selected_supervised_steps":selected.get("supervised_steps",max(0,(selected["epoch"]+1)*steps_per_epoch-warmup_steps)),
              "causal_effects_identified":False,"encoders":getattr(cohort,"encoders",{})}
    atomic_save(bundle,out_dir/"inference.pt")
    report = {"selected_epoch":selected["epoch"],"best_validation_nll":selected["best"],
              "parameter_count":sum(p.numel() for p in model.parameters()),"test_evaluated":False,
              "warmstart_copied_keys":copied,"contract_id":contract_id,"epochs_completed":len(history),
              "optimizer_steps":optimizer_steps,"supervised_steps":supervised_steps,
              "selected_optimizer_steps":bundle["selected_optimizer_steps"],
              "selected_supervised_steps":bundle["selected_supervised_steps"],
              "steps_per_full_epoch":steps_per_epoch,"optimizer_step_budget":budget,
              "selection_policy":train_cfg.checkpoint_selection,
              "planned_epochs":train_cfg.epochs,"planned_optimizer_steps":budget,
              "actual_epochs":len(history),"actual_optimizer_steps":optimizer_steps,
              "budget_fulfilled":optimizer_steps==budget,
              "validation_evaluations":sum(bool(row["validation"].get("evaluated",True)) for row in history),
              "final_validation_nll":history[-1]["validation"].get("selection_nll"),
              "validation_used_for_checkpoint_selection":not fixed_budget,
              "early_stopping_unit":"epoch",
              "stop_reason":"early_stopping" if not fixed_budget and stale>=train_cfg.patience else
                            "optimizer_step_limit" if max_steps is not None and optimizer_steps>=max_steps else "epoch_limit"}
    write_json(report,out_dir/"training_report.json")
    return report
