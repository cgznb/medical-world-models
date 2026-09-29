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


def optimizer_parameter_groups(model,train_cfg):
    """Expose the exact LR contract, with every trainable parameter present once."""
    parameters = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    groups,manifest = [],[]
    split = train_cfg.readout_learning_rate is not None
    for group_name in (("world","readout") if split else ("all",)):
        selected = [(name,p) for name,p in parameters if not split or
                    name.startswith(("outcome.","pcr_output.")) == (group_name == "readout")]
        if not selected:
            continue
        lr = train_cfg.readout_learning_rate if group_name == "readout" else train_cfg.learning_rate
        groups.append({"params":[p for _,p in selected],"lr":lr})
        manifest.append({"name":group_name,"learning_rate":lr,"parameter_names":[name for name,_ in selected],
                         "parameter_count":sum(p.numel() for _,p in selected)})
    identities = [id(p) for group in groups for p in group["params"]]
    if len(identities) != len(set(identities)) or set(identities) != {id(p) for _,p in parameters}:
        raise RuntimeError("Optimizer parameter groups contain duplicate or missing parameters")
    return groups,manifest


def loss_gradient_probe(terms,model):
    branches = {"prior":("prior_","transition."),"observation_attention":("observation_attention.",),
                "observation_gate":("observation_gate",),"decoder":("decoder.",),"outcome":("outcome.",)}
    named = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    result = {}
    for loss_name,value in terms.items():
        gradients = torch.autograd.grad(value,[p for _,p in named],allow_unused=True,retain_graph=True)
        result[loss_name] = {}
        for branch,prefixes in branches.items():
            values = [g for (name,_),g in zip(named,gradients) if name.startswith(prefixes) and g is not None]
            result[loss_name][branch] = {"present":bool(values),
                "norm":float(torch.stack([g.detach().float().square().sum() for g in values]).sum().sqrt()) if values else 0.}
    return result


def train(cohort_path,split,model_cfg,train_cfg,out_dir,resume=False,warmstart=None):
    model_cfg.validate(); train_cfg.validate()
    if model_cfg.architecture == "predictive_ct" and (train_cfg.ct_weight or train_cfg.kl_weight or train_cfg.flow_weight or train_cfg.prior_ct_weight):
        raise ValueError("Predictive CT uses prior_weight latent likelihood; disable legacy CT/KL/flow objectives")
    if train_cfg.prior_weight and (model_cfg.architecture != "predictive_ct" or not model_cfg.predictive_transition):
        raise ValueError("Direct prior supervision requires a learned predictive CT transition")
    if train_cfg.include_initial_baseline and not model_cfg.clinical_anchor:
        raise ValueError("Initial clinical baseline requires an explicitly clinical-anchored model")
    if train_cfg.observation_recon_weight and (model_cfg.architecture != "token_world" or model_cfg.observation_update != "residual"):
        raise ValueError("Observation reconstruction requires the residual token-world observation update")
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
        if fingerprint(old["contract"]) != old["id"]:
            raise ValueError("Existing run contract fingerprint mismatch")
        normalized = dict(old["contract"])
        normalized["model"] = asdict(ModelConfig(**normalized["model"]).validate())
        normalized["train"] = asdict(TrainConfig(**normalized["train"]).validate())
        if fingerprint(normalized)!=contract_id or not resume:
            raise ValueError("Existing run requires --resume and identical data/split/configuration")
        # Missing fields acquire explicit compatibility defaults; checkpoint IDs stay bound.
        contract_id = old["id"]
        if not (out_dir/"last.pt").exists():
            raise ValueError("Cannot resume a run without a recovery checkpoint last.pt")
    elif any((out_dir/name).exists() for name in ("last.pt","best.pt","inference.pt","history.json",
              "training_report.json","effective_config.json","optimizer_groups.json","baseline_candidates.json")):
        raise ValueError("A new run cannot overwrite training artifacts without an existing verified contract")
    if not existing.exists():
        write_json({"id":contract_id,"contract":contract},existing)
    write_json({"model":asdict(model_cfg),"train":asdict(train_cfg)},out_dir/"effective_config.json")
    seed_all(train_cfg.seed)
    device = torch.device(train_cfg.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = model_from_config(model_cfg).to(device)
    model.fit_statistics(cohort.batch(roles["train"],device))
    if model_cfg.clinical_anchor:
        model.fit_clinical_anchors(cohort.batch(roles["train"],device))
    copied = warmstart_spatial(model,warmstart,split["train"]) if warmstart else []
    groups,group_manifest = optimizer_parameter_groups(model,train_cfg)
    optimizer = torch.optim.AdamW(groups,lr=train_cfg.learning_rate,weight_decay=train_cfg.weight_decay)
    write_json(group_manifest,out_dir/"optimizer_groups.json")
    support = fit_support(cohort.batch(roles["train"]))
    start,best,stale,history = 0,None if fixed_budget else float("inf"),0,[]
    optimizer_steps,supervised_steps,resume_batch,patients_seen = 0,0,0,0
    baseline_candidates,gradient_probes,task_counts = [],[],{}
    head_start_recorded = False
    selected_kind = "neural"
    last_validation_supervised_steps = 0
    recovery = out_dir/"last.pt"
    if resume and recovery.exists():
        ckpt = torch.load(recovery,map_location="cpu",weights_only=True)
        if ckpt["contract_id"]!=contract_id:
            raise ValueError("Recovery contract mismatch")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start = ckpt.get("resume_epoch",ckpt["epoch"]+1)
        resume_batch = ckpt.get("resume_batch",0)
        best,stale,history = ckpt["best"],ckpt["stale"],ckpt["history"]
        optimizer_steps = int(ckpt.get("optimizer_steps",start*steps_per_epoch))
        supervised_steps = int(ckpt.get("supervised_steps",max(0,optimizer_steps-warmup_steps)))
        if (optimizer_steps<0 or not 0<=supervised_steps<=optimizer_steps or optimizer_steps>budget or
                not 0<=resume_batch<steps_per_epoch or not 0<=start<=train_cfg.epochs):
            raise ValueError("Recovery optimizer-step counters conflict with the training budget")
        if history and "optimizer_steps" in history[-1]:
            if (history[-1]["optimizer_steps"]!=optimizer_steps or
                    history[-1]["supervised_steps"]!=supervised_steps or
                    sum(row["epoch_optimizer_steps"] for row in history)!=optimizer_steps or
                    sum(row["epoch_supervised_steps"] for row in history)!=supervised_steps):
                raise ValueError("Recovery optimizer-step counters disagree with checkpoint history")
        baseline_candidates = ckpt.get("baseline_candidates",[])
        gradient_probes = ckpt.get("gradient_probes",[])
        task_counts = ckpt.get("task_counts",{})
        patients_seen = ckpt.get("patients_seen",sum(row.get("training_patients_seen",0) for row in history))
        head_start_recorded = ckpt.get("head_start_recorded",False)
        selected_kind = ckpt.get("selected_kind","neural")
        last_validation_supervised_steps = ckpt.get("last_validation_supervised_steps",supervised_steps)
        restore_rng(ckpt["rng"])

    def evaluate():
        rng,training = capture_rng(),model.training
        try:
            predictions,health = collect_predictions(model,cohort,roles["validation"],train_cfg.samples_eval,
                train_cfg.batch_size,train_cfg.seed+10000,mc_seed_policy=train_cfg.mc_seed_policy,
                mc_antithetic=train_cfg.mc_antithetic,return_diagnostics=True,free_nats=train_cfg.free_nats)
            value = evaluate_predictions(predictions,cohort,roles["validation"],model_cfg,roles["train"],
                                         stage_weights=train_cfg.stage_weights)
            value.update({"evaluated":True,"used_for_checkpoint_selection":not fixed_budget,"health":health})
            score = value["selection_nll"]
            if (score is None and not fixed_budget) or (score is not None and not np.isfinite(score)):
                raise ValueError("No finite validation endpoint likelihood; inspect label support")
            return value
        finally:
            restore_rng(rng)
            model.train(training)

    def payload(epoch,next_epoch,next_batch):
        return {"contract_id":contract_id,"model_config":asdict(model_cfg),"model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),"epoch":epoch,"best":best,"stale":stale,
                "history":history,"rng":capture_rng(),"fit_ids":split["train"],"support":support,
                "optimizer_steps":optimizer_steps,"supervised_steps":supervised_steps,
                "selection_policy":train_cfg.checkpoint_selection,"selected_kind":selected_kind,
                "resume_epoch":next_epoch,"resume_batch":next_batch,"patients_seen":patients_seen,
                "baseline_candidates":baseline_candidates,"head_start_recorded":head_start_recorded,
                "task_counts":task_counts,"gradient_probes":gradient_probes,
                "last_validation_supervised_steps":last_validation_supervised_steps}

    def save_candidate(value,path):
        atomic_save({k:v for k,v in value.items() if k not in ("optimizer_state","rng","history")},path)

    def baseline(name,epoch,next_epoch,next_batch):
        nonlocal best,selected_kind,head_start_recorded
        head = model.outcome.output[-1] if hasattr(model.outcome,"output") else model.outcome
        if any(torch.count_nonzero(p).item() for p in head.parameters()):
            raise ValueError("Clinical baseline candidate requires a zero neural outcome head")
        validation = evaluate()
        chosen = validation["selection_nll"] < best-train_cfg.min_delta
        if chosen:
            best,selected_kind = validation["selection_nll"],"clinical_baseline"
        head_start_recorded = name == "baseline_head_start" or warmup_steps == 0
        baseline_candidates.append({"name":name,"selected":chosen,"optimizer_steps":optimizer_steps,
                                    "supervised_steps":supervised_steps,"validation":validation})
        value = payload(epoch,next_epoch,next_batch)
        value["selected_kind"] = "clinical_baseline"
        save_candidate(value,out_dir/(name+".pt"))
        if chosen:
            save_candidate(value,out_dir/"best.pt")
        if name == "baseline_initial":
            atomic_save(payload(epoch,next_epoch,next_batch),recovery)
        write_json(baseline_candidates,out_dir/"baseline_candidates.json")

    if train_cfg.include_initial_baseline and not baseline_candidates:
        baseline("baseline_initial",-1,0,0)

    def at_budget():
        return optimizer_steps>=budget or (train_cfg.max_supervised_steps is not None and
                                          supervised_steps>=train_cfg.max_supervised_steps)

    for epoch in range(start,train_cfg.epochs):
        if (not fixed_budget and stale>=train_cfg.patience) or at_budget():
            break
        model.train()
        order_generator = torch.Generator().manual_seed(train_cfg.seed*100003+epoch)
        order = roles["train"][torch.randperm(len(roles["train"]),generator=order_generator)]
        training_metrics = []
        segment_start_steps,segment_start_supervised = optimizer_steps,supervised_steps
        for batch_index,rows in enumerate(order.split(train_cfg.batch_size)):
            if batch_index < resume_batch:
                continue
            if at_budget() or (not fixed_budget and stale>=train_cfg.patience):
                break
            warm,_ = warmup_state(train_cfg,epoch,optimizer_steps)
            if train_cfg.include_initial_baseline and not warm and not head_start_recorded:
                baseline("baseline_head_start",epoch,epoch,batch_index)
            batch = cohort.batch(rows,device)
            # Observation dropout changes legal inputs, while paired targets stay factual.
            if train_cfg.observation_dropout:
                drop = torch.rand(len(rows),device=device)<train_cfg.observation_dropout
                batch["ct1_available_stage"] = torch.where(drop,torch.full_like(batch["ct1_available_stage"],3),batch["ct1_available_stage"])
            optimizer.zero_grad(set_to_none=True)
            probe = train_cfg.gradient_probe_interval is not None and optimizer_steps % train_cfg.gradient_probe_interval == 0
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=train_cfg.amp and device.type=="cuda"):
                output = model(batch,train_cfg.samples_train,compute_aux=True,force_gaussian=warm)
                result = total_loss(output,batch,model,train_cfg,epoch,optimizer_step=optimizer_steps,return_terms=probe)
                loss,metrics = result[:2]
            if probe:
                gradient_probes.append({"optimizer_step":optimizer_steps+1,"losses":loss_gradient_probe(result[2],model)})
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),train_cfg.gradient_clip,error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            supervised_steps += int(bool(metrics["endpoint_active"] or metrics["pcr_active"]))
            patients_seen += len(rows)
            metrics["gradient_norm_before_clip"] = float(norm)
            training_metrics.append((len(rows),metrics))
            for key,enabled in metrics.items():
                if key.endswith("_active"):
                    task = key[:-7]
                    count = task_counts.setdefault(task,{"effective_updates":0,"effective_targets":0})
                    targets = metrics.get(task+"_targets",metrics.get("ct_targets",0) if task in ("ct","kl","flow","prior","prior_ct") else 0)
                    count["effective_updates"] += int(enabled)
                    count["effective_targets"] += int(targets)*int(enabled)
            epoch_complete = batch_index+1 == steps_per_epoch
            final = at_budget() or (epoch == train_cfg.epochs-1 and epoch_complete)
            interval_due = (train_cfg.validation_interval_steps is not None and
                            supervised_steps-last_validation_supervised_steps>=train_cfg.validation_interval_steps)
            should_evaluate = final if fixed_budget else final or interval_due or (train_cfg.validation_interval_steps is None and epoch_complete)
            if not (epoch_complete or final or should_evaluate):
                continue
            validation = evaluate() if should_evaluate else {"selection_nll":None,"evaluated":False,"used_for_checkpoint_selection":False}
            score = validation["selection_nll"]
            eligible = supervised_steps>last_validation_supervised_steps
            chosen = should_evaluate and eligible and (fixed_budget or score<best-train_cfg.min_delta)
            if fixed_budget:
                best,stale = None,0
            elif chosen:
                best,stale,selected_kind = score,0,"neural"
            elif should_evaluate and eligible:
                stale += 1
            if should_evaluate:
                last_validation_supervised_steps = supervised_steps
            averages = {k:sum(n*d[k] for n,d in training_metrics)/sum(n for n,_ in training_metrics)
                        for k in training_metrics[0][1]}
            history.append({"epoch":epoch,"epoch_complete":epoch_complete,"selected":chosen,
                "training":averages,"validation":validation,"optimizer_steps":optimizer_steps,"supervised_steps":supervised_steps,
                "epoch_optimizer_steps":optimizer_steps-segment_start_steps,
                "epoch_supervised_steps":supervised_steps-segment_start_supervised,
                "training_patients_seen":sum(n for n,_ in training_metrics),"cumulative_patients_seen":patients_seen,
                "task_counts":{k:dict(v) for k,v in task_counts.items()}})
            value = payload(epoch,epoch+1 if epoch_complete else epoch,0 if epoch_complete else batch_index+1)
            if chosen:
                save_candidate(value,out_dir/"best.pt")
            atomic_save(value,recovery)
            write_json(history,out_dir/"history.json")
            write_json(gradient_probes,out_dir/"gradient_probes.json")
            print(f"epoch={epoch} optimizer_steps={optimizer_steps} supervised_steps={supervised_steps} val_nll={score} selected={chosen}",flush=True)
            training_metrics = []
            segment_start_steps,segment_start_supervised = optimizer_steps,supervised_steps
        resume_batch = 0
    best_path = out_dir/"best.pt"
    if not best_path.exists():
        raise RuntimeError("No eligible baseline or supervised checkpoint selected")
    selected = torch.load(best_path,map_location="cpu",weights_only=True)
    if fixed_budget and (not at_budget() or selected["optimizer_steps"]!=optimizer_steps):
        raise RuntimeError("Fixed-budget export requires a selected checkpoint at the complete planned budget")
    bundle = {"schema":"tcwm-inference-v1","model_config":selected["model_config"],
              "model_state":selected["model_state"],"support":support,"metadata":cohort.metadata,
              "contract_id":contract_id,"selected_epoch":selected["epoch"],"locked_selection":True,
              "selection_policy":train_cfg.checkpoint_selection,
              "selected_kind":selected.get("selected_kind","neural"),
              "evaluation_config":{"stage_weights":train_cfg.stage_weights,"samples_eval":train_cfg.samples_eval,
                                   "mc_seed":train_cfg.seed+10000,
                                   "mc_seed_policy":train_cfg.mc_seed_policy,"mc_antithetic":train_cfg.mc_antithetic},
              "selected_optimizer_steps":selected.get("optimizer_steps",(selected["epoch"]+1)*steps_per_epoch),
              "selected_supervised_steps":selected.get("supervised_steps",max(0,(selected["epoch"]+1)*steps_per_epoch-warmup_steps)),
              "causal_effects_identified":False,"encoders":getattr(cohort,"encoders",{})}
    atomic_save(bundle,out_dir/"inference.pt")
    epochs_completed = sum(bool(row.get("epoch_complete",True)) for row in history)
    actual_epochs = max((row["epoch"]+1 for row in history),default=0)
    report = {"selected_epoch":selected["epoch"],"best_validation_nll":selected["best"],
              "parameter_count":sum(p.numel() for p in model.parameters()),"test_evaluated":False,
              "warmstart_copied_keys":copied,"contract_id":contract_id,"epochs_completed":epochs_completed,
              "optimizer_steps":optimizer_steps,"supervised_steps":supervised_steps,
              "selected_optimizer_steps":bundle["selected_optimizer_steps"],
              "selected_supervised_steps":bundle["selected_supervised_steps"],
              "steps_per_full_epoch":steps_per_epoch,"optimizer_step_budget":budget,
              "selection_policy":train_cfg.checkpoint_selection,
              "selected_kind":bundle["selected_kind"],"baseline0_won":bundle["selected_kind"]=="clinical_baseline",
              "baseline_candidates":baseline_candidates,"stage_weights":list(train_cfg.stage_weights),
              "samples_eval":train_cfg.samples_eval,"mc_seed_policy":train_cfg.mc_seed_policy,
              "mc_antithetic":train_cfg.mc_antithetic,"optimizer_groups":group_manifest,
              "task_counts":task_counts,"training_patients_seen":patients_seen,
              "train_patients":len(roles["train"]),
              "train_events":int(cohort.tensors["binary"][roles["train"]][cohort.tensors["binary_valid"][roles["train"]]].sum()) if model_cfg.endpoint=="binary" else int((cohort.tensors["event"][roles["train"]]>0).sum()),
              "planned_epochs":train_cfg.epochs,"planned_optimizer_steps":budget,
              "actual_epochs":actual_epochs,"actual_optimizer_steps":optimizer_steps,
              "budget_fulfilled":at_budget(),"supervised_step_budget":train_cfg.max_supervised_steps,
              "validation_evaluations":len(baseline_candidates)+sum(bool(row["validation"].get("evaluated",True)) for row in history),
              "final_validation_nll":next((row["validation"].get("selection_nll") for row in reversed(history) if row["validation"].get("evaluated",True)),None),
              "validation_used_for_checkpoint_selection":not fixed_budget,
              "early_stopping_unit":"validation_cycle" if train_cfg.validation_interval_steps else "epoch",
              "stop_reason":"early_stopping" if not fixed_budget and stale>=train_cfg.patience else
                            "supervised_step_limit" if train_cfg.max_supervised_steps is not None and supervised_steps>=train_cfg.max_supervised_steps else
                            "optimizer_step_limit" if max_steps is not None and optimizer_steps>=max_steps else "epoch_limit"}
    write_json(report,out_dir/"training_report.json")
    return report
