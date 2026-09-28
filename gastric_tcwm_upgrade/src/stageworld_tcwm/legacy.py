"""Bridge to the exact Generated651 raw cache contract; original encoders are reused.

No ad-hoc drug vocabulary, fitted transform from the full cohort, invented
follow-up date, or inferred surgery-to-follow-up interval is permitted.
"""
from pathlib import Path
import csv
import json
import torch
from .data import Cohort,split_indices

BASE_COMMIT="aec8cbe08157687fd447594c8a9406d6ee5a464d"


def original_encoders():
    try:
        from stageworld.data.baseline_clinical import encode_baseline,fit_clinical_transform,CT6_CLINICAL_SCHEMA
        from stageworld.data.treatment_compact import encode_compact,fit_name_support
    except ImportError as exc:
        raise RuntimeError("Legacy conversion requires the original repository installed (or its src on PYTHONPATH). Native training does not.") from exc
    return encode_baseline,fit_clinical_transform,CT6_CLINICAL_SCHEMA,encode_compact,fit_name_support


def convert_legacy(pool_path,event_path,split,out_path,acknowledge_retrospective=False):
    if not acknowledge_retrospective:
        raise ValueError("The original treatments are retrospective summaries; explicitly acknowledge this before conversion")
    raw=torch.load(pool_path,map_location="cpu",weights_only=True)
    event=torch.load(event_path,map_location="cpu",weights_only=True)
    ids=list(raw["ids"])
    if len(ids)!=651 or len(set(ids))!=651:
        raise ValueError("This bridge targets the audited complete651 cohort")
    event_ids=event["patient_ids"]
    if event.get("source_pool_id")!=raw["artifact_id"] or set(event_ids)!=set(ids) or len(set(event_ids))!=651:
        raise ValueError("Event/cache identity or provenance mismatch")
    lookup={p:i for i,p in enumerate(event_ids)}
    events=event["events"][[lookup[p] for p in ids]]
    if events.shape!=(651,3) or events.dtype!=torch.long or not (events[:,1]==1).all():
        raise ValueError("The original complete651 audit expects present surgery for every patient")
    if not raw["ct0_valid"].all() or not raw["ct1_valid"].all() or raw["ct0"].shape!=(651,27,768):
        raise ValueError("Original feature audit failed")
    all_ids=[p for role in split for p in split[role]]
    if set(split)!={"train","validation","test"} or len(all_ids)!=651 or len(set(all_ids))!=651 or set(all_ids)!=set(ids):
        raise ValueError("Provide a complete disjoint patient split before fitting original encoders")
    encode_baseline,fit_clinical_transform,schema,encode_compact,fit_name_support=original_encoders()
    clinical_transform=fit_clinical_transform(raw["clinical"],set(split["train"]),schema_version=schema)
    treatment_support=fit_name_support(list(raw["treatments"].values()),set(split["train"]))
    clinical=encode_baseline([raw["clinical"][p] for p in ids],clinical_transform).ridge_features().float()
    treatment,unseen=encode_compact([raw["treatments"][p] for p in ids],treatment_support)
    labels,valid=raw["labels"],raw["valid"]
    if labels.shape!=(651,2) or valid.shape!=(651,2):
        raise ValueError("Expected pCR and recorded recurrence labels")
    tensors={"ct0":raw["ct0"].float(),"ct1":raw["ct1_tokens"].float(),
             "image_valid":torch.stack((raw["ct0_valid"],raw["ct1_valid"]),1).bool(),
             "clinical":clinical,"treatment":treatment.float(),"interval_days":raw["interval"].float(),
             "surgery":events[:,1],"prefix_valid":torch.ones(651,3,dtype=torch.bool),
             "binary":labels[:,1].float(),"binary_valid":valid[:,1].bool(),
             "pcr":labels[:,0].float(),"pcr_valid":valid[:,0].bool(),
             "ct1_available_stage":torch.ones(651,dtype=torch.long),
             "unseen_treatment_names_count":torch.tensor([len(x) for x in unseen])}
    metadata={"treatment_semantics":"explicit_interval_scenario","source_commit":BASE_COMMIT,
              "endpoint_definition":"Original CG recorded recurrence/metastasis status; no fixed horizon or event time",
              "original_pool_id":raw["artifact_id"],"original_event_id":event["artifact_id"],
              "treatment_origin":"retrospective_interval_summary_not_baseline_known_fact",
              "landmark_claim":"retrospective information subsets, NOT verified dated prospective landmarks",
              "all_received_surgery":True,"synthetic":False}
    cohort=Cohort({"schema":"tcwm-cohort-v1","ids":ids,"tensors":tensors,"metadata":metadata,
                   "encoders":{"clinical":clinical_transform,"treatment_support":treatment_support,"fit_ids":split["train"]}})
    split_indices(cohort,split)
    cohort.save(out_path)
    return {"patients":len(cohort),"endpoint":"recorded_binary_status","survival_enabled":False,
            "prospective_baseline_claim":False,"all_received_surgery":True}


def encode_legacy_inputs(bundle_path,clinical_rows,treatment_rows):
    """Transform new raw rows with frozen original training-fold encoders."""
    bundle=torch.load(bundle_path,map_location="cpu",weights_only=True)
    enc=bundle.get("encoders",{})
    if not {"clinical","treatment_support"}<=set(enc):
        raise ValueError("This bundle does not contain original raw-input encoders")
    encode_baseline,_,_,encode_compact,_=original_encoders()
    clinical=encode_baseline(clinical_rows,enc["clinical"]).ridge_features().float()
    treatment,warnings=encode_compact(treatment_rows,enc["treatment_support"])
    return {"clinical":clinical,"treatment":treatment},warnings


def attach_survival(cohort_path,sidecar_path,out_path,time_origin,administrative_cutoff=None):
    """Attach clinician/data-manager verified follow-up; require exact patient coverage.

    CSV columns: patient_id,time_months,event,entry_s0_months,entry_s1_months,
    entry_s2_months,s0_valid,s1_valid,s2_valid. event=0/1/2. Valid flags are 0/1.
    Preoperative predictions targeting surgery-origin recurrence use entry=0,
    conditional on membership in that surgery cohort; never invent calendar dates.
    """
    cohort=Cohort.load(cohort_path)
    with open(sidecar_path,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if len({r["patient_id"] for r in rows})!=len(rows) or {r["patient_id"] for r in rows}!=set(cohort.ids):
        raise ValueError("Follow-up must cover exactly the same unique patients")
    lookup={r["patient_id"]:r for r in rows}
    ordered=[lookup[p] for p in cohort.ids]
    time=torch.tensor([float(r["time_months"]) for r in ordered])
    event=torch.tensor([int(r["event"]) for r in ordered],dtype=torch.long)
    entry=torch.tensor([[float(r[f"entry_s{k}_months"]) for k in range(3)] for r in ordered])
    valid_int=torch.tensor([[int(r[f"s{k}_valid"]) for k in range(3)] for r in ordered])
    if not ((valid_int==0)|(valid_int==1)).all():
        raise ValueError("Landmark validity must be explicitly 0/1")
    valid=valid_int.bool()
    if administrative_cutoff is not None:
        if administrative_cutoff<=0:
            raise ValueError("Administrative cutoff must be positive")
        event=torch.where(time>administrative_cutoff,torch.zeros_like(event),event)
        time=time.clamp_max(administrative_cutoff)
        valid &= time[:,None]>entry
    cohort.tensors.update({"time":time,"event":event,"entry":entry,"prefix_valid":valid})
    cohort.metadata.update({"time_unit":"months","time_origin":time_origin,
                            "endpoint_definition":"Verified first recurrence (cause1); competing death (cause2) when supplied",
                            "administrative_cutoff_months":administrative_cutoff})
    cohort.validate().save(out_path)
