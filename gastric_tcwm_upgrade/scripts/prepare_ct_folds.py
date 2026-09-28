#!/usr/bin/env python
"""Prepare three inner folds using only the existing 521 training patients."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"src"))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold,train_test_split
from prepare_local import normalize_split
from stageworld_tcwm.data import Cohort,file_sha256,fingerprint,split_indices,write_json
from stageworld_tcwm.legacy import convert_legacy,original_encoders


def make_internal_splits(original_split,ids,labels,valid,seed=20260928):
    flat = [patient for patients in original_split.values() for patient in patients]
    if (set(original_split)!={"train","validation","test"} or
            len(flat)!=len(set(flat)) or set(flat)!=set(ids)):
        raise ValueError("Original patient partitions must be complete and disjoint")
    if not all(original_split.values()):
        raise ValueError("Original patient partitions must be nonempty")
    training_ids = original_split["train"]
    lookup = {patient:index for index,patient in enumerate(ids)}
    rows = torch.tensor([lookup[patient] for patient in training_ids])
    # Only original training labels may determine the internal partitions.
    y = labels[rows]
    if not bool(valid[rows].all()) or not bool(((y==0)|(y==1)).all()):
        raise ValueError("Internal stratification requires valid binary training labels")
    values,counts = torch.unique(y,return_counts=True)
    if len(values)!=2 or int(counts.min())<3:
        raise ValueError("Three-fold stratification requires at least three of each class")
    splitter = StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
    excluded = original_split["validation"]+original_split["test"]
    folds = []
    for train_rows,validation_rows in splitter.split(np.zeros(len(rows)),y.numpy()):
        folds.append({"train":[training_ids[index] for index in train_rows],
                      "validation":[training_ids[index] for index in validation_rows],
                      "test":list(excluded)})
    selected = [patient for fold in folds for patient in fold["validation"]]
    if len(selected)!=len(set(selected)) or set(selected)!=set(training_ids):
        raise AssertionError("Every original training patient must be selected exactly once")
    return folds


def add_nested_selection(outer_split,ids,labels,valid,seed):
    lookup = {patient:index for index,patient in enumerate(ids)}
    outer_train = outer_split["train"]
    rows = torch.tensor([lookup[patient] for patient in outer_train])
    y = labels[rows]
    if not bool(valid[rows].all()) or not bool(((y==0)|(y==1)).all()):
        raise ValueError("Nested selection requires valid binary outer-training labels")
    train_rows,selection_rows = train_test_split(
        np.arange(len(outer_train)),test_size=.2,stratify=y.numpy(),random_state=seed)
    split = {"train":[outer_train[index] for index in train_rows],
             "validation":[outer_train[index] for index in selection_rows],
             "test":list(outer_split["validation"])+list(outer_split["test"])}
    if set(split["train"]+split["validation"])!=set(outer_train):
        raise AssertionError("Nested training and selection must exactly partition outer training")
    return split,list(outer_split["validation"])


def verify_fold_encoders(cohort,raw,split):
    fit_ids = set(split["train"])
    if (set(cohort.encoders.get("fit_ids",[]))!=fit_ids or
            len(cohort.encoders.get("fit_ids",[]))!=len(fit_ids)):
        raise ValueError("Converted encoders do not bind exactly this fold's training patients")
    encode_clinical,fit_clinical,schema,encode_treatment,fit_treatment = original_encoders()
    # Refit the audit reference from physically restricted inputs, independently
    # of the converter's complete-pool inputs and training-ID filter.
    clinical_rows = {patient:raw["clinical"][patient] for patient in split["train"]}
    treatment_rows = [raw["treatments"][patient] for patient in split["train"]]
    clinical = fit_clinical(clinical_rows,fit_ids,schema_version=schema)
    treatment = fit_treatment(treatment_rows,fit_ids)
    if clinical!=cohort.encoders["clinical"] or treatment!=cohort.encoders["treatment_support"]:
        raise ValueError("Fold encoders differ from a training-only reference refit")
    expected_clinical = encode_clinical([raw["clinical"][p] for p in cohort.ids],clinical).ridge_features().float()
    expected_treatment,_ = encode_treatment([raw["treatments"][p] for p in cohort.ids],treatment)
    torch.testing.assert_close(cohort.tensors["clinical"],expected_clinical,rtol=0,atol=0)
    torch.testing.assert_close(cohort.tensors["treatment"],expected_treatment.float(),rtol=0,atol=0)
    return {"fit_patients":len(fit_ids),"fit_ids_exactly_match_train":True,
            "clinical_matches_training_only_refit":True,"treatment_matches_training_only_refit":True,
            "encoded_features_match_frozen_fold_encoders":True,
            "clinical_transform_sha256":fingerprint(clinical),
            "treatment_support_sha256":fingerprint(treatment)}


def exclude_holdout_labels(cohort,split,excluded_ids=None):
    split_indices(cohort,split)
    excluded_ids = list(split["test"] if excluded_ids is None else excluded_ids)
    if len(excluded_ids)!=len(set(excluded_ids)) or not set(excluded_ids)<=set(split["test"]):
        raise ValueError("Only unique patients in the test role can be excluded")
    lookup = {patient:index for index,patient in enumerate(cohort.ids)}
    rows = torch.tensor([lookup[patient] for patient in excluded_ids],dtype=torch.long)
    for name in ("binary","pcr"):
        cohort.tensors[name] = cohort.tensors[name].clone()
        cohort.tensors[name][rows] = 0
        cohort.tensors[name+"_valid"] = cohort.tensors[name+"_valid"].clone()
        cohort.tensors[name+"_valid"][rows] = False
    cohort.tensors["prefix_valid"] = cohort.tensors["prefix_valid"].clone()
    cohort.tensors["prefix_valid"][rows] = False
    cohort.metadata.update({"excluded_test_role":set(excluded_ids)==set(split["test"]),
                            "excluded_indices":rows.tolist(),
                            "excluded_ids":excluded_ids,
                            "excluded_outcomes_redacted":True,
                            "excluded_scoring_permitted":False,
                            "selection_scope":"original_training_patients_only",
                            "permitted_scoring_roles":["validation"],
                            "test_role_semantics":"excluded_original_validation_and_test_not_for_scoring"})
    cohort.validate()


def prepare_folds(data_dir,out_dir,seed=20260928,nested_selection=False):
    data_dir,out_dir = Path(data_dir),Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use a new output directory; internal folds are immutable")
    prepared = json.loads((data_dir/"preparation.json").read_text())
    provenance = prepared["provenance"]
    sources = {name:Path(provenance["inputs"][name]["path"])
               for name in ("pool","events","existing_split")}
    for name,path in sources.items():
        if file_sha256(path)!=provenance["inputs"][name]["sha256"]:
            raise ValueError("An original source differs from the existing preparation contract")
    legacy_source = Path(provenance["legacy_source"])
    sys.path.insert(0,str(legacy_source.parent))
    sys.path.insert(0,str(legacy_source))
    encode_clinical,_,_,encode_treatment,_ = original_encoders()
    for encoder in (encode_clinical,encode_treatment):
        module_path = Path(sys.modules[encoder.__module__].__file__).resolve()
        if not module_path.is_relative_to(legacy_source.resolve()):
            raise ValueError("Loaded original encoder source differs from the requested legacy tree")
    source_hashes = {str(path.relative_to(legacy_source)):file_sha256(path)
                     for path in sorted((legacy_source/"stageworld").rglob("*.py"))}
    source_hashes["../research_release.py"] = file_sha256(legacy_source.parent/"research_release.py")
    if source_hashes!=provenance["legacy_source_sha256"]:
        raise ValueError("Original encoder source differs from the existing preparation contract")
    raw = torch.load(sources["pool"],map_location="cpu",weights_only=True)
    events = torch.load(sources["events"],map_location="cpu",weights_only=True)
    document = json.loads(sources["existing_split"].read_text())
    original = normalize_split(document,raw,events)
    if len(raw["ids"])!=651 or len(original["train"])!=521:
        raise ValueError("This preparation requires the original complete651 / train521 contract")
    folds = make_internal_splits(original,raw["ids"],raw["labels"][:,1],raw["valid"][:,1],seed)
    excluded_ids = original["validation"]+original["test"]
    out_dir.mkdir(parents=True,exist_ok=True)
    reports = []
    for index,outer_split in enumerate(folds):
        split,outer_ids = (add_nested_selection(outer_split,raw["ids"],raw["labels"][:,1],raw["valid"][:,1],seed+index)
                           if nested_selection else (outer_split,[]))
        folder = out_dir/f"fold-{index}"
        folder.mkdir()
        with tempfile.TemporaryDirectory(prefix=".convert-",dir=out_dir) as temporary:
            converted = Path(temporary)/"cohort.pt"
            convert_legacy(sources["pool"],sources["events"],split,converted,
                           acknowledge_retrospective=True)
            cohort = Cohort.load(converted)
        encoder_audit = verify_fold_encoders(cohort,raw,split)
        rows = split_indices(cohort,split)
        lookup = {patient:index for index,patient in enumerate(cohort.ids)}
        outer_rows = torch.tensor([lookup[patient] for patient in outer_ids],dtype=torch.long)
        eligible_rows = torch.cat((rows["train"],rows["validation"],outer_rows))
        for new,old in (("ct0","ct0"),("ct1","ct1_tokens"),("interval_days","interval")):
            torch.testing.assert_close(cohort.tensors[new],raw[old],rtol=0,atol=0)
        for name,column in (("pcr",0),("binary",1)):
            torch.testing.assert_close(cohort.tensors[name][eligible_rows],raw["labels"][eligible_rows,column],rtol=0,atol=0)
            if not torch.equal(cohort.tensors[name+"_valid"][eligible_rows],raw["valid"][eligible_rows,column]):
                raise ValueError("Original training-label validity changed during conversion")
        exclude_holdout_labels(cohort,split,excluded_ids)
        partition_counts = {"train":len(split["train"]),"selection":len(split["validation"]),
                            "outer_evaluation":len(outer_ids),"excluded":len(excluded_ids),
                            "test_role_total":len(split["test"])}
        cohort.metadata.update({"nested_selection":nested_selection,
                                "outer_evaluation_ids":outer_ids,
                                "outer_evaluation_indices":outer_rows.tolist(),
                                "partition_counts":partition_counts})
        if nested_selection:
            cohort.metadata.update({"selection_scope":"nested_selection_within_outer_training_only",
                                    "test_role_semantics":"outer_evaluation_plus_excluded_original_holdouts",
                                    "outer_evaluation_requires_explicit_id_filter":True})
        cohort.metadata["audited_upstream_commit"] = cohort.metadata.pop("source_commit")
        cohort.metadata["inner_fold"] = {"index":index,"folds":3,"seed":seed,
                                         "nested_selection":nested_selection,
                                         "selection_seed":seed+index if nested_selection else None,
                                         "selection_fraction_of_outer_training":.2 if nested_selection else None,
                                         "stratification":"recorded_recurrence_within_original_train",
                                         "original_train_membership_sha256":fingerprint(sorted(original["train"])),
                                         "original_split_sha256":file_sha256(sources["existing_split"]),
                                         "legacy_source":str(legacy_source),"legacy_source_sha256":source_hashes,
                                         "encoder_audit":encoder_audit}
        cohort.save(folder/"cohort.pt")
        write_json(split,folder/"split.json")
        partitions = {}
        reporting_rows = {role:rows[role] for role in ("train","validation")}
        if nested_selection:
            reporting_rows["outer_evaluation"] = outer_rows
        for role,partition_rows in reporting_rows.items():
            batch = cohort.batch(partition_rows)
            partitions[role] = {"patients":len(partition_rows),
                                "recurrence_positive":int(batch["binary"][batch["binary_valid"]].sum()),
                                "recurrence_missing":int((~batch["binary_valid"]).sum()),
                                "pcr_positive":int(batch["pcr"][batch["pcr_valid"]].sum()),
                                "pcr_missing":int((~batch["pcr_valid"]).sum()),
                                "ct0_present":int(batch["image_valid"][:,0].sum()),
                                "ct1_present":int(batch["image_valid"][:,1].sum())}
        report = {"status":"passed","fold":index,"seed":seed,"partitions":partitions,
                  "nested_selection":nested_selection,"partition_counts":partition_counts,
                  "excluded_patients":len(excluded_ids),"excluded_outcomes_redacted":True,
                  "excluded_prefixes_invalid":True,"excluded_scoring_permitted":False,
                  "encoder_audit":encoder_audit,"cohort_sha256":file_sha256(folder/"cohort.pt"),
                  "split_sha256":file_sha256(folder/"split.json")}
        write_json(report,folder/"preparation.json")
        reports.append(report)
    result = {"status":"passed","seed":seed,"folds":reports,"patients_in_selection":521,
              "nested_selection":nested_selection,
              "excluded_patients":130,"original_validation_or_test_scored":False,
              "all_original_training_patients_selected_exactly_once":not nested_selection,
              "all_original_training_patients_evaluated_outer_once":nested_selection,
              "sources":provenance["inputs"],"legacy_source":str(legacy_source),
              "legacy_source_sha256":source_hashes}
    write_json(result,out_dir/"preparation.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=20260928)
    parser.add_argument("--nested-selection",action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(4)
    result = prepare_folds(args.data_dir,args.out,args.seed,args.nested_selection)
    print(json.dumps({"status":result["status"],"patients_in_selection":521,
                      "excluded_patients":130,"folds":[{"fold":fold["fold"],
                      "partitions":fold["partitions"],"encoder_audit":fold["encoder_audit"]}
                      for fold in result["folds"]]},indent=2))


if __name__=="__main__":
    main()
