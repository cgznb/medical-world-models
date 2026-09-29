#!/usr/bin/env python3
"""Fixed-checkpoint diagnostics on inner validation patients only."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from stageworld_tcwm.data import Cohort,file_sha256,fingerprint,split_indices,write_json
from stageworld_tcwm.diagnostics import dependency_probe,mc_stability_probe
from stageworld_tcwm.evaluation import collect_predictions
from stageworld_tcwm.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data","split","bundle","out"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--mode",choices=("mc","dependency","all"),default="all")
    parser.add_argument("--samples",type=int,nargs="+",default=(16,32,64))
    parser.add_argument("--seeds",type=int,nargs="+",default=(17,29,43))
    parser.add_argument("--stage-weights",type=float,nargs=3,default=(.5,.5,0.))
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--antithetic",action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Probe output must be new; preserve previous diagnostic evidence")
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("Require positive threads/batch size")
    os.umask(0o077)
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    cohort = Cohort.load(args.data)
    if not cohort.metadata.get("synthetic",False) and (
            cohort.metadata.get("nested_selection") is not True or
            cohort.metadata.get("excluded_scoring_permitted") is not False):
        raise ValueError("Real-data next-round probes require audited nested folds with original holdouts excluded")
    split = json.loads(args.split.read_text())
    rows = split_indices(cohort,split)
    predictor = Predictor(args.bundle,args.device)
    contract = json.loads((args.bundle.parent/"contract.json").read_text())
    if contract["id"] != predictor.bundle["contract_id"] or fingerprint(contract["contract"]) != contract["id"]:
        raise ValueError("Bundle does not match the recorded training contract")
    if contract["contract"]["cohort_sha256"] != file_sha256(args.data) or contract["contract"]["split"] != fingerprint(split):
        raise ValueError("Probe cohort/split does not match the fixed checkpoint")
    excluded = set(cohort.metadata.get("excluded_ids",[]))
    if excluded.intersection(split["validation"]):
        raise ValueError("Original excluded holdouts cannot enter next-round probes")
    report = {"schema":"tcwm-next-round-probe-v1","role":"inner_validation",
        "bundle_sha256":file_sha256(args.bundle),"contract_id":predictor.bundle["contract_id"],
        "patients":len(rows["validation"]),"original_test_or_validation_scored":False,
        "independent_clinical_validation":False,"stage_weights":list(args.stage_weights)}
    if args.mode in ("mc","all"):
        report["mc"] = mc_stability_probe(predictor.model,cohort,rows["validation"],
            samples=args.samples,seeds=args.seeds,stage_weights=args.stage_weights,
            batch_size=args.batch_size,antithetic=args.antithetic)
    if args.mode in ("dependency","all"):
        report["dependency"] = dependency_probe(predictor.model,cohort,rows["validation"],
            samples=max(args.samples),seed=args.seeds[0],stage_weights=args.stage_weights,
            batch_size=args.batch_size,antithetic=args.antithetic)
        _,report["branch_health"] = collect_predictions(predictor.model,cohort,rows["validation"],
            max(args.samples),args.batch_size,args.seeds[0],mc_seed_policy="case_key",
            mc_antithetic=args.antithetic,return_diagnostics=True)
    write_json(report,args.out)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
