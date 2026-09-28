"""Command-line interface; test evaluation is a separate, explicitly requested step."""
from pathlib import Path
import argparse
import json
import torch
from .config import load_config
from .data import Cohort,split_indices,write_json,fingerprint,file_sha256
from .synthetic import write_synthetic
from .training import train
from .inference import Predictor,jsonable
from .evaluation import collect_predictions,evaluate_predictions
from .legacy import convert_legacy,attach_survival


def main(argv=None):
    parser=argparse.ArgumentParser(description="TC-BWM research extension for Generated651")
    parser.add_argument("--threads",type=int,default=4)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("synth",help="Create synthetic engineering-test inputs, not clinical evidence")
    p.add_argument("--out",required=True);p.add_argument("--n",type=int,default=96)
    p.add_argument("--image-dim",type=int,default=32);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--survival",action="store_true");p.add_argument("--causes",type=int,default=1)
    p.add_argument("--postoperative-dim",type=int,default=0)
    p=sub.add_parser("train")
    for k in ("data","split","config","out"):
        p.add_argument(f"--{k}",required=True)
    p.add_argument("--resume",action="store_true");p.add_argument("--warmstart")
    p.add_argument("--device")
    p=sub.add_parser("evaluate")
    for k in ("data","split","run"):
        p.add_argument(f"--{k}",required=True)
    p.add_argument("--role",choices=("validation","test"),default="validation")
    p.add_argument("--samples",type=int,default=32);p.add_argument("--device",default="cpu")
    p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("predict")
    for k in ("bundle","query","out"):
        p.add_argument(f"--{k}",required=True)
    p.add_argument("--stage",type=int,choices=(0,1,2),required=True)
    p.add_argument("--horizons",type=float,nargs="+")
    p.add_argument("--samples",type=int,default=32);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--allow-extrapolation",action="store_true");p.add_argument("--device",default="cpu")
    p=sub.add_parser("legacy-convert")
    for k in ("pool","events","split","out"):
        p.add_argument(f"--{k}",required=True)
    p.add_argument("--acknowledge-retrospective",action="store_true")
    p=sub.add_parser("attach-survival")
    for k in ("data","sidecar","out","time-origin"):
        p.add_argument(f"--{k}",required=True)
    p.add_argument("--administrative-cutoff",type=float)
    args=parser.parse_args(argv)
    if args.threads<1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command=="synth":
        cohort=write_synthetic(args.out,n=args.n,image_dim=args.image_dim,seed=args.seed,survival=args.survival,
                               causes=args.causes,postoperative_dim=args.postoperative_dim)
        print(json.dumps({"patients":len(cohort),"synthetic":True,"clinical_evidence":False}))
    elif args.command=="train":
        m,t=load_config(args.config)
        if args.device:
            t.device=args.device
        result=train(args.data,json.loads(Path(args.split).read_text()),m,t,args.out,args.resume,args.warmstart)
        print(json.dumps(result,indent=2))
    elif args.command=="evaluate":
        root=Path(args.run)
        destination=root/f"evaluation_{args.role}.json"
        if destination.exists() and not args.overwrite:
            raise ValueError("Evaluation already exists; repeated scoring requires --overwrite")
        contract=json.loads((root/"contract.json").read_text())
        split=json.loads(Path(args.split).read_text())
        if contract["contract"]["cohort_sha256"]!=file_sha256(args.data) or contract["contract"]["split"]!=fingerprint(split):
            raise ValueError("Evaluation dataset/split differs from the locked training contract")
        predictor=Predictor(root/"inference.pt",args.device)
        cohort=Cohort.load(args.data);roles=split_indices(cohort,split)
        pred=collect_predictions(predictor.model,cohort,roles[args.role],samples=args.samples)
        report=evaluate_predictions(pred,cohort,roles[args.role],predictor.cfg,roles["train"])
        report.update({"role":args.role,"synthetic":cohort.metadata.get("synthetic",False),"external_validation":False})
        write_json(report,destination);print(json.dumps(report,indent=2))
    elif args.command=="predict":
        query=torch.load(args.query,map_location="cpu",weights_only=True)
        result=Predictor(args.bundle,args.device).predict(query,args.stage,samples=args.samples,seed=args.seed,
                      horizons=args.horizons,allow_extrapolation=args.allow_extrapolation)
        write_json(jsonable(result),args.out);print(json.dumps(jsonable(result),indent=2))
    elif args.command=="legacy-convert":
        result=convert_legacy(args.pool,args.events,json.loads(Path(args.split).read_text()),args.out,args.acknowledge_retrospective)
        print(json.dumps(result,indent=2))
    elif args.command=="attach-survival":
        attach_survival(args.data,args.sidecar,args.out,args.time_origin,args.administrative_cutoff)
        print("Verified survival sidecar attached")

if __name__=="__main__":
    main()
