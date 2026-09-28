"""Small synthetic data for engineering tests ONLY, not clinical validation."""
from pathlib import Path
import torch
from .data import Cohort,make_split,write_json,atomic_save


def treatment_template(code):
    action = torch.zeros(4,82)
    for g in range(4):
        action[g,78+g]=1
    action[0,0]=1; action[0,39]=1
    for j in range(5,8):
        action[1,39+j]=1
    action[1,5+code]=1
    return action


def synthetic_cohort(n=96,image_dim=32,seed=17,survival=False,causes=1,postoperative_dim=0):
    g=torch.Generator().manual_seed(seed)
    randn=lambda *shape:torch.randn(*shape,generator=g)
    clinical=randn(n,32)
    # Treatment is associated with baseline severity: toy confounding, not randomization.
    category=(torch.sigmoid(clinical[:,0])*2+torch.rand(n,generator=g)).floor().long().clamp_max(2)
    treatment=torch.stack([treatment_template(int(x)) for x in category])
    base=clinical[:,:6]+.25*randn(n,6)
    response=.6*base-.3*category[:,None]+.35*randn(n,6)
    projection=randn(6,image_dim)/6**.5
    spatial=randn(1,27,image_dim)*.2
    ct0=(base@projection)[:,None]+spatial+.1*randn(n,27,image_dim)
    ct1=(response@projection)[:,None]+spatial+.1*randn(n,27,image_dim)
    severity=response[:,0]+.25*clinical[:,1]
    recurrence_rate=torch.nn.functional.softplus(severity-.7)/25
    recurrence_time=-torch.log(torch.rand(n,generator=g).clamp_min(1e-6))/recurrence_rate
    death_time=-torch.log(torch.rand(n,generator=g).clamp_min(1e-6))/.007 if causes==2 else torch.full((n,),1e8)
    censor_time=12+48*torch.rand(n,generator=g)
    time=torch.minimum(torch.minimum(recurrence_time,death_time),censor_time)
    event=torch.where((recurrence_time<=death_time)&(recurrence_time<censor_time),1,
                      torch.where((death_time<recurrence_time)&(death_time<censor_time),2,0)).long()
    y=(event==1).float()
    tensors={"ct0":ct0,"ct1":ct1,"image_valid":torch.ones(n,2,dtype=torch.bool),
             "clinical":clinical,"treatment":treatment,"interval_days":60+90*torch.rand(n,generator=g),
             "surgery":torch.ones(n,dtype=torch.long),"prefix_valid":torch.ones(n,3,dtype=torch.bool),
             "binary":y,"binary_valid":torch.ones(n,dtype=torch.bool),
             "pcr":(response.mean(1)<-.3).float(),"pcr_valid":torch.ones(n,dtype=torch.bool),
             "ct1_available_stage":torch.ones(n,dtype=torch.long)}
    metadata={"treatment_semantics":"explicit_interval_scenario","endpoint_definition":"SYNTHETIC recorded recurrence status",
              "synthetic":True,"source":"engineering_test_generator","post_available_stage":2}
    if survival:
        tensors.update({"time":time,"event":event,"entry":torch.zeros(n,3)})
        metadata.update({"time_origin":"SYNTHETIC surgery reference","time_unit":"months",
                         "endpoint_definition":"SYNTHETIC first recurrence; cause2=competing death"})
    if postoperative_dim:
        tensors["post"]=severity[:,None,None]+.3*randn(n,4,postoperative_dim)
        tensors["post_mask"]=torch.ones(n,4,dtype=torch.bool)
    return Cohort({"schema":"tcwm-cohort-v1","ids":[f"SYNTHETIC-{i:04d}" for i in range(n)],
                   "tensors":tensors,"metadata":metadata})


def write_synthetic(root,**kwargs):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cohort=synthetic_cohort(**kwargs)
    cohort.save(root/"cohort.pt")
    split=make_split(cohort.ids,cohort.tensors["binary"].numpy(),kwargs.get("seed",17))
    write_json(split,root/"split.json")
    names={"clinical","treatment","interval_days","surgery","ct0","ct1","image_valid","ct1_available_stage","entry","post","post_mask"}
    q={k:v[:2] for k,v in cohort.tensors.items() if k in names}
    q["plan_available_stage"]=torch.zeros(2,dtype=torch.long)
    atomic_save({"schema":"tcwm-query-v1","plan_source":"hypothetical","interval_source":"specified_query","tensors":q},root/"query.pt")
    return cohort
