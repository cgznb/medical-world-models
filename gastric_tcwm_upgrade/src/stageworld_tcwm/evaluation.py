"""Probability evaluation, with train-only censoring estimates for survival.

IPCW here assumes independent censoring within each evaluated landmark cohort.
No claim is made that unconditional KM resolves covariate-dependent censoring.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from .survival import mixture_binary_nll,mixture_survival_nll,competing_curves


def binary_metrics(y,p):
    y,p = np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if not len(y):
        return {"n":0}
    p = np.clip(p,1e-7,1-1e-7)
    result = {"n":len(y),"positive":int(y.sum()),"brier":float(np.mean((p-y)**2)),
              "nll":float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
              "auroc":None,"average_precision":None}
    if len(np.unique(y))==2:
        result["auroc"] = float(roc_auc_score(y,p))
        result["average_precision"] = float(average_precision_score(y,p))
    return result


class CensoringKM:
    def __init__(self,time,event,entry=None):
        time,event = np.asarray(time),np.asarray(event)
        entry = np.zeros_like(time) if entry is None else np.asarray(entry)
        self.times = np.unique(time[event==0])
        values = []
        g = 1.
        for t in self.times:
            at_risk = ((entry <= t)&(time >= t)).sum()
            # Resolve ties with observed failure before administrative censoring.
            failures = ((time == t)&(event != 0)).sum()
            censored = ((time == t)&(event == 0)).sum()
            denominator = at_risk-failures
            if denominator:
                g *= max(0.,1-censored/denominator)
            values.append(g)
        self.values = np.asarray(values)
    def at(self,t,left=False):
        t = np.asarray(t)
        idx = np.searchsorted(self.times,t,side="left" if left else "right")-1
        result = np.ones_like(t,dtype=float)
        valid = idx >= 0
        if self.values.size:
            result[valid] = self.values[idx[valid]]
        return result


def ipcw_metrics(time,event,entry,p,horizon,km,min_g=.05):
    time,event,entry,p = [np.asarray(x) for x in (time,event,entry,p)]
    eligible = entry <= horizon
    time,event,entry,p = [x[eligible] for x in (time,event,entry,p)]
    if not len(time):
        return {"n":0,"supported":False}
    g_entry = km.at(entry)
    g_horizon = km.at(np.full(len(time),horizon))
    g_event = km.at(time,left=True)
    before = (time <= horizon)&(event != 0)
    after = time > horizon
    if (g_entry <= 0).any() or (g_horizon[after] < min_g).any() or (g_event[before] < min_g).any():
        return {"n":len(time),"supported":False,"reason":"insufficient_censoring_support"}
    weight = np.zeros(len(time))
    weight[before] = g_entry[before]/g_event[before]
    weight[after] = g_entry[after]/g_horizon[after]
    y = ((event==1)&(time<=horizon)).astype(float)
    brier = float(np.mean(weight*(y-p)**2))
    known = weight>0
    auc = None
    if len(np.unique(y[known])) == 2:
        auc = float(roc_auc_score(y[known],p[known],sample_weight=weight[known]))
    calibration = []
    groups = np.array_split(np.argsort(p),min(5,len(p)))
    for indices in groups:
        if len(indices):
            calibration.append({"n":len(indices),"predicted":float(p[indices].mean()),
                   "observed_ipcw":float((weight[indices]*y[indices]).mean())})
    return {"n":len(time),"supported":True,"ipcw_brier":brier,"ipcw_auc":auc,"calibration":calibration}


@torch.inference_mode()
def collect_predictions(model,cohort,indices,samples=16,batch_size=16,seed=17):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for start in range(0,len(indices),batch_size):
        rows = indices[start:start+batch_size]
        batch = cohort.batch(rows,device)
        value = model(batch,samples=samples,seed=seed+start,compute_aux=False)["predictions"]
        predictions.append(value.cpu())
    return torch.cat(predictions)


def evaluate_predictions(pred,cohort,rows,cfg,train_rows=None,horizons=(12.,24.,36.)):
    batch = cohort.batch(rows)
    report = {"interpretation":"scenario_conditioned_association","causal_effects_identified":False,"stages":{}}
    all_nll = []
    edges = torch.tensor(cfg.bin_edges)
    for stage in range(3):
        valid = batch["prefix_valid"][:,stage].clone()
        if cfg.endpoint == "binary":
            valid &= batch["binary_valid"]
            p = pred[valid,stage].sigmoid().mean(1)
            stats = binary_metrics(batch["binary"][valid].numpy(),p.numpy())
            nll = mixture_binary_nll(pred[valid,stage],batch["binary"][valid]) if valid.any() else torch.empty(0)
        else:
            rates = pred[valid,stage]
            t,e,l = batch["time"][valid],batch["event"][valid],batch["entry"][valid,stage]
            stats = {"n":int(valid.sum()),"events":int((e>0).sum()),"recurrences":int((e==1).sum()),"horizons":{}}
            nll = mixture_survival_nll(rates,t,e,l,edges) if valid.any() else torch.empty(0)
            if train_rows is not None and valid.any():
                tb = cohort.batch(train_rows)
                tv = tb["prefix_valid"][:,stage]
                km = CensoringKM(tb["time"][tv].numpy(),tb["event"][tv].numpy(),tb["entry"][tv,stage].numpy())
                for h in horizons:
                    eligible = l <= h
                    if h>edges[-1] or not eligible.any():
                        continue
                    _,cif = competing_curves(rates[eligible],torch.tensor([h]),l[eligible],edges)
                    p = cif.mean(1)[:,0,0].numpy()
                    stats["horizons"][str(h)] = ipcw_metrics(t[eligible].numpy(),e[eligible].numpy(),l[eligible].numpy(),p,h,km)
            stats["nll"] = float(nll.mean()) if len(nll) else None
        report["stages"][f"S{stage}"] = stats
        all_nll.append((valid,nll))
    # Same patient-normalized selection score as training; not a pooled prefix score.
    sums = torch.zeros(len(rows))
    counts = torch.zeros(len(rows))
    for valid,loss in all_nll:
        sums[valid] += loss
        counts[valid] += 1
    report["selection_nll"] = float((sums[counts>0]/counts[counts>0]).mean()) if (counts>0).any() else None
    return report
