"""Training-only empirical support. This is NOT a conditional positivity proof."""
import torch

def fit_support(batch):
    treatment = batch["treatment"].float().flatten(1).cpu()
    keys = torch.cat((treatment,batch["surgery"].float().cpu()[:,None]),1)
    values, counts = torch.unique(keys,dim=0,return_counts=True)
    return {"values":values,"counts":counts,"interval_min":float(batch["interval_days"].min()),
            "interval_max":float(batch["interval_days"].max()),"training_patients":len(treatment)}

def support_flags(support,batch,min_count=5):
    keys = torch.cat((batch["treatment"].float().flatten(1).cpu(),batch["surgery"].float().cpu()[:,None]),1)
    flags = []
    for i,key in enumerate(keys):
        distance = (support["values"]-key).abs().amax(-1)
        exact = distance < 1e-6
        count = int(support["counts"][exact].sum())
        patient = []
        if not count:
            patient.append("unseen_treatment_surgery_combination")
        elif count < min_count:
            patient.append("sparse_combination_support")
        interval = float(batch["interval_days"][i])
        if interval < support["interval_min"] or interval > support["interval_max"]:
            patient.append("interval_outside_training_range")
        if int(batch["surgery"][i]) in (2,3):
            patient.append("surgery_unknown_or_conflicting")
        flags.append({"count":count,"warnings":patient})
    return flags
