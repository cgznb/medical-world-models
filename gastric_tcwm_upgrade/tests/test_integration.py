from pathlib import Path
import importlib.util
import csv
import pytest
import torch
from stageworld_tcwm.legacy import attach_survival
from stageworld_tcwm.synthetic import synthetic_cohort
from stageworld_tcwm.data import Cohort
from stageworld_tcwm.losses import feature_set_loss


def test_non_destructive_install(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/install_into_repo.py'
    spec=importlib.util.spec_from_file_location('installer',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    repo=tmp_path/'original';source=tmp_path/'extension';source.mkdir();(source/'README.md').write_text('extension')
    files=['src/stageworld/generated700_models.py','src/stageworld/data/treatment_compact.py','src/stageworld/surgery_s2_models.py']
    for name in files:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original unchanged')
    report=module.install(repo,source)
    assert not report['remote_repository_modified']
    assert all((repo/name).read_text()=='original unchanged' for name in files)
    assert (repo/'scripts/run_tcwm.py').exists()
    with pytest.raises(FileExistsError):module.install(repo,source)


def test_survival_sidecar_exact_join_and_administrative_censor(tmp_path):
    data=synthetic_cohort(20,16);path=tmp_path/'data.pt';data.save(path)
    fields=['patient_id','time_months','event','entry_s0_months','entry_s1_months','entry_s2_months','s0_valid','s1_valid','s2_valid']
    side=tmp_path/'followup.csv'
    with side.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,p in reversed(list(enumerate(data.ids))):
            writer.writerow(dict(zip(fields,[p,70 if i==0 else 24,1,0,0,0,1,1,1])))
    out=tmp_path/'surv.pt';attach_survival(path,side,out,'Verified surgery origin',60)
    attached=Cohort.load(out)
    assert attached.tensors['time'][0]==60 and attached.tensors['event'][0]==0
    assert attached.tensors['event'][1]==1 and attached.ids==data.ids


def test_survival_sidecar_does_not_guess_missing_patients(tmp_path):
    data=synthetic_cohort(20,16);path=tmp_path/'data.pt';data.save(path)
    side=tmp_path/'missing.csv';side.write_text('patient_id,time_months\nUNKNOWN,12\n')
    with pytest.raises(ValueError):attach_survival(path,side,tmp_path/'out.pt','surgery')


def test_feature_loss_is_set_permutation_invariant():
    pred=torch.randn(3,27,16);target=torch.randn_like(pred);valid=torch.ones(3,dtype=torch.bool)
    a=feature_set_loss(pred,target,valid)
    b=feature_set_loss(pred[:,torch.randperm(27)],target[:,torch.randperm(27)],valid)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
