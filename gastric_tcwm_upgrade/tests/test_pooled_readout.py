from dataclasses import replace

import pytest
import torch

from stageworld_tcwm.model import TreatmentBeliefWorld,PooledOutcomeReadout,OutcomeReadout


def test_pooled_readout_reduces_only_outcome_capacity_and_remains_shared(config,cohort):
    reference = TreatmentBeliefWorld(config)
    cfg = replace(config,readout_kind="pooled",clinical_anchor=True,observation_update="residual")
    model = TreatmentBeliefWorld(cfg).eval()
    assert isinstance(reference.outcome,OutcomeReadout)
    assert isinstance(model.outcome,PooledOutcomeReadout)
    assert sum(p.numel() for p in model.outcome.parameters()) < sum(p.numel() for p in reference.outcome.parameters())
    original_world = {name:p.shape for name,p in reference.named_parameters() if not name.startswith(("outcome.","observation_"))}
    pooled_world = {name:p.shape for name,p in model.named_parameters() if not name.startswith(("outcome.","observation_"))}
    assert original_world == pooled_world
    batch = cohort.batch(torch.arange(8))
    batch["image_valid"][:] = True
    batch["ct1_available_stage"][:] = 1
    model.fit_statistics(batch)
    model.fit_clinical_anchors(batch)
    initial = model(batch,samples=2,seed=5)
    anchor = model.recurrence_anchor(batch["clinical"])
    torch.testing.assert_close(initial["predictions"],anchor[:,None,None].expand(-1,3,2))
    with torch.no_grad():
        model.outcome.output[-1].weight.normal_(std=.1)
    before = model(batch,samples=2,seed=5)
    torch.testing.assert_close(before["predictions"][:,1],before["predictions"][:,2],rtol=0,atol=0)
    batch["ct1"] = torch.randn_like(batch["ct1"])*5
    batch["binary"] = 1-batch["binary"]
    after = model(batch,samples=2,seed=5)
    torch.testing.assert_close(before["predictions"][:,0],after["predictions"][:,0],rtol=0,atol=0)
    torch.testing.assert_close(before["prior_features"],after["prior_features"],rtol=0,atol=0)
    assert not torch.allclose(before["predictions"][:,1],after["predictions"][:,1])
    restored = TreatmentBeliefWorld(cfg).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(restored(batch,samples=2,seed=5)["predictions"],after["predictions"],rtol=0,atol=0)


@pytest.mark.parametrize("changes",[{"endpoint":"survival"},{"architecture":"predictive_ct"},{"readout_kind":"unknown"}])
def test_pooled_readout_scope_is_explicit(config,changes):
    with pytest.raises(ValueError):
        replace(config,readout_kind="pooled",**changes).validate() if "readout_kind" not in changes else replace(config,**changes).validate()
