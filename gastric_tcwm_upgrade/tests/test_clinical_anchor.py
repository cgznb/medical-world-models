from dataclasses import replace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stageworld_tcwm.clinical import ClinicalAnchor
from stageworld_tcwm.model import TreatmentBeliefWorld


def test_anchor_matches_training_only_logistic_and_roundtrips(cohort):
    train = cohort.batch(torch.arange(20))
    held = cohort.batch(torch.arange(20, 30))
    anchor = ClinicalAnchor()
    anchor.fit(train['clinical'], train['binary'], train['binary_valid'])
    valid = train['binary_valid']
    reference = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., solver='lbfgs', max_iter=2000, tol=1e-8, random_state=17))
    reference.fit(train['clinical'][valid].double().numpy(), train['binary'][valid].numpy())
    actual = anchor(held['clinical']).sigmoid().numpy()
    np.testing.assert_allclose(actual, reference.predict_proba(held['clinical'].numpy())[:, 1], atol=2e-6)
    assert not list(anchor.parameters())
    restored = ClinicalAnchor()
    restored.load_state_dict(anchor.state_dict())
    torch.testing.assert_close(restored(held['clinical']), anchor(held['clinical']))


def test_unfitted_anchor_rejected():
    with pytest.raises(RuntimeError, match='training patients'):
        ClinicalAnchor()(torch.zeros(2, 32))


@pytest.mark.parametrize('all_valid', [False, True])
def test_anchor_handles_missing_or_single_class_labels(all_valid):
    anchor = ClinicalAnchor()
    anchor.fit(torch.randn(4, 32), torch.zeros(4), torch.full((4,), all_valid))
    assert torch.isfinite(anchor(torch.randn(3, 32))).all()


def test_residual_model_starts_at_clinical_baseline_and_keeps_stage_boundaries(config, cohort):
    cfg = replace(config, clinical_anchor=True, residual_scale=.25, observation_update='residual')
    model = TreatmentBeliefWorld(cfg).eval()
    train = cohort.batch(torch.arange(20))
    model.fit_statistics(train)
    model.fit_clinical_anchors(train)
    batch = cohort.batch(torch.arange(20, 24))
    result = model(batch, samples=2, seed=17)
    anchor = model.recurrence_anchor(batch['clinical'])
    torch.testing.assert_close(result['predictions'], anchor[:, None, None].expand(-1, 3, 2))
    pcr = model.pcr_anchor(batch['clinical'])
    torch.testing.assert_close(result['pcr_logits'], pcr[:, None].expand(-1, 2))
    with torch.no_grad():
        model.outcome.output[-1].weight.normal_(std=.1)
    before = model(batch, samples=2, seed=17)
    batch['ct1'] = torch.randn_like(batch['ct1'])*10
    batch['binary'] = 1-batch['binary']
    batch['pcr'] = 1-batch['pcr']
    after = model(batch, samples=2, seed=17)
    torch.testing.assert_close(before['predictions'][:, 0], after['predictions'][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(before['pcr_logits'], after['pcr_logits'], rtol=0, atol=0)
    assert not torch.allclose(before['predictions'][:, 1], after['predictions'][:, 1])
