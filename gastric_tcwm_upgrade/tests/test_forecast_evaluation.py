import copy

import pytest
import torch

from stageworld_tcwm.forecast_evaluation import (
    ForecastSpace, distribution_scores, merge_point_scores, point_scores,
    prior_feature_draws, score_feature_sets,
)
from stageworld_tcwm.model import model_from_config


def test_prior_forward_has_no_ct1_or_targets_and_is_batch_stable(cohort, config, monkeypatch):
    model = model_from_config(config).eval()
    model.fit_statistics(cohort.batch(torch.arange(24)))
    original_forward = model.forward

    def guard(batch, **kwargs):
        assert not {"ct1", "binary", "pcr", "post", "binary_valid", "ct1_available_stage"} & set(batch)
        assert kwargs["max_stage"] == 0 and kwargs["compute_aux"] is False
        assert not batch["image_valid"][:, 1].any()
        return original_forward(batch, **kwargs)

    monkeypatch.setattr(model, "forward", guard)
    batch = cohort.batch(torch.arange(24, 28))
    first = prior_feature_draws(model, batch, ["a", "b", "c", "d"], samples=4)
    batch["ct1"].fill_(1e9)
    batch["binary"].fill_(-100)
    batch["pcr"].fill_(-100)
    second = prior_feature_draws(model, batch, ["a", "b", "c", "d"], samples=4)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    separate = torch.cat([prior_feature_draws(model, {key: value[index:index+1] for key, value in batch.items()},
                                             [key], samples=4) for index, key in enumerate(["a", "b", "c", "d"])])
    torch.testing.assert_close(first, separate, rtol=1e-5, atol=1e-6)


def test_forecast_space_only_fits_training_and_mc_mean_follows_decoding(cohort):
    training = cohort.batch(torch.arange(24))
    space = ForecastSpace(training, cohort.ids[:24], rank=4)
    before = copy.deepcopy(space.provenance)
    cohort.tensors["ct1"][24:] += 1000
    again = ForecastSpace(training, cohort.ids[:24], rank=4)
    assert before == again.provenance
    assert space.ridge(cohort.batch(torch.arange(24, 30))).shape == (6, 8)
    tokens = training["ct0"][:2, None].repeat(1, 2, 1, 1)
    tokens[:, 1] = tokens[:, 1].flip(1)
    encoded_draws = space.encode_draws(tokens)
    torch.testing.assert_close(encoded_draws[:, 0], encoded_draws[:, 1])
    # The pooled SD is nonlinear: decoding/aggregating the mean state is not E[feature].
    assert not torch.allclose(encoded_draws.mean(1), space.encode(tokens.mean(1)))


def test_point_scores_use_same_training_mean_denominator():
    target = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    mean = torch.zeros(2, dtype=torch.float64)
    scores = point_scores({"mean": mean.expand_as(target), "perfect": target}, target, mean)
    assert merge_point_scores([scores["mean"]])["skill_vs_training_CT1_mean"] == 0
    assert merge_point_scores([scores["perfect"]])["skill_vs_training_CT1_mean"] == 1
    parts = [point_scores({"copy": target[:1]}, target[:1], mean)["copy"],
             point_scores({"copy": target[1:]}, target[1:], mean)["copy"]]
    assert merge_point_scores(parts)["patients"] == 2


def test_distribution_score_matches_degenerate_distance_and_counts_patients():
    draws = torch.zeros(2, 4, 3)
    target = torch.ones(2, 3)
    scores = distribution_scores(draws, target)
    torch.testing.assert_close(scores["energy_score"], torch.full((2,), 3**.5, dtype=torch.float64))
    assert scores["marginal_90pct_coverage"].sum() == 0
    assert scores["mean_projected_variance"].sum() == 0
    with pytest.raises(ValueError, match="K>=2"):
        distribution_scores(draws[:, :1], target)


def test_set_scores_ignore_cross_time_token_correspondence(cohort):
    target = cohort.tensors["ct1"][:2]
    draws = cohort.tensors["ct0"][:2, None].repeat(1, 4, 1, 1)
    scale = torch.ones(target.shape[-1])
    mean = target.mean((0, 1))[None, None]
    first = score_feature_sets(draws, target, draws[:, 0], mean, scale)
    permuted = score_feature_sets(draws, target.flip(1), draws[:, 0], mean, scale)
    for name in first:
        assert first[name] == pytest.approx(permuted[name], rel=1e-5, abs=1e-6)
