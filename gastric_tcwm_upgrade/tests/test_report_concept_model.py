from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.modality_inference import ModalityPredictor
from stageworld_tcwm.modality_support import fit_modality_support
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import EVENT_FIELDS, TimelineModel


def fitted(batch, enabled=True):
    model = TimelineModel(TimelineConfig(image_dim=16, hidden=64, dropout=0.,
                                         objective="terminal_state_v1", s1_report_concepts=enabled))
    model.fit_statistics(batch)
    return model.eval()


def report_statistics():
    return {"report_mean": torch.tensor([.5, .2, 2., 3.]),
            "report_scale": torch.tensor([1., .1, .5, .25]),
            "report_counts": torch.tensor([6, 6, 6, 6]), "report_fitted": torch.tensor(True)}


def predictor(model, batch, trained=True, statistics=None):
    return ModalityPredictor(model, fit_modality_support(batch),
                             {"source_mode": "synthetic", "report_concept_head_trained": trained},
                             report_statistics() if statistics is None else statistics)


def test_report_option_is_explicit_boolean_and_terminal_only():
    assert TimelineConfig().s1_report_concepts is False
    with pytest.raises(ValueError, match="require terminal_state_v1"):
        TimelineConfig(s1_report_concepts=True).validate()
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="must be boolean"):
            TimelineConfig(objective="terminal_state_v1", s1_report_concepts=value).validate()


def test_optional_head_does_not_change_a_initialization_or_old_state_contract():
    batch = modality_batch()
    torch.manual_seed(123)
    original = fitted(batch, enabled=False)
    torch.manual_seed(123)
    extended = fitted(batch)
    old_state = original.state_dict()
    assert all(torch.equal(value, extended.state_dict()[name]) for name, value in old_state.items())
    added = set(extended.state_dict()) - set(old_state)
    assert added and all(name.startswith(("s1_report_pool.", "s1_report_output.")) for name in added)
    assert not any(name.startswith("s1_report_") for name in old_state)
    old_config = asdict(original.cfg)
    old_config.pop("s1_report_concepts")
    loaded = TimelineModel(TimelineConfig.from_dict(old_config)).eval()
    loaded.load_state_dict(old_state, strict=True)
    before, after = original(batch), loaded(batch)
    assert "s1_concept_logits" not in before
    for key in ("logits", "pcr_logits", "forecast"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_report_head_only_reads_full_s1_and_never_targets_or_ct1():
    batch = modality_batch()
    model = fitted(batch)
    result = model(batch)
    assert result["s1_concept_logits"].shape == (2, 4)
    assert result["s1_concept_target_stage"] == 1
    assert result["s1_concept_available_stage"] == 2
    state = result["checkpoint_states"][1]
    expected = model.s1_report_output(model.s1_report_pool(
        torch.cat((state.z, state.memory, state.clinical), 1)).mean(1))
    torch.testing.assert_close(result["s1_concept_logits"], expected, rtol=0, atol=0)
    changed = deepcopy(batch)
    changed["ct1"].fill_(float("nan"))
    changed["binary"].fill_(float("nan"))
    changed["pcr"].fill_(float("nan"))
    changed["s1_concepts"] = torch.full((2, 4), float("nan"))
    changed["s1_concept_valid"] = torch.zeros(2, 4, dtype=torch.bool)
    changed["modality_value"][:, 1:] = 0
    altered = model(changed)
    torch.testing.assert_close(result["s1_concept_logits"], altered["s1_concept_logits"], rtol=0, atol=0)
    for name in EVENT_FIELDS:
        changed[name] = changed[name][:, :1]
    changed["query_order"] = torch.ones(2, 1, dtype=torch.long)
    changed["query_mask"] = torch.ones(2, 1, dtype=torch.bool)
    torch.testing.assert_close(result["s1_concept_logits"], model(changed)["s1_concept_logits"], rtol=0, atol=1e-6)


def test_concept_mask_is_factual_s1_without_ct1_or_nac_requirement():
    batch = modality_batch(n=4)
    model = fitted(batch)
    batch["image_valid"][:, 1] = False
    batch["modality_value"][1, 0] = 0
    batch["role"][2, 0] = 1
    batch["role"][3, 0] = 3
    result = model(batch)
    assert result["s1_concept_mask"].tolist() == [True, True, False, False]
    assert result["pcr_mask"].tolist() == [True, False, False, False]
    result["s1_concept_logits"].retain_grad()
    result["s1_concept_logits"][result["s1_concept_mask"]].square().mean().backward()
    assert (result["s1_concept_logits"].grad[2:] == 0).all()
    for module in (model.s1_report_pool, model.s1_report_output, model.event_jump):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and any(gradient.abs().sum() > 0 for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_zero_event_prefix_has_no_s1_concept_prediction():
    batch = modality_batch()
    model = fitted(batch)
    for name in EVENT_FIELDS:
        batch[name] = batch[name][:, :0]
    batch["query_order"] = torch.zeros(2, 1, dtype=torch.long)
    batch["query_mask"] = torch.ones(2, 1, dtype=torch.bool)
    output = model(batch)
    assert not output["s1_concept_mask"].any()
    assert output["s1_concept_logits"].shape == (2, 4)


def test_report_inference_unscales_coordinates_with_explicit_units_and_scope():
    batch = modality_batch()
    model = fitted(batch)
    with torch.no_grad():
        model.s1_report_output.weight.zero_()
        model.s1_report_output.bias.copy_(torch.tensor([0., 1., 2., -2.]))
    output = predictor(model, batch).predict(batch)
    names = model.s1_concept_names
    expected = [.5, 30., torch.expm1(torch.tensor(3.)).item(), torch.expm1(torch.tensor(2.5)).item()]
    for index, name in enumerate(names):
        torch.testing.assert_close(output["s1_report_predictions"][name], torch.full((2,), expected[index]))
        assert output["s1_report_valid"][name].all()
    info = output["metadata"]["s1_report_concepts"]
    assert info["output_units"] == ["probability", "percent", "specimen_node_count_scale", "cm"]
    assert info["target_stage"] == 1 and info["available_stage"] == 2
    assert info["clinically_adjudicated"] is False
    assert info["S2_patient_residual_burden"] is False
    assert info["longitudinal_semantics_validated"] is False
    assert output["risk"][:, :3].isnan().all()


def test_report_inference_masks_untrained_unsupported_and_out_of_domain_predictions():
    batch = modality_batch()
    model = fitted(batch)
    untrained = predictor(model, batch, trained=False).predict(batch)
    assert all(values.isnan().all() for values in untrained["s1_report_predictions"].values())
    assert not untrained["metadata"]["s1_report_concepts"]["head_trained"]
    with torch.no_grad():
        model.s1_report_output.weight.zero_()
        model.s1_report_output.bias.copy_(torch.tensor([0., -3., -6., -16.]))
    result = predictor(model, batch).predict(batch)
    for name in model.s1_concept_names[1:]:
        assert result["s1_report_out_of_domain"][name].all()
        assert result["s1_report_predictions"][name].isnan().all()
        assert result["s1_report_transformed_predictions"][name].lt(0).all()
    with torch.no_grad():
        model.s1_report_output.bias.zero_()
    batch["modality_value"][1, 0] = 0
    result = predictor(model, batch).predict(batch)
    for name in model.s1_concept_names[:2]:
        assert not result["s1_report_valid"][name][1]
    for name in model.s1_concept_names[2:]:
        assert result["s1_report_valid"][name][1]


def test_report_checkpoint_requires_valid_statistics_and_roundtrips(tmp_path):
    batch = modality_batch()
    model = fitted(batch)
    with pytest.raises(ValueError, match="target statistics"):
        ModalityPredictor(model, fit_modality_support(batch), {})
    for key, bad in (("report_scale", torch.zeros(4)), ("report_mean", torch.full((4,), float("nan"))),
                     ("report_counts", torch.full((4,), -1)), ("report_fitted", torch.tensor(False))):
        statistics = report_statistics()
        statistics[key] = bad
        with pytest.raises(ValueError, match="Invalid S1"):
            predictor(model, batch, statistics=statistics)
    path = tmp_path / "report.pt"
    torch.save({"schema": "modality-event-v2", "model_config": asdict(model.cfg),
                "model_state": model.state_dict(), "target_statistics": report_statistics(),
                "support": fit_modality_support(batch), "fit_ids": ["SYN-A", "SYN-B"],
                "encoders": {"fit_ids": ["SYN-A", "SYN-B"]},
                "metadata": {"report_concept_head_trained": True}}, path)
    expected = predictor(model, batch).predict(batch)
    actual = ModalityPredictor.load(path).predict(batch)
    for name in model.s1_concept_names:
        torch.testing.assert_close(actual["s1_report_predictions"][name],
                                   expected["s1_report_predictions"][name], rtol=0, atol=0, equal_nan=True)
