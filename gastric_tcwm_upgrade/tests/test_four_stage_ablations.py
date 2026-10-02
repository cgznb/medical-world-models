"""Target/freeze ablations preserve stage semantics and held-out boundaries."""
import hashlib
import json

import pytest
import torch
from torch.nn import functional as F

from stageworld_tcwm import four_stage_training
from stageworld_tcwm.four_stage_model import FourStageModel
from stageworld_tcwm.four_stage_training import (
    _auxiliary_loss, _evaluate_auxiliary, evaluate_four_stage,
    load_four_stage_export, train_four_stage,
)
from test_four_stage_training import cohort_fixture


def _batch(cohort, start=0, end=12):
    return {name: values[start:end].clone() for name, values in cohort.tensors.items()}


def _fitted_model(cohort, *, adapter_rank=0):
    model = FourStageModel(image_dim=8, hidden_dim=8, rank=2,
                           state_adapter_rank=adapter_rank)
    model.fit_statistics(_batch(cohort))
    return model.eval()


def test_zero_initialized_adapter_preserves_all_states_and_predictions():
    cohort = cohort_fixture()
    original = _fitted_model(cohort)
    adapted = _fitted_model(cohort, adapter_rank=2)
    incompatible = adapted.load_state_dict(original.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "state_adapter.down.weight", "state_adapter.down.bias",
        "state_adapter.up.weight", "state_adapter.up.bias",
    }
    batch = _batch(cohort)
    with torch.no_grad():
        for name, expected in original(batch).items():
            torch.testing.assert_close(adapted(batch)[name], expected, rtol=0, atol=0)
    assert adapted.claims()["state_adapter_rank"] == 2
    assert adapted.claims()["validated_concept_dynamics"] is False


def test_recurrence_head_is_called_only_on_complete_s3_during_adaptation():
    cohort = cohort_fixture()
    model = _fitted_model(cohort, adapter_rank=2)
    model.configure_terminal_adaptation()
    model.train()
    observed = []
    hook = model.risk_head.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone()))
    output = model(_batch(cohort))
    hook.remove()
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], output["states"][:, 3], rtol=0, atol=0)
    assert output["logits"].shape == (12,)
    assert not model.representation_dropout.training
    assert all(not parameter.requires_grad for parameter in model.pcr_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.ct1_decoder.parameters())


def test_pcr_only_loss_never_reads_ct1_and_excludes_inapplicable_s1(monkeypatch):
    cohort = cohort_fixture()
    model = _fitted_model(cohort)
    batch = _batch(cohort)
    output = model(batch)
    # These four labels must never enter the pCR objective: absent NAC,
    # hypothetical S1, wrong CT stage, and explicitly missing pCR.
    batch["role"][1, 0] = 3
    batch["scan_event_index"][2] = 0
    batch["pcr_valid"][3] = False
    batch["pcr"][:4] = float("nan")
    batch["ct1"].fill_(float("nan"))

    def reject_ct_access(*args, **kwargs):
        raise AssertionError("pCR-only objective must not derive a CT1 target")

    monkeypatch.setattr(model, "ct1_target", reject_ct_access)
    loss, metrics = _auxiliary_loss(model, output, batch, experiment="pcr_only_frozen")
    expected = F.binary_cross_entropy_with_logits(output["pcr_logits"][4:], batch["pcr"][4:])
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert metrics["pcr_patients"] == 8
    loss.backward()
    assert all(parameter.grad is None for parameter in model.ct1_decoder.parameters())
    assert all(parameter.grad is None for parameter in model.risk_head.parameters())
    assert all(parameter.grad is None or not parameter.grad.count_nonzero()
               for parameter in model.surgery_transition.parameters())


@pytest.fixture(scope="module")
def auxiliary_source(tmp_path_factory):
    cohort = cohort_fixture()
    out = tmp_path_factory.mktemp("four_stage_auxiliary_source") / "source"
    result = train_four_stage(cohort, list(range(12)), list(range(12, 16)),
                              out, seed=29, diagnostic=True)
    return cohort, out, result


@pytest.fixture(scope="module")
def pcr_only_run(tmp_path_factory):
    cohort = cohort_fixture()
    cohort.tensors["ct1"][:16] = float("nan")
    out = tmp_path_factory.mktemp("four_stage_pcr_only") / "run"
    with pytest.MonkeyPatch.context() as patch:
        def reject_ct_access(*args, **kwargs):
            raise AssertionError("pCR-only training must never construct a CT1 target")
        patch.setattr(FourStageModel, "ct1_target", reject_ct_access)
        result = train_four_stage(cohort, list(range(12)), list(range(12, 16)),
                                  out, seed=29, diagnostic=True,
                                  experiment="pcr_only_frozen")
    return cohort, out, result


def test_pcr_only_training_works_without_ct1_and_keeps_decoder_fixed(pcr_only_run):
    cohort, out, result = pcr_only_run
    model, payload = load_four_stage_export(out / "inference.pt")
    assert result["experiment"] == "pcr_only_frozen"
    assert result["test_evaluated"] is False
    assert result["auxiliary"]["completed_steps"] == 2
    assert result["completed_steps"] == 2
    torch.manual_seed(29)
    initial = _fitted_model(cohort)
    last = torch.load(out / "aux_last.pt", weights_only=True)["model"]
    for name, expected in initial.ct1_decoder.state_dict().items():
        torch.testing.assert_close(last["ct1_decoder." + name], expected, rtol=0, atol=0)
    assert not torch.equal(last["pcr_head.weight"], initial.pcr_head.weight)
    metrics = _evaluate_auxiliary(model, _batch(cohort, 12, 16), experiment="pcr_only_frozen")
    assert metrics["selection_loss"] == pytest.approx(metrics["pcr_nll"])
    assert payload["metrics"] == result
    assert torch.isnan(cohort.tensors["binary"][16:]).all()


@pytest.fixture(scope="module")
def adapter_run(auxiliary_source, tmp_path_factory):
    cohort, source, _ = auxiliary_source
    out = tmp_path_factory.mktemp("four_stage_adapter") / "run"
    result = train_four_stage(cohort, list(range(12)), list(range(12, 16)),
                              out, seed=29, diagnostic=True,
                              experiment="ct_pcr_adapter",
                              auxiliary_checkpoint=source / "aux_best.pt")
    return cohort, source, out, result


def test_adapter_training_reuses_auxiliary_and_preserves_frozen_parameters(adapter_run):
    cohort, source, out, result = adapter_run
    before, _ = load_four_stage_export(source / "aux_best.pt")
    after, _ = load_four_stage_export(out / "last.pt")
    after.configure_terminal_adaptation()
    before_state = before.state_dict()
    after_state = after.state_dict()
    frozen = [name for name, parameter in after.named_parameters() if not parameter.requires_grad]
    assert frozen
    for name in frozen:
        torch.testing.assert_close(after_state[name], before_state[name], rtol=0, atol=0)
    for name, value in after.named_buffers():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    assert after.state_adapter.up.weight.abs().sum() > 0
    assert any(not torch.equal(after_state[name], before_state[name])
               for name in before_state if name.startswith("medical_transition."))
    assert result["terminal_trainable_parameter_count"] == 201
    assert result["completed_steps"] == 2
    assert result["auxiliary"]["executed_steps_this_run"] == 0
    assert result["auxiliary"]["executed_epochs_this_run"] == 0
    assert result["auxiliary"]["reused"] is True
    assert result["auxiliary_source"]["reproduced"] is True
    assert result["auxiliary_source"]["path"] == str((source / "aux_best.pt").resolve())
    assert len(result["auxiliary_source"]["sha256"]) == 64
    assert result["terminal_parameter_groups"] == {
        "adaptation": {"parameters": 141, "lr": .0001},
        "endpoint": {"parameters": 60, "lr": .0005},
    }
    imported = torch.load(out / "aux_last.pt", weights_only=True)
    assert imported["executed_steps_this_run"] == 0
    assert imported["reused"] is True
    source_checkpoint = torch.load(source / "aux_best.pt", weights_only=True)
    for name, value in source_checkpoint["model"].items():
        torch.testing.assert_close(imported["model"][name], value, rtol=0, atol=0)
    assert result["test_evaluated"] is False
    assert torch.isnan(cohort.tensors["binary"][16:]).all()


def test_adapter_export_roundtrips_and_selects_s3_validation_nll(adapter_run):
    cohort, _, out, result = adapter_run
    exported, payload = load_four_stage_export(out / "inference.pt")
    best, _ = load_four_stage_export(out / "best.pt")
    assert payload["config"]["state_adapter_rank"] == 2
    assert result["experiment"] == "ct_pcr_adapter"
    batch = _batch(cohort, 12, 16)
    actual, predictions = evaluate_four_stage(exported, batch)
    expected, best_predictions = evaluate_four_stage(best, batch)
    assert actual == expected == result["validation"]
    torch.testing.assert_close(predictions["logits"], best_predictions["logits"], rtol=0, atol=0)
    history = json.loads((out / "history.json").read_text())
    assert all("auxiliary" in item for item in history)
    assert all("pcr_auc" in item["auxiliary"]["validation"] for item in history)
    assert "terminal_auxiliary" in result
    selected = next(item for item in history if item["step"] == result["selected_step"])
    # The auxiliary regularizer affects optimization, but selection retains
    # the factual S3 validation NLL contract, including the eligible step0.
    minimum = min(item["validation"]["nll"] for item in history)
    assert selected["validation"]["nll"] <= minimum + four_stage_training.DEFAULTS["min_delta"]


def test_adapter_restores_parent_terminal_minibatch_sequence_and_rng(adapter_run):
    _, source, out, result = adapter_run
    source_path = source / "aux_last.pt"
    source_last = torch.load(source_path, weights_only=True)
    child_last = torch.load(out / "last.pt", weights_only=True)
    parent_terminal_last = torch.load(source / "last.pt", weights_only=True)
    provenance = result["auxiliary_source"]
    assert provenance["sampler_checkpoint_path"] == str(source_path.resolve())
    assert provenance["sampler_checkpoint_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert provenance["terminal_sampler_policy"] == (
        "restore_parent_aux_last_rng; same paired terminal minibatch sequence")

    # The new terminal phase starts a fresh permutation after the source's
    # last auxiliary epoch; it must not restart from the seed or aux_best RNG.
    generator = torch.Generator().set_state(source_last["sampler_rng"])
    expected_permutation = torch.randperm(12, generator=generator)
    expected_batches = list(expected_permutation.split(8))
    assert [len(batch) for batch in expected_batches] == [8, 4]
    assert child_last["step"] == 2
    assert child_last["sampler_cursor"] == sum(map(len, expected_batches))
    torch.testing.assert_close(child_last["sampler_permutation"],
                               torch.cat(expected_batches), rtol=0, atol=0)
    torch.testing.assert_close(child_last["sampler_rng"], generator.get_state(), rtol=0, atol=0)
    for name in ("sampler_rng", "sampler_permutation", "torch_rng"):
        torch.testing.assert_close(child_last[name], parent_terminal_last[name], rtol=0, atol=0)
    # Both terminal recipes disable representation dropout, so terminal
    # optimization consumes no torch random draws after this restored state.
    torch.testing.assert_close(child_last["torch_rng"], source_last["torch_rng"], rtol=0, atol=0)


@pytest.mark.parametrize("mismatch", ["seed", "partition", "encoding", "selection"])
def test_adapter_rejects_unmatched_auxiliary_source(auxiliary_source, tmp_path, mismatch):
    cohort, source, _ = auxiliary_source
    selected = torch.load(source / "aux_best.pt", weights_only=True)
    parent = json.loads((source / "metrics.json").read_text())
    if mismatch == "seed":
        selected["seed"] = 43
        expected = "seed mismatch"
    elif mismatch == "partition":
        selected["partition_sha256"]["validation"] = "another-patient-membership"
        expected = "partition mismatch"
    elif mismatch == "encoding":
        selected["config"]["medical_condition_encoding"] = "legacy_one_hot"
        expected = "config mismatch"
    else:
        parent["auxiliary"]["selected_step"] += 1
        expected = "selected auxiliary state"
    altered = tmp_path / "altered_source"
    altered.mkdir()
    torch.save(selected, altered / "aux_best.pt")
    (altered / "metrics.json").write_text(json.dumps(parent))
    with pytest.raises(ValueError, match=expected):
        train_four_stage(cohort, list(range(12)), list(range(12, 16)),
                         tmp_path / "rejected_run", seed=29, diagnostic=True,
                         experiment="ct_pcr_adapter",
                         auxiliary_checkpoint=altered / "aux_best.pt")


@pytest.mark.parametrize("mismatch", ["seed", "partition", "step"])
def test_adapter_rejects_unmatched_auxiliary_sampler(auxiliary_source, tmp_path, mismatch):
    cohort, source, _ = auxiliary_source
    last = torch.load(source / "aux_last.pt", weights_only=True)
    if mismatch == "seed":
        last["seed"] = 43
    elif mismatch == "partition":
        last["partition_sha256"]["train"] = "another-patient-membership"
    else:
        last["step"] += 1
    altered = tmp_path / "altered_source"
    altered.mkdir()
    for name in ("aux_best.pt", "metrics.json"):
        (altered / name).write_bytes((source / name).read_bytes())
    torch.save(last, altered / "aux_last.pt")
    with pytest.raises(ValueError, match="last-checkpoint sampler identity mismatch"):
        train_four_stage(cohort, list(range(12)), list(range(12, 16)),
                         tmp_path / "rejected_run", seed=29, diagnostic=True,
                         experiment="ct_pcr_adapter",
                         auxiliary_checkpoint=altered / "aux_best.pt")


@pytest.mark.parametrize("experiment,checkpoint", [
    ("ct_pcr_adapter", None), ("pcr_only_frozen", "unexpected-source.pt"),
])
def test_auxiliary_import_is_required_only_for_adapter(tmp_path, experiment, checkpoint):
    with pytest.raises(ValueError, match="paired auxiliary checkpoint"):
        train_four_stage(cohort_fixture(), list(range(12)), list(range(12, 16)),
                         tmp_path / "rejected_run", seed=29, diagnostic=True,
                         experiment=experiment, auxiliary_checkpoint=checkpoint)
    assert not (tmp_path / "rejected_run").exists()
