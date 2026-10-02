import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.modality_support import fit_modality_support, phase_action_support_flags


def test_absent_surgery_is_not_supported_by_present_surgery():
    batch = modality_batch()
    support = fit_modality_support(batch)
    flags = phase_action_support_flags(batch, support)
    assert flags["support_audited"]
    assert not flags["unsupported_events"].any()
    assert flags["sparse_events"].all()
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    changed["modality_value"][:, 1, 6] = False
    flags = phase_action_support_flags(changed, support)
    assert flags["unsupported_events"][:, 1].all()
    assert not flags["unsupported_events"][:, 0].any()


def test_hypothetical_actions_do_not_create_training_support():
    batch = modality_batch()
    batch["role"][:] = 3
    support = fit_modality_support(batch)
    assert support["phase_action_patients"] == {}
    assert phase_action_support_flags(batch, support)["unsupported_events"].all()


def test_legacy_support_cannot_certify_phase_specific_scenarios():
    batch = modality_batch()
    support = fit_modality_support(batch)
    del support["phase_action_patients"]
    flags = phase_action_support_flags(batch, support)
    assert flags["support_audited"] is False
    assert flags["unsupported_events"].all()
