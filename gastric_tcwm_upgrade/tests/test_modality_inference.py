from dataclasses import asdict

import pytest
import torch

from modality_fixtures import modality_batch
from stageworld_tcwm.modality_inference import ModalityPredictor
from stageworld_tcwm.modality_support import fit_modality_support, modality_support_flags
from stageworld_tcwm.timeline_config import TimelineConfig
from stageworld_tcwm.timeline_model import TimelineModel


def test_strict_inference_roundtrip_and_mask(tmp_path):
    batch = modality_batch()
    cfg = TimelineConfig(image_dim=16, hidden=64, dropout=0.)
    model = TimelineModel(cfg)
    model.fit_statistics(batch)
    model.eval()
    path = tmp_path / "inference.pt"
    torch.save({"schema": "modality-event-v2", "model_config": asdict(cfg),
                "model_state": model.state_dict(), "support": fit_modality_support(batch),
                "fit_ids": ["SYNTHETIC-1", "SYNTHETIC-2"],
                "encoders": {"fit_ids": ["SYNTHETIC-1", "SYNTHETIC-2"]},
                "metadata": {"source_mode": "synthetic", "query_names": ["S0", "S1", "S2_replay", "S3"]}}, path)
    predictor = ModalityPredictor.load(path)
    result = predictor.predict(batch)
    torch.testing.assert_close(result["risk"], model(batch)["logits"].sigmoid(), rtol=0, atol=0)
    assert result["risk"].shape == (2, 4)
    assert result["metadata"]["prospective_supported"] is False
    assert result["metadata"]["radiotherapy_supported"] is False
    batch["query_mask"][:, -1] = False
    assert predictor.predict(batch)["risk"][:, -1].isnan().all()
    with pytest.raises(ValueError):
        predictor.predict({**batch, "drugs": ["EXAMPLE"]})
    batch["modality_applicable"][:, 0, 1] = True
    with pytest.raises(ValueError):
        predictor.predict(batch)


def test_old_inference_schema_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"schema": "tcwm-cohort-v1"}, path)
    with pytest.raises(ValueError):
        ModalityPredictor.load(path)


def test_modality_support_ignores_names_and_counts_patients():
    batch = modality_batch()
    support = fit_modality_support(batch)
    assert support["present_patients"][0] == 2
    changed = {**batch, "drugs": ["anything"], "unseen_treatment_names_count": torch.full((2,), 100)}
    assert fit_modality_support(changed) == support
    flags = modality_support_flags(changed, support)
    assert not flags["unseen_modalities"].any()
