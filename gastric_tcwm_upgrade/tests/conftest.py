import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

@pytest.fixture(autouse=True)
def deterministic_threads():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(71)

@pytest.fixture
def config():
    from stageworld_tcwm.config import ModelConfig
    return ModelConfig(image_dim=16,hidden=32,latent_dim=8,world_blocks=1,surgery_blocks=1,
                       update_blocks=1,readout_blocks=1,readout_slots=4,dropout=0,
                       flow_blocks=2,flow_train_steps=2,flow_eval_steps=3)

@pytest.fixture
def cohort():
    from stageworld_tcwm.synthetic import synthetic_cohort
    return synthetic_cohort(n=48,image_dim=16,seed=71)
