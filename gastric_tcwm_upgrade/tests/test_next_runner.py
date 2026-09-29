import copy
import json
from pathlib import Path
import sys

import pytest

from stageworld_tcwm.config import load_config, config_dict
from stageworld_tcwm.data import file_sha256, fingerprint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_next_round import verify_study_case


def test_locked_matrix_changes_only_declared_factor():
    values = {case: config_dict(*load_config(ROOT / f"configs/next/{case}.json"))
              for case in ("G0", "G1", "G2", "G3")}
    for case, reference, section, field in (("G1", "G0", "train", "learning_rate"),
                                           ("G2", "G1", "train", "observation_recon_weight"),
                                           ("G3", "G1", "model", "readout_kind")):
        expected = copy.deepcopy(values[reference])
        expected[section][field] = values[case][section][field]
        assert values[case] == expected
    assert values["G0"]["train"]["samples_eval"] == 64


@pytest.mark.parametrize("tamper", [None, "model", "steps", "provenance", "fold"])
def test_report_binds_exact_case_and_export(tmp_path, tamper):
    config = {"model": {"hidden": 64}, "train": {"seed": 17}}
    path = tmp_path / "G0.json"
    path.write_text(json.dumps(config))
    hashes = {"cohort": "abc", "split": "def"}
    study = {"configs": {"G0": config}, "fold_indices": [0], "fold_hashes": [hashes]}
    fold = {"index": 0, "hashes": hashes.copy()}
    bundle = {"contract_id": "locked", "selected_kind": "clinical_baseline", "selected_epoch": -1,
              "selected_optimizer_steps": 0, "selected_supervised_steps": 0}
    training = {**bundle, "optimizer_steps": 250, "supervised_steps": 250}
    provenance = {"study_id": fingerprint(study), "case": "G0", "fold": 0,
                  "config_sha256": file_sha256(path)}
    contract = copy.deepcopy(config)
    if tamper == "model":
        contract["model"]["hidden"] = 128
    elif tamper == "steps":
        training["selected_supervised_steps"] = 25
    elif tamper == "provenance":
        provenance["case"] = "G1"
    elif tamper == "fold":
        fold["hashes"]["split"] = "different"
    if tamper:
        with pytest.raises(ValueError):
            verify_study_case(study, "G0", fold, contract, bundle, training, provenance, path)
    else:
        verify_study_case(study, "G0", fold, contract, bundle, training, provenance, path)
