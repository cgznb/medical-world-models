# ResponseWM: future MRI and pCR prediction

A standalone upgrade of the workflow from
`cgznb/breast-world-model-v2-pcr` (audited base commit
`11220d38dd076439951d08928335d5fe4fa880b7`). The model combines a shared
three-phase 3D encoder, history Transformer, coupled image/semantic flow,
stochastic future trajectories, and probability-marginalized pCR prediction.

Read [README_ZH.md](README_ZH.md) for the architecture and general interfaces,
and [the existing-data guide](docs/ISPY2_EXISTING_DATA_ZH.md) for the real cohort
adaptation. Public aggregate results are in [results/breast](../results/breast).

## Install and verify

Use Python 3.10 or later and an appropriate PyTorch installation. The optional
MONAI backend requires MONAI 1.5.1; the current real-data run uses `native`.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python joint.py smoke --output runs/synthetic_smoke
```

The smoke command creates synthetic data and performs a small engineering
check. Its metrics are not real-patient performance results.

## Reproduce the existing-data run

The private source manifest, continuous VQ latent arrays, and matching codec
must be supplied locally. They are not distributed in this repository.

```bash
export BREAST_DATA_ROOT="$PWD/private_data"
python scripts/prepare_existing_ispy2.py \
  --manifest "$BREAST_DATA_ROOT/world_v2_manifest.json" \
  --latent-root "$BREAST_DATA_ROOT/raw_latents" \
  --codec "$BREAST_DATA_ROOT/vq_codec.pt" \
  --output data/ispy2
python joint.py audit --manifest data/ispy2/direct_t0_t3.json \
  --scan-arrays --output runs/ispy2_audit.json
python -u scripts/run_existing_ispy2.py \
  --config configs/ispy2_t0_t3_5090.yaml \
  --manifest data/ispy2/direct_t0_t3.json \
  --output runs/ispy2_t0_t3
```

The controller runs representation, flow, readout, and joint training in order
and resumes each stage from `last.pt` when the same command is restarted.
Configuration, manifests, and cached arrays must remain unchanged within a run.

## Result scope

The real run retains 764 training and 102 development-validation patients.
There is no independent test set. At the recorded snapshot, representation
had completed 10,000 steps and flow had reached 27,990 of 30,000 steps;
readout and joint had not started. Their absence means no final pCR AUROC is
available from this run. Intermediate representation/flow objectives must not
be reported as pCR performance. See the timestamped
[aggregate snapshot](../results/breast/training_snapshot.json).

The frozen VQ codec was selected using the same development-validation
patients, and upstream localizer patient overlap remains unverified. These
results do not establish independent clinical generalization.

## License and contents

This publication contains source code, tests, example configurations, and
aggregate results. It excludes patient data, actual split manifests,
patient-level predictions, and trained weights. [LICENSE](LICENSE) documents
the mixed licensing: original additions are MIT, while the VQ and DiT
adaptations retain non-commercial terms. See [sources](docs/SOURCES.md) and
the notices in `licenses/`.
