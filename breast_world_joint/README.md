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

The real run completed all four stages on 2026-09-29 at 08:24:57 (UTC+8),
using 10,000 / 30,000 / 3,000 / 5,000 optimization steps. Selected checkpoints
are at steps 250 / 12,500 / 250 / 750. It retains 764 training and 102
development-validation patients, including 32 validation pCR positives.
There is no independent test set. The trained task is a single T0-to-T3
interval; support for longitudinal inputs does not establish a trained or
validated T0-to-T1-to-T2-to-T3 model.

| Selected checkpoint | AUROC | AP | NLL | Accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| readout/best | 0.7096 | 0.6230 | 0.5640 | 74.51% | 31.25% | 94.29% |
| joint/best | 0.7136 | 0.6067 | 0.5774 | 73.53% | 46.88% | 85.71% |

Validation averages probabilities from four generated trajectories per patient,
with 20 Heun steps per trajectory. Accuracy, sensitivity, and specificity use
a fixed probability threshold of 0.5. AP is average precision, not a
trapezoidal PR area. Readout selects the minimum NLL; joint selects the minimum
`NLL + 0.05 * generation_objective`, rather than the highest AUROC.

Re-running the original validation function reproduced all five logged scalar
metrics exactly for the best and last readout/joint checkpoints. Representation
classification and late joint training show overfitting. Joint/last AUROC is
0.6772 with NLL 0.7519; longer training with the same setup is not supported by
these results. The existing T0-only branches slightly outperform their
generated-future counterparts on AUROC, AP, and NLL, so an incremental benefit
from generated T3 has not been demonstrated. These are branch diagnostics,
not independently trained ablations.

See the [completed training review](../results/breast/training_review_20260929/README.md),
[machine-readable summary](../results/breast/training_review_20260929/summary.json),
and [training diagnostics](../results/breast/training_review_20260929/training_diagnostics.png).
Intermediate representation/flow objectives are not pCR accuracy or decoded
MRI quality. The [2026-09-28 snapshot](../results/breast/training_snapshot.json)
is retained as a historical progress record.

The review includes reproducible audit scripts in `scripts/audit_training.py`,
`scripts/representation_components.py`, and `scripts/baseline_diagnostics.py`.
They require authorized private manifests, training logs, and checkpoints;
keep their inputs and raw reports outside tracked repository files. The
repository-level `tools/export_breast_review.py` exports only approved aggregate
fields. Install `python -m pip install -e '.[report]'` for the optional
Matplotlib report dependency.

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
