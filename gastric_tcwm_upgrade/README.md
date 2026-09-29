# Gastric TCWM upgrade

Treatment-contextualized stochastic landmark prediction for the Generated651
gastric cancer project. This standalone extension includes data adapters,
training, inference, strict patient-split contracts, clinical baselines, CT
development experiments, and automated tests.

The reviewed base is
`cgznb/gastric-multistage-generated651@aec8cbe08157687fd447594c8a9406d6ee5a464d`.
This directory provides the upgrade path, not every historical module from that
repository. See [NOTICE.md](NOTICE.md) for attribution and licensing boundaries.

## Results and scope

The clinical-anchored repair reached recurrence AUROC 0.6232 on the original
validation partition and 0.6240 on the internal test partition scored once
after locking. Clinical-only AUROC was 0.6218 and 0.6253, respectively: reliable
incremental benefit from the CT/world model has not been demonstrated.

Later development used only the original 521 training patients in three outer
folds with separate inner epoch selection. The best of six fixed neural
candidates had OOF S1 AUROC 0.5989 versus 0.5892 for clinical-only; the paired
95% interval for their difference crossed zero. Its fixed-budget refit had
original-validation S1 AUROC 0.5196 and was not promoted. The original test
partition was not scored again during this CT study.

The 2026-09-29 update adds explicit stage weights, an eligible zero-step
clinical candidate, stable per-case Monte Carlo noise, branch diagnostics,
observed-CT reconstruction and an optional pooled outcome readout. All four
G0-G3 candidates completed three folds: 12 runs and 5,400 supervised updates.
G0 OOF S1 AUROC was 0.5964 versus clinical 0.5892; the paired difference interval
[-0.0025, 0.0166] crossed zero. None of the added ablations established a stable
benefit, and the retained research checkpoint was not replaced. No original
validation/test patients were rescored. This reuses development OOF folds and
does not establish independent confirmation.

- [Repair report and aggregate results](../results/gastric/repair/REPAIR_REPORT_ZH.md)
- [CT development report and negative findings](../results/gastric/ct_optimization/CT_REPORT_ZH.md)
- [Latest aggregate results](../results/gastric/next_round_20260929/README.md), [training report](docs/NEXT_ROUND_RESULTS.md), and [fixed-feature/forecast diagnostics](docs/NEXT_ROUND_DIAGNOSTICS.md)
- [Next-round implementation](docs/NEXT_ROUND_CHANGES.md) and [protocol/reproduction](docs/NEXT_ROUND_PROTOCOL.md)
- [Data adaptation](docs/LOCAL_ADAPTATION.md), [repair reproduction](docs/LOCAL_REPAIR.md), and [CT reproduction](docs/CT_OPTIMIZATION.md)
- [Algorithm](docs/ALGORITHM_ZH.md), [data schema](docs/DATA_SCHEMA.md), and [source audit](docs/SOURCE_AUDIT.md)

Public files contain source, configurations, tests, and aggregate results.
Patient data, caches, identifiers, split lists, individual predictions, model
weights, private contracts, and raw execution logs are excluded. Reproducing the
clinical metrics requires authorized access to the original private cohort and
its fixed split. The synthetic workflow below verifies engineering execution
only and is not clinical validation.

## Install and test

Python 3.11 or later is required. Install PyTorch for the intended CPU/CUDA
environment, then run from this directory:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The latest public-copy test result is recorded as aggregate counts in
[engineering_checks.json](../results/gastric/next_round_20260929/engineering_checks.json).
The [older test summary](../results/gastric/test_summary.json) remains a historical record.

## Synthetic execution check

Use a new output directory for each new configuration. Existing runs can only
resume with the same data, configuration, and source contract.

```bash
python scripts/run_tcwm.py --threads 1 synth \
  --out artifacts/toy --n 64 --image-dim 32
python scripts/run_tcwm.py --threads 1 train \
  --data artifacts/toy/cohort.pt --split artifacts/toy/split.json \
  --config configs/smoke.json --out artifacts/toy/run
python scripts/run_tcwm.py --threads 1 evaluate \
  --data artifacts/toy/cohort.pt --split artifacts/toy/split.json \
  --run artifacts/toy/run --role test --samples 8
python scripts/run_tcwm.py --threads 1 predict \
  --bundle artifacts/toy/run/inference.pt --query artifacts/toy/query.pt \
  --stage 0 --samples 8 --out artifacts/toy/s0.json
```

`configs/smoke_flow.json` tests the optional Flow prior.
`scripts/run_validation.sh` executes synthetic Gaussian, Flow, and competing-risk
workflows. Its generated artifacts and logs are private execution outputs and
are not part of the published clinical results.

## Private cohort adaptation

Set the following variables to authorized private input/output locations:
`GASTRIC_POOL`, `GASTRIC_EVENTS`, `GASTRIC_SPLIT`, `GASTRIC_LEGACY_SRC`, and
`GASTRIC_DATA_DIR`. `GASTRIC_LEGACY_SRC` is the original repository's `src`
directory. The adapter verifies pool/event/split provenance and refits clinical
and treatment encoders using training patients only.

```bash
python scripts/prepare_local.py \
  --pool "$GASTRIC_POOL" --events "$GASTRIC_EVENTS" \
  --split "$GASTRIC_SPLIT" --legacy-source "$GASTRIC_LEGACY_SRC" \
  --out "$GASTRIC_DATA_DIR"
python scripts/run_local.py --data-dir "$GASTRIC_DATA_DIR" \
  --config configs/repair_anchored.json --out artifacts/repair_reproduction
```

The runner evaluates validation only. The existing test partition has a prior
development history and a recorded post-lock evaluation; it must not become a
repeated model-selection set. The original legacy source and clinical cohort
are not bundled with this extension.

## Interpretation and optional modes

The current private cache supports recorded binary recurrence/metastasis and
auxiliary gastric pCR, not fixed-horizon survival probabilities. Actual
treatment summaries and CT intervals are retrospective scenarios; predictions
are associations, not causal treatment effects or treatment recommendations.
The cohort is internal and previously used for development, not an external
validation cohort. With no new postoperative observation, equal S1/S2 outputs
are expected. CT1 never enters S0 or the pCR inference path.

Gaussian and Flow configurations, optional survival/competing-risk endpoints,
and postoperative observation inputs are available in the implementation.
Survival requires verified event/censoring times and stage eligibility; see
[the schema](docs/DATA_SCHEMA.md) and the empty
[follow-up template](docs/followup_template.csv). Do not infer follow-up times
from CT intervals or substitute synthetic labels.

The optional installer adds the extension to a pinned original checkout:

```bash
python scripts/install_into_repo.py --repo "$GASTRIC_REPO"
```

`scripts/build_full_repository.py` can reconstruct a combined checkout from
the pinned public original. Neither tool uploads data or pushes Git changes.
