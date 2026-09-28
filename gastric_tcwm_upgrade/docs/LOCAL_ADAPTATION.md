# Generated651 data adaptation

This document describes the private-cohort adaptation completed on 2026-09-28.
The public repository contains no source patient data or fitted feature caches.
Later repair and CT studies are described in [LOCAL_REPAIR.md](LOCAL_REPAIR.md)
and [CT_OPTIMIZATION.md](CT_OPTIMIZATION.md).

## Inputs and isolation

The existing feature pool supplies both CT grids, each `[651,27,768]`, clinical
rows, treatment rows, intervals, and binary labels. Event V2 surgery artifacts
must identify the same pool. The original seed-17 split identifies those event
artifacts; the adapter checks the complete provenance chain and preserves
patient membership. It does not silently substitute a different experiment's
partition.

Clinical and named-treatment encoders, normalization, and treatment-support
statistics are fitted only to the 521 training patients. The adapter checks
CT tensors, intervals, labels, and masks against the source artifacts. Exact
patient lists, input hashes, and source paths remain in private run contracts.

| Partition | Patients | Recurrence positive | pCR positive |
| --- | ---: | ---: | ---: |
| Training | 521 | 118 | 101 |
| Validation | 65 | 14 | 13 |
| Test | 65 | 15 | 13 |

Both CTs and binary labels are available for all patients, and all received
surgery. A missing BMI retains the original encoder's train-fitted handling.
Recorded CT intervals are not survival times. Out-of-support treatment and
interval queries are flagged; no outcome labels or intervals are fabricated.

## Prepare and run

Install the package, then set `GASTRIC_POOL`, `GASTRIC_EVENTS`, `GASTRIC_SPLIT`,
`GASTRIC_LEGACY_SRC`, and `GASTRIC_DATA_DIR` to authorized private paths. The
legacy-source variable points to the original repository's `src` directory.

```bash
python scripts/prepare_local.py \
  --pool "$GASTRIC_POOL" --events "$GASTRIC_EVENTS" \
  --split "$GASTRIC_SPLIT" --legacy-source "$GASTRIC_LEGACY_SRC" \
  --out "$GASTRIC_DATA_DIR"
python scripts/run_local.py --data-dir "$GASTRIC_DATA_DIR" \
  --config configs/local_binary_gaussian.json --out artifacts/initial_adaptation
```

The initial H128/Z32 model used batch 32, 15 warmup epochs, and patience 25.
It completed 42 epochs; validation NLL selected epoch 17. Validation S0/S1
AUROC was 0.3655/0.3641. This established execution and data compatibility, not
clinical usefulness. The test partition was not scored at that initial stage;
it was scored once later after the repair model was locked.

The initial model is a historical control. The repaired configuration is
`configs/repair_anchored.json`. All generated data/contracts, checkpoints,
predictions, and raw logs should remain outside the public repository.

## Limits

No verified event/censoring dates are available in these caches, so the clinical
experiments use binary outcomes. Actual treatment summaries, actual CT intervals,
and surgery are retrospective scenario inputs. They do not establish causal
effects or prospectively available treatment plans. CT1 is legal only from S1;
without new postoperative observations, S1 and S2 use the same information.
