# Gastric repair and reproduction

The retained research configuration is
[repair_anchored.json](../configs/repair_anchored.json). Its trained weights are
private. See the [full repair report](../../results/gastric/repair/REPAIR_REPORT_ZH.md)
and [declared protocol](../../results/gastric/repair/PROTOCOL.md).

## Changes and findings

Disabled objectives no longer create gradients or optimizer updates for inactive
heads. Explicit step budgets and supervised-update counters make warmup and
batch-size effects auditable. The compact H64/Z16 network uses legal CT1 residual
updates and a frozen clinical logistic baseline fitted only to training patients;
the neural path learns a scaled residual on top of the clinical logits.

The fixed primary seed was 17. Seeds 29 and 43 repeated initialization on the
same patient split and were not independent folds or candidates for best-seed
selection. Seven declared runs completed. The primary model's validation/test
recurrence AUROC was 0.6232/0.6240, compared with 0.6218/0.6253 for clinical-only.
The improvement over the initial adaptation is mainly clinical; CT/world-model
incremental value remains unproven. Later neural training still degraded
validation quality, and all three anchored checkpoints were selected after
the first neural epoch.

The original internal test partition was scored once after locking the model.
It has 65 patients and prior cohort development history. It is not external
validation and must not be reused for repeated model selection.

## Reproduction

Prepare authorized private inputs as described in
[LOCAL_ADAPTATION.md](LOCAL_ADAPTATION.md). Use a new output directory:

```bash
python scripts/run_local.py --data-dir "$GASTRIC_DATA_DIR" \
  --config configs/repair_anchored.json --out artifacts/repair_reproduction
python scripts/run_repair_study.py --data-dir "$GASTRIC_DATA_DIR" \
  --out artifacts/repair_comparisons
```

The runner evaluates validation only. A stopped run may resume only under its
unchanged data, configuration, and source contract; a new configuration requires
a new directory. Older inference exports remain supported, but old training
runs require their matching archived source for recovery.

Published evidence contains aggregate development and locked-test metrics,
protocols, and figures. Weights, patient contracts, raw JUnit reports, source
archives, and individual predictions are excluded. The subsequent
[CT study](CT_OPTIMIZATION.md) did not rescore the original test partition and
did not promote its new candidate.
