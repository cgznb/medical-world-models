# Gastric Development Study, 2026-09-29

All 12 G0-G3 fold runs completed successfully, with 5,400 actual supervised
updates. All selected models were neural models. Every run stopped through its
specified early-stopping rule before its 1,000-update maximum. Original and
public-copy engineering suites each passed 176 tests.

These are development results on the original training cohort: 521 patients,
118 recorded recurrence events, and three reused outer folds. The original
validation and test cohorts were not scored. Reuse of the outer folds prevents
interpreting these results as independent confirmation. Historical results in
the parent directory remain separate snapshots.

| Arm | Change | S1 AUROC | S1 NLL | Weighted NLL |
| --- | --- | ---: | ---: | ---: |
| Clinical anchor | Frozen clinical model, C=1 | 0.58925 | 0.54661 | 0.54661 |
| G0 | Training and model-selection repairs | 0.59635 | 0.54371 | 0.54387 |
| G1 | G0 with lower world-model learning rate | 0.59179 | 0.54669 | 0.54694 |
| G2 | G1 with observed CT1 reconstruction | 0.59019 | 0.54565 | 0.54607 |
| G3 | G1 with shared pooled readout | 0.59482 | 0.54954 | 0.54973 |

G0's S1 AUROC difference from the clinical anchor is 0.0071077, with a fixed-OOF
paired bootstrap 95% interval of [-0.0025476, 0.0166148]. Reliable incremental
benefit has not been established. G1-G3 also do not establish consistent benefit
over their specified reference arms. The previous research checkpoint was not
replaced, and additional training seeds were not triggered.

S0 uses clinical information, CT0 and the explicit treatment/time scenario.
S1 adds legally available observed CT1; S2 has no additional observation and
matches S1. Selection and final scoring use 64 antithetic MC draws and stage
weights (0.5, 0.5, 0). The training seed is 17, selection MC seed is 10017, and
final evaluation MC seed is 17. Fixed-OOF intervals exclude uncertainty from
retraining and candidate selection. The endpoint is recorded recurrence binary
status without a fixed prediction horizon, not a causal treatment effect.

## Files

- [summary.json](summary.json): all arms, fold counts, training/selection budgets,
  stage metrics, information sets and paired differences with confidence intervals.
- [training_status.json](training_status.json): 12 exit codes, actual and selected
  updates, early-stopping outcomes, optimizer groups and effective task counts.
- [diagnostic_baselines.json](diagnostic_baselines.json): B0-B4, ranks 8 and 32,
  all 96 regularized candidate fits, fold/pooled metrics and paired intervals.
  All candidate fits converged. B1 selects its regularization independently and
  is distinct from the G0-G3 frozen C=1 clinical anchor. B4 is an observed-CT1
  diagnostic, not an S0 deployment model.
- [forecast_prior.json](forecast_prior.json): prior forecast diagnostics for
  **historical token_prior checkpoints, before G0-G3**. The mean of decoded MC
  forecasts has skill versus the training CT1 mean of -0.78567 at rank 8 and -0.86138 at rank
  32. Marginal 90% feature coverage is approximately 0.18% and 0.12%, respectively,
  indicating a severely narrow predictive distribution. Ranks define different
  target spaces, so raw errors across ranks are not directly comparable.
- [mc_stability.json](mc_stability.json): the historical fold-0 inner-selection
  MC probe at K16/K32/K64 and seeds 17/29/43. K64 NLL SD is 0.00007629, still
  material relative to min_delta=0.0001. These seeds are simulation repetitions,
  not additional completed training runs.
- [dependency.json](dependency.json): aggregate inner-validation perturbations
  for all 12 selected models, zero-error information-boundary checks, and exact
  selected-model export/frozen-anchor checks. CT1 permutation worsened weighted
  NLL in 1/3, 1/3, 2/3 and 1/3 folds for G0-G3. Input dependence alone does not
  establish clinical benefit or causality.
- [engineering_checks.json](engineering_checks.json): original training smoke
  checks, gradient routing, resume/export agreement, original source hashes,
  and a separate aggregate result from testing the public copy.

## Publication Scope

The exporter uses explicit field allowlists. Reports contain cohort/fold
aggregates, algorithm settings, source hashes and projection/statistics
provenance. Patient identifiers, membership hashes/lists, individual
predictions/labels, private execution contracts, deployment paths, weights and
raw logs are excluded. All calibration bins are omitted because some original
bins contain a single patient; only cohort-level calibration summaries remain.

The raw reports remain in the research environment. Source hashes refer to the
original training source and are not a new claim that every publication file is
byte-identical. Publication adaptations are verified separately by tests.
To regenerate the aggregate JSON with authorized local reports:

```bash
python3 tools/export_gastric_next_round.py \
  --source "$GASTRIC_STUDY_ROOT" \
  --output results/gastric/next_round_20260929 \
  --publication-test-report "$GASTRIC_PUBLIC_TEST_XML"
```
