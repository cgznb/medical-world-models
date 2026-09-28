# CT network development

The [full report](../../results/gastric/ct_optimization/CT_REPORT_ZH.md) records
18 nested-fold training runs and a fixed-budget refit on 521 training patients.
The refitted candidate was not promoted. The clinical-anchored repair remains
the retained research baseline; no trained weights are published.

## Implemented comparisons

`prior_ct_weight` directly supervises the token prior's forecast against CT1,
with a permutation-invariant feature objective. CT1 remains a target, not an S0
input. `token_control` and `token_prior` differ only by this loss, weighted 0.1.

The optional `predictive_ct` architecture fits a frozen CT PCA projection on
inner-training patients, predicts absolute future CT state, and uses a shared
linear risk readout with a learned reference transform. The complete network
remains nonconvex. `predictive_transition=false` provides direct-CT controls.
Direct/predictive comparisons also change conditioning, variance, and warmup;
they do not isolate generation alone. All forecasts are latent CT features,
not raw CT images.

## Protocol and results

Only the original 521 training patients enter three outer folds of 174/174/173.
Inner training uses 277/277/278 patients with 70 for epoch selection. Encoders,
normalizers, PCA, support statistics, and clinical anchors fit only inner
training rows. Original validation/test labels and eligibility are redacted in
fold caches; the evaluator processes only explicit outer IDs.

Six fixed candidates at seed 17 completed 510 total epochs and 9,180 updates.
The lowest OOF NLL selected `token_prior`: S1 AUROC 0.5989 versus clinical-only
0.5892. The paired bootstrap difference interval [-0.0055, 0.0253] crossed zero
and does not include all retraining/selection uncertainty. Direct CT1 forecast
R2 was negative relative to the training mean at both ranks, so reliable
individual future-state prediction has not been established.

Inner selected epochs 11/10/9 gave a fixed final budget of 10 epochs / 330
updates on all 521 training patients. No original-validation selection or
early stopping occurred. Final descriptive validation S0/S1 AUROC was
0.5252/0.5196, worse than the retained repair. The original test was not scored
again and no further configuration changes followed this result.

## Reproduction

Use the private adaptation output from [LOCAL_ADAPTATION.md](LOCAL_ADAPTATION.md)
and new output directories:

```bash
python scripts/prepare_ct_folds.py --data-dir "$GASTRIC_DATA_DIR" \
  --out artifacts/ct_folds --nested-selection
python scripts/run_ct_cv.py --folds artifacts/ct_folds --out artifacts/ct_runs
python scripts/evaluate_ct_cv.py --folds artifacts/ct_folds \
  --runs artifacts/ct_runs --out artifacts/ct_evaluation --device cuda \
  --cases token_control token_prior direct8 direct32 predictive8 predictive32
```

Fold preparation reads private adaptation provenance and refits the original
encoders, so it requires the original authorized artifacts and encoder source.
Treat generated fold caches, predictions, contracts, and weights as private.
Candidate configs are `configs/ct_cv_*.json`; the historical final budget is in
[final_config.json](../../results/gastric/ct_optimization/final_config.json).

The public aggregate reports preserve the weak/negative findings. Coarse
contextual ROIs, stomach fallback, and independent CT crops limit interpretation;
these observations motivate further representation checks but do not establish
the cause of poor accuracy.
