# Next-round training implementation

Date: 2026-09-29. These changes were implemented and tested in the original
research training tree, then synchronized into this public source snapshot.
The audited previous publication is commit
`ba33c542b3d48e874b26f06cefddd34258a90e03`. Historical evidence remains unchanged;
new aggregate reports are in
[the dated results directory](../../results/gastric/next_round_20260929/README.md).
Patient data, checkpoints, raw execution contracts and local backup archives
remain in the private research environment.

## Implemented

- `config.py`, `losses.py`: strict stage_weights with historical(1,1,1) default;
  each patient contributes the weighted mean over valid stages, with zero-total
  patients skipped. Eligibility stays distinct. Disabled objectives never join
  the autograd graph. Legal CT1 reconstruction is an optional separate term.
- `training.py`: explicitly anchored models may select baseline_initial at
  zero neural supervision; warmup records a fresh head-training start. best.pt
  and last.pt have different roles. Exports disclose clinical_baseline/neural,
  selected/actual steps, task targets/updates and patient exposure. Added
  supervised-step validation, early stop and exact mid-epoch recovery, LR
  manifests and periodic individual-loss branch gradient probes.
- `monte_carlo.py`, `evaluation.py`, `inference.py`: fixed per-case/per-draw
  random numbers, optional antithetic pairs and matching inference contract.
  Weighted selection_nll and legacy_three_stage_nll are reported separately.
  Old model state dictionaries load strictly; no silent strict=False migration.
- `model.py`, `predictive.py`, `belief.py`: explicit validated epsilon and
  diagnostic state outputs. Token-world updated_features decodes the legal
  S1 pre-surgery state. The extra observation reconstruction path adds no model
  parameters and defaults off. G3 optionally uses a shared pooled readout;
  attention readout stays the backwards-compatible default.
- `diagnostics.py`: anchor/residual/probability distributions, prior/posterior
  statistics, raw/free-nats KL, active units, state/readout differences, fixed
  noise CT0/CT1/paired/scenario dependency probes and prefix boundary checks.
- `diagnostic_baselines.py`: patient-level frozen shared CT features and true
  deterministic FP64 regularized logistic B0-B4. Separate names preserve all
  historical direct_ct definitions and tables.
- `configs/next/G0-G3.json`, `scripts/run_next_round.py` and
  `scripts/evaluate_next_round.py`: source/data/config-locked study runner and
  development OOF report. Existing CLI and CV evaluation honor new bundle
  policies while maintaining historical defaults for old bundles.

## Evidence and limitations

The endpoint, original partitions, encoder and hidden/latent dimensions remain
unchanged. Historical published evidence is immutable: repaired test AUROC
.6240 vs clinical.6253; old OOF clinical.5892, token_control S1.5789,
token_prior S1.5989; prior-vs-clinical delta.00963 with approximate CI
[-.00547,.02526]; CT1 permutation approximately.5980. Prior predictive8/32
CT1 R2(-.233/-.500) are not token_prior forecast scores. Historical full521
refit validation S1 AUROC.5196 is not used for this study's selection.

P1 diagnostics and MC probe aggregates are published in the dated results directory.
Rank32 B4-B3 OOF AUROC delta.0180 remains uncertain despite positive direction
in3/3 folds; the confidence interval crosses zero. This motivates exploratory
single-change ablations, not a claim of reliable clinical improvement.

The MC probe caused a uniform K64 lock. Its NLL SD remains near min_delta;
small changes must not be overinterpreted. Bootstrap conditions on fixed fits
and does not include all retraining or candidate-selection uncertainty.

No larger Flow/backbone, survival relabeling, pCR@CT1, staged schedule, CT
recaching or deterministic-center objective is activated. Such changes require
separate evidence and names. Reconstruction quality is not prognostic benefit.
The existing research model is not replaced merely because a candidate has a
wider prediction range or stronger input sensitivity.

See [the protocol](NEXT_ROUND_PROTOCOL.md) for fixed comparisons and commands,
[the results](NEXT_ROUND_RESULTS.md) for actual training counts and selected
candidates, and the dated public engineering summary for verification.

New inference bundles record `evaluation_config` including the selection seed
and MC count. Default independent inference honors those values. Under the new
`case_key` policy, provide one stable anonymous key per patient as the query's
top-level `case_keys` list, outside `tensors`; for existing cohort rows use
`monte_carlo.cohort_case_keys(cohort, rows)`. These keys only choose evaluation
noise and never enter the predictive features. Historical bundles keep their
old defaults. Private source manifests bind the executed training runs; this
Git commit fixes the corresponding public source and aggregate reports.
