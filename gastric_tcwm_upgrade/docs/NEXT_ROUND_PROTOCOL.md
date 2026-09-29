# Next-round development protocol

Implementation and all training were completed in the original research tree.
This public snapshot contains the matching runtime source and configurations.
Historical reference is `ba33c542b3d48e874b26f06cefddd34258a90e03`.
No historical report, weights, cache or partition is overwritten.

## Endpoint and patients

Recorded recurrence binary status remains the endpoint. No survival horizon is
invented. Treatment descriptors, query interval and surgery are explicit
retrospective/hypothetical scenarios; no causal intervention is identified.
Four treatment tokens remain descriptor groups, not four temporal visits.

Reuse the existing original-train521 nested folds, seed20260928: inner train
277/277/278, selection70 each, outer174/174/173. Inner-train recurrence events
63/62/63. Original validation65/test65 remain redacted and unscored.
These outer folds were already used in development. Reuse is development
research, not independent confirmation; changing an initialization seed does
not create independent validation.

## Fixed-feature diagnosis

B0 is inner-training prevalence. B1 is clinical32. B2 adds the explicitly
declared treatment/query interval. B3 adds CT0. B4 additionally reads legal CT1.
For ranks8/32, each fold fits one shared CT0/CT1 projection using only inner
training scans, aggregates token mean/population SD, then fits patient-level
scales on that same training fold. Existing clinical/treatment encoder fit IDs
must match exactly. No outer-training refit with incompatible cached encoders.

Full-batch FP64 L-BFGS minimizes mean(BCE)+lambda/2*||w||^2 with unpenalized
intercept, maximum2000 iterations. Lambda grid: .001/.01/.1/1. Only converged
fits are eligible; selection uses inner-selection NLL. Publish all candidates,
convergence, NLL/Brier/AUROC/AP/calibration and summaries of paired patient-level
differences. Individual predictions and individual differences remain private.
The real run completed all96 lambda fits with maximum247 iterations and
maximum absolute final gradient below1e-7. Results remain developmental.

## Monte Carlo lock

Case-key MC derives each draw from seed, anonymous case key and draw index.
Keys stay outside model features. Consecutive antithetic pairs share one
independent simulation unit. Every patient's S0/S1 uses the same epsilon.

Before training, a fixed historical token_prior fold0 checkpoint was evaluated
on its70 inner-selection patients at K16/32/64 and MC seeds17/29/43. Antithetic
NLL SD was6.5463e-5/8.2333e-5/7.6293e-5. The prespecified trigger is K32 SD at
least half min_delta1e-4. Therefore this study uniformly locks K64.
K64 still has noise near min_delta; small NLL differences remain uncertain.
No candidate-specific MC seed selection. Training selection uses seed10017
(the existing seed+10000 convention); final reports uniformly use seed17/K64.

## Training matrix

All arms: seed17, H64/Z16, world2/surgery1/update1, Gaussian prior, residual
observation update, clinical anchor, dropout.3, residual_scale.25, batch16,
trainK4/evalK64, weight_decay.05, gradient_clip1, warmup0, stage weights
(.5,.5,0). Eligibility is unchanged and S2 stays in outputs and reports.
Auxiliary weights: prior_CT.1, posterior_CT.01, KL.02, pCR.2.

| Arm | Change |
| --- | --- |
| G0 | P0 corrections to token_prior; all parameter LR1e-4 |
| G1 | G0 with world LR2e-5; outcome/pCR-output LR1e-4 |
| G2 | G1 with observation_recon_weight.01 only |
| G3 | G1 with pooled shared outcome readout only |

P1 rank32 B4 vs B3 has positive AUROC direction in3/3 folds, OOF delta.0180,
95% interval[-.0110,.0497]. This is exploratory evidence sufficient to run
the small G2/G3 diagnostic ablations, not proof of CT benefit.
G2 assimilates legal CT1 and is not a future-prediction score. G3 changes the
readout architecture and is not an overall convex model.

Maximum1000 supervised/optimizer updates per arm/fold. Validate every50
supervised updates and once at a final budget boundary. Five consecutive
eligible validation cycles without an NLL improvement >1e-4 stop the arm.
The frozen initial clinical candidate competes before neural supervision.
Both last.pt recovery and best.pt selection survive a baseline win.
Every100 updates, log individual-loss gradient presence/norm by branch.
Every validation logs stage distributions, KL before/after free nats, active
units and state/readout differences. Record all actual and selected updates.

This step-based protocol is different from the historical80-epoch study.
Do not attribute the entire old/new score difference to one architecture term.
Compare G1-G0, G2-G1 and G3-G1 within this locked study.

## Conditional later work

Evaluate the historical token_prior forecast in a fixed train-fitted target
space using decoded MC features, train mean, copyCT0 and regularized linear
comparators. Do not equate decode(mean z) with mean(decode z). No tokenwise
cross-time spatial MSE or anatomical-registration claim.

Do not activate a larger backbone/Flow/network, new CT cache, pCR@CT1,
deterministic-center loss or staged schedule in this matrix. Seed29/43 repeat
training is conditional on consistent direction and must report every seed.
No full521 refit budget is chosen using the original65-person validation set.

## Commands

Run from `gastric_tcwm_upgrade` after installing its dependencies. Set
`GASTRIC_FOLDS` to the existing authorized nested-fold cache, `GASTRIC_NEXT_RUNS`
to a new private training directory, and `GASTRIC_NEXT_EVALUATION` to a new
private evaluation directory. Reuse the audited partitions; these commands do
not create a new independent validation cohort.

```bash
python scripts/run_next_round.py \
  --folds "$GASTRIC_FOLDS" \
  --out "$GASTRIC_NEXT_RUNS" \
  --cases G0 G1 G2 G3 --mc-samples 64
python scripts/evaluate_next_round.py \
  --folds "$GASTRIC_FOLDS" \
  --runs "$GASTRIC_NEXT_RUNS" \
  --out "$GASTRIC_NEXT_EVALUATION" \
  --cases G0 G1 G2 G3 --samples 64
```

The runner locks source hashes, effective configs and fold hashes before
launching. Use a fresh directory for changed contracts; unchanged interrupted
runs resume through last.pt. Patient-level artifacts remain private.
