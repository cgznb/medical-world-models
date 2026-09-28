# Gastric repair study, declared before training

Only the existing 521 training and 65 validation patients are used for this
development study. Test patients are never scored by this runner. The patient
split and train-fitted encoders are unchanged. All historical outputs remain.

Every case uses batch 8, no endpoint warmup, at most 5,280 optimizer steps / 80
epochs, patience 15, and validation mean-prefix NLL for checkpoint selection.
The step budget is an upper limit; early stopping can use fewer steps. Counts
and selected supervised updates are recorded. No AUROC-based cherry-picking.

The following cases are fixed before scoring:

1. schedule: original H128 architecture/loss/LR, corrected training schedule.
2. compact: H64/Z16, world depth 2, surgery/update/readout depth 1, four readout
   slots, dropout 0.3, LR 1e-4, weight decay 0.05. This is a capacity and
   regularization bundle, not evidence about each change individually.
3. aux_balance: compact with CT feature loss weight reduced from 0.1 to 0.01.
4. observation: aux_balance with legal CT1 cross-attention residual update;
   no voxel correspondence, no CT1 at S0, no future observations for pCR.
5. anchored: observation with frozen training-only clinical32 logistic logits
   for recurrence and pCR, plus zero-initialized world-model logit residuals
   scaled by 0.25. Fixed logistic C=1, unweighted BCE, no validation fitting.

The five primary cases use initialization seed 17. Anchored also uses seeds
29 and 43 with the SAME patient split to quantify optimization variation.
Seeds are not new patient folds. Report all three, never select the best seed.
These are development comparisons, not a claim of independent validation.

Checkpoint scoring uses NLL, with AUROC, AP, Brier, probability spread, pCR,
train/validation gaps, and CT1 sensitivity reported as secondary diagnostics.
Training and validation use no synthetic labels, no altered-treatment outcome
targets, and no invented survival times or postoperative observations.

The complete anchored candidate is the proposed repair. Whether its learned
CT/world component improves beyond clinical-only must be reported explicitly;
an improvement over the broken original alone does not establish CT benefit.
