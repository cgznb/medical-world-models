# CT network development protocol

Declared before the neural comparisons, 2026-09-28. The prior 65-patient test
set has already been scored; it will not be scored or used for model selection
again in this study. The original 65-patient validation set is also excluded
from the development comparisons.

## Patients and fitting

Only the original 521 training patients enter three outer stratified folds,
seed 20260928. Each outer training partition is split 80/20 into training and
epoch-selection patients. Outer-held patients do not select epochs. Clinical
and treatment encoders, image normalization, PCA, clinical logistic anchors,
and treatment support are all fitted to the inner training patients only.
The original validation/test 130 patients have labels and eligibility masks
redacted in the new fold caches. Separate metadata identifies the outer-held
patients that may be scored by the dedicated CV evaluator.

Outer out-of-fold predictions are development scores used to compare these
six fixed candidates, not independent confirmation after choosing one.
Clinical-only predictions from the same fitted training rows are a required
comparator. All stages retain legal information boundaries.

## Fixed comparisons

1. token_control: previous compact clinical-anchored token world network.
2. token_prior: same network, plus weight 0.1 direct prior CT feature-set loss.
   It supervises the S0 forecast without letting the forecast read CT1.
3. direct8: frozen rank-8 patient-mean CT PCA, mean/std pooled latent states,
   small shared regularized risk readout, and real CT1 update. No learned future
   transition; S0 uses CT0. This is the direct CT control, not a world-model claim.
4. direct32: same direct CT control, with rank 32, so the effect of forecasting
   can be compared at both predeclared representation ranks.
5. predictive8: same CT representation/readout, conditional Gaussian forecast
   of the actual CT1 latent state, direct future-state NLL with weight 0.2.
6. predictive32: same as predictive8, with a predeclared rank of 32 motivated
   by the training-only representation probes.

All neural runs use initialization seed 17, batch 16, at most 80 epochs / 1,440
optimizer updates, and patience 20 based on inner-selection mean-prefix NLL.
Predictive variants have 5 prior-only warmup epochs; disabled risk heads retain
no optimizer state during warmup. Direct/control variants start supervision
immediately. This difference and actual update counts must be reported.

Predictive/direct readouts use LR 1e-3 and explicit squared-weight penalty 0.05.
The transition uses LR 5e-4 and weight decay 0.01. Candidate details are fixed in
`configs/ct_cv_*.json`. These are architectural/training bundles; only
token_control versus token_prior isolates direct forecast supervision exactly.

## Decision and interpretation

Report every candidate's outer OOF S0/S1 NLL, AUROC, AP, pCR, and comparison
with clinical-only. Choose the lowest mean-prefix OOF NLL neural candidate for
a final fit, while explicitly reporting whether it beats clinical-only. Do
not select the best seed. Final epoch budget is the rounded median of the
candidate's three inner-selected epoch counts; fit all original 521 patients
for that fixed budget, without selecting on the original 65-person validation.
That original validation may only be reported descriptively after the choice.
No further original test scoring.

Future CT forecasting must be assessed in absolute CT1 latent space against
both the training CT1 mean and copying CT0. A good delta reconstruction score
alone is insufficient: CT1-CT0 contains the known negative CT0 term. CT1 shuffle
sensitivity demonstrates use, not useful or causal information.

Existing cached CTs are tumor-candidate/stomach fallback crops, not precisely
masked lesion voxels. Approximately 42% of training scans used stomach fallback;
different scan crops have no verified token-level anatomical registration.
Do not replace set/state supervision with spatially paired voxel targets.
