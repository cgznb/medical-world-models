# CT World Model Architecture Audit

Date: 2026-09-28. Scope: existing source and cache provenance; CPU probes only.
All numerical probes in this audit use training patients only. No validation or
test outcome scores were computed for these architecture probes. This public
report contains aggregate measurements. Original probe artifacts and private
cache manifests remain in the authorized research environment. Source line
numbers below refer to the audited pre-optimization implementation.

## 1. The CT objective does not directly supervise the forecast used at S0

`model.py:186` generates the S0 state from the prior. In contrast, lines 224-225
decode a posterior state that already read CT1. `losses.py:83` applies the CT
objective only to that posterior reconstruction. In a 16-patient training probe,
all 23 prior-pool/prior-parameter tensors have `grad=None` under the CT objective.
The prior receives indirect KL and outcome supervision; the shared deterministic
transition and decoder do receive reconstruction gradients. This is a VAE-style
design, not an automatic gradient bug, but the reported CT reconstruction is not
a direct forecast-quality measure.

The gap is real in the graph but is not, by itself, evidence of a large numerical
forecast failure: for the original selected model, posterior reconstruction loss
is 0.46321, prior decoding with the same sample is 0.47101, and copying CT0 scores
0.68926. The repaired anchored model gives 0.63308 versus 0.63309, respectively.
Thus the decoder improves over a copy on these training patients, while its
generative objective does not establish useful patient-specific future dynamics.

The pCR head reads `[initial, generated_prior_state, conditions]` directly
(`model.py:226-227`), not decoded CT1 or a forecast-validated representation. The
generated branch is a transformed version of baseline information. It is not
universally ignored: replacing generated tokens by baseline tokens changes the
original model's pCR probability by mean 0.01719; permuting them across patients
changes it by 0.03159. In the selected anchored repair, the same changes are only
0.0000276 and 0.0001219, so that checkpoint barely uses the neural future branch.
These perturbations measure dependence, not predictive value.

**Low-capacity change:** freeze a train-only CT projection, and directly supervise
the prior's Gaussian forecast of the next CT representation. Report prior
forecast NLL and compare against persistence, train mean, and a small linear
forecast. Keep the original posterior objective optional rather than using its
reconstruction metric as evidence that S0 predicts future CT.

## 2. The selected risk head suppresses much of the stage-update signal

The original architecture compresses CT1 into a global latent and injects it
with a 0.1 multiplier. The risk head then applies bidirectional fusion, seed
pooling, query attention and its final MLP (`model.py:32-46`,
`backbone.py:64-96`). On the same 16 training patients, the RMS S1-S0 difference
falls from 0.03309 at the reference input, to 0.01462 after fusion, 0.00530 after
pooling, 0.00301 after query attention, and 0.0000644 in logits. The final
probability difference is only 0.0000121 RMS.

The residual-observation repair successfully preserves an internal CT1 signal:
its reference, pooled and query RMS differences are 0.06917, 0.13309 and 0.05470.
Nevertheless, the selected anchored risk logits differ by only 0.0004337 before
the 0.25 residual multiplier; probability difference is 0.0000200 RMS. Therefore
the evidence does NOT support blaming LayerNorm alone. The selected readout and
its small learned residual also suppress signal, and a clinical anchor can mask
that lack of learned CT contribution.

**Low-capacity change:** use a shared linear risk readout of stable baseline CT
summary and `reference_summary - baseline_summary`. Use the predicted CT summary
at S0 and legal observed CT1 summary at S1/S2. Keep a direct-CT persistence
ablation with the same projection and readout, so a forecast must demonstrate
value beyond a simpler CT model. Penalize the readout explicitly instead of
adding more attention blocks. No stage-specific heads are needed.

## 3. The cache is a coarse contextual ROI, with lost physical geometry

The private source manifest is `${GASTRIC_CACHE_ROOT}/ct_manifest.json`.
Its preprocessing version is
`flare23-tumor-near40-margin30-stomachfallback40-min192-cube96-v1`.
It uses a frozen self-supervised SwinUNETR deepest 768-channel feature map;
`encoders/swinunetr.py:118` requests `normalize=True`. This is NOT a whole-scan,
non-ROI cache, and it is not an expert-verified tumor-only representation.

The source crop (`data/tumor_roi.py:312-351`) uses a gastric-associated tumor
candidate, or stomach fallback, with 30/40 mm context margins and at least
192 mm field of view. It resamples to 96 cubed, preserves tissue outside the mask,
and yields 3x3x3 deep features. Among the 1,042 scans belonging to the 521 current
training patients, 603 use tumor candidates and 439 use stomach fallback
(42.1%). Field of view is 192-320.875 mm, mean 209.651 mm. A grid cell therefore
spans roughly 64-106.96 mm. Fine lesion-local response can be diluted by context.

`generated700_data.py:132-139` orders feature tokens spatially but exports their
values without the original physical coordinates. The new cohort likewise
contains only the grid and masks. CT0/CT1 crops are selected independently, so
same-index feature subtraction would assume correspondence that the cache does
not establish. The existing set loss avoids this assumption, while spatial
convolutions still only see ordinal grid positions and not actual crop scale.

Normalization is not currently a numerical failure: training CT0/CT1 token
channel means are about zero, their channel SD is about 0.9998, and no channel
hits the model's 0.05 scale floor. Additional LayerNorm cannot recover spatial
detail already discarded by the frozen encoder and coarse crop.

**Low-capacity change:** fit a shared PCA basis on training CT0/CT1 patient mean
vectors, using feature scales estimated from those same patient means. Apply
that basis to tokens and aggregate mean/std summaries. This prioritizes between-
patient variation over within-scan spatial anatomy, without an index-registration
assumption.
Preserve crop field of view, fallback status and localization QC in a future
adapter revision; validate whether they carry acquisition/ROI confounding.
If local lesion signal is required, re-extract multiscale or mask-weighted
features from reviewed ROIs rather than deepening the current token network.

## Constraints For The Predictive Alternative

- Refit all projection and latent scaling buffers inside each training fold.
- A shared projection fitted on training CT0 and CT1 is allowed; validation/test
  images must not enter that fit.
- Give Gaussian variances a floor, initialize from training change variability,
  and average NLL over latent dimensions so rank does not change loss scale.
- Only CT0, baseline clinical variables and declared treatment scenario enter
  the forecast. CT1 enters only as a training target or a legal S1/S2 observation.
- pCR remains on the prior/preoperative path and cannot read actual CT1, surgery
  or labels during inference.
- Current factual surgery is homogeneous, so any present-only residual is an
  association component, not an identifiable causal surgery effect.
- Use train-only folds to compare clinical-only, direct CT and predictive CT.
  Improvement over a clinical anchor alone does not establish useful forecasting.
- Report absolute CT1 forecast error against a target mean fitted on inner-train
  patients, and against persistence. Delta R-squared can be inflated by the known
  negative-CT0 component and is not sufficient evidence of future prediction.

## Post-implementation Interpretation Limits

1. **A shared linear readout does not make the complete model convex.**
   `predictive.py:115-117` applies a learned present-only surgery residual before
   the endpoint head. For present surgery, let the residual be `A z + c`, and
   write the head weights on baseline and change as `a` and `b`. The resulting
   neural logit is `(a-b)^T x0 + b^T (I+A) z + b^T c + intercept`. Training jointly
   optimizes the product of `A` and `b`. `readout_l2` penalizes the head weights,
   but does not directly penalize the effective future coefficient `(I+A)^T b`.
   Increasing the surgery transform while reducing `b` can partly bypass that
   head penalty. AdamW also decays surgery weights, but this is not the same
   explicit regularization on effective regression coefficients. The appropriate
   description is "shared linear readout with a learned reference transform",
   not a convex or exact ridge-logistic fit. This is a mathematical limitation,
   not evidence that scale amplification actually caused the current results.
   In a read-only snapshot of available direct-model checkpoints, the measured
   effective/raw future coefficient norm ratio was approximately 0.957-1.018.

2. **Direct and predictive CT are architecture/training comparisons, not a pure
   single-variable test of generation.** Both use the same fold-local CT
   projection, clinical anchors, shared heads and legal S1 observations at a
   matching rank. However, the direct model omits the conditional transition and
   therefore does not use treatment descriptors or interval in its neural path.
   It centers its S0 state at CT0 with fixed measurement SD 0.05. The predictive
   model adds the 361-input condition path, an MLP, learned future-state variance,
   direct CT1 likelihood supervision, and five prior-only warmup epochs in the
   locked configurations. Differences may reflect these combined changes.
   They cannot be attributed solely to the utility of generated future CT.
   `token_control` versus `token_prior` is the deliberately matched single-change
   comparison of added direct prior CT reconstruction supervision. Ranks 8 and
   32 define different CT target spaces, so absolute CT MSE or skill scores must
   be compared against the corresponding same-rank persistence/mean baselines.
