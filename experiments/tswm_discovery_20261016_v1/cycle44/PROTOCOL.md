# C44 — Trained world-law and ordinary query-estimation screen

Frozen before new comparative fits or evaluation. Overall goal remains a reusable time-series world-model framework; this controlled screen decides whether the coarse-law/query route requires a new mechanism. It is a cross-task adaptation/new experiment, not a reproduction of climate headline numbers. No candidate method is being renamed as a baseline.

## Claim–evidence contract

|Question|Required evidence|Contenders|Rejection condition|
|---|---|---|---|
|Does ordinary multilevel estimation already provide useful path queries?|same fitted law, independent high-sample reference, actual variance/error and total cost|MC, antithetic MC, common-RNG MLMC and antithetic MLMC|ordinary controls absorb utility; no special hierarchy justified|
|Is a cheap sampler already sufficient?|independent future quality and truth-based function error/cost|probabilistic AR, learned LGSSM, native SPF, ordinary condition-trainable SPF|cheap model dominates useful quality/cost|
|Is apparent benefit just model bias or a trivial mean query?|separate model bias, reference uncertainty and estimator error; fixed nontrivial functions|known-process reference; full law energy; all fixed functions|bias dominates or only linear mean improves|

## Fixed data and information

Two independent-trajectory carriers: stable linear-Gaussian latent dynamics and stable nonlinear four-state dynamics (including a bistable component). Shared 64-dimensional noisy linear observation, represented as8×8 for SPF. The four latent states and exact parameters are never supplied to any learner. Initial state is N(0,I), followed by100 burn transitions. Observe history8, forecast16. Each trajectory/split is disjoint. TRAIN2048, VAL128, sealed EVAL64 trajectories per carrier. All normalization, PCA, rank selection, thresholds and functional scales use TRAIN only. All learners receive the same8 observed history fields; there are no future-known observations or true states in their inputs.

Observation basis/parameters are stored separately for truth evaluation. Learned PCA uses TRAIN empirical covariance, noise estimated from bottom32 eigenvalues and rank = eigenvalues above1.5×noise, clipped1–8. Its whitening is a learned preprocessing choice for the ordinary models, not access to the true latent basis. Both native variants receive full raw normalized history stacked as8 input channels, repeated at each forecast timescale; native API names decadal/yearly/monthly have no climate meaning here. Coarse target is the16-step temporal mean. The fixed coarse sampler for MLMC is the native yearly output, not selected after effects.

## Complete fixed methods

1. `spf_native`: author core/collator/scheduler and sampler qualified in C43, fixed commit a7624ae25b47b474d5a31ca22b3ac6e6907795dd. Small transfer configuration: inner128, depth2,4heads, patch2, native SDPA/sinusoidal positions, temporal1/1/16,8 history channels,8 Euler steps/stage. C43 optional-return metadata fix retained.
2. `spf_condition_trainable`: same model/initial weights/data/noise schedule; remove only no_grad around condition-token packing. This is an ordinary implementation control, not a proposed method. Forward values must match before training, and condition gradients must be shown. Original source remains unchanged.
3. `ar_mixture`: TRAIN-PCA/whitened observed coordinates, two-layer GRU256,4-component full-covariance Gaussian next-observation mixture; estimated independent orthogonal residual observation noise. Teacher-force all23 transitions of each TRAIN trajectory. All history observations enter through learned PCA; no true latent labels.
4. `lgssm`: same learned coordinates, jointly optimized transition, process covariance, initial covariance and isotropic physical observation noise, using exact Kalman likelihood plus orthogonal-residual likelihood. Initialize from TRAIN moments and denoised lag covariance. No oracle matrix is given. Transition operator norm capped1.05 after updates for finite rollout stability.

Two initialization seeds2026441/2026442 for each carrier/method:16 fresh fits. Common precomputed TRAIN minibatch indices, batch24. Paired SPF versions have identical initial numerical parameters and random training schedules. Cross-family objectives/amount of per-update supervision differ; do not claim exact matched compute or capacity. Report nominal/gradient-active parameters, actual update/time/memory, selected checkpoint and cap/plateau status. MC strategies compare the **same selected weights**, so their training cost is identical. Cross-model comparisons are quality/cost controls, not architecture-superiority claims.

## Training/selection and exclusion rules

SPF and AR AdamW3e-4/weight decay1e-4/clip1; min4000,max10000 updates, evaluate every1000. LGSSM Adam5e-3/clip5, min1000,max3000, evaluate every500. Half-cosine learning rate to10% across maximum updates. After minimum, stop on two successive checkpoints with less than0.2% best-relative improvement. Select minimum fine joint future energy score using the fixed first16 VAL origins and32 fixed-iid draws; initial checkpoint eligible. Formula is U-statistic energy/√1024, including off-diagonal pair correction. Fixed sampling randomness reduces selection noise. No EVAL or truth reference participates in fitting, stopping or model selection. Qualification weights from C43 and C44 are discarded.

Before formal fits, qualify the changed native history-channel/condition-gradient paths, mixture log-density against an independent Gaussian implementation, exact Kalman likelihood against independent NumPy recursion, and finite samples. These are numerical checks, not performance evidence. New code errors must be fixed with versioned files and preserved failures before effect claims.

## Fixed scored query protocol (after all selections)

Use the independent first4 EVAL histories as query conditions, all64 EVAL trajectories for ordinary full-law quality. Functions are (1) global trajectory mean, (2) smooth maximum of spatial means with tau=.25 and log-H correction, (3) mean squared temporal increment, (4) occupancy above TRAIN pointwise90th percentile, (5) indicator that a spatial-mean path crosses its TRAIN90th-percentile threshold. Normalize function units using TRAIN empirical function standard deviations, floor1e-3. Function1 is a trivial positive control;3 is a negative control for a constant coarse path. Report all five; success cannot depend only on1.

Coarse proxy repeats a generated temporal-mean field16 times. Ordinary MLMC uses independently generated coarse draws plus paired fine-minus-coarse draws with the **same coarse marginal**. Common RNG is a valid coupling even if native coarse/fine intermediate states differ. The learned LGSSM may use its analytic coarse Gaussian marginal and fine temporal mean for a stronger ordinary coupling. AR without a cheap coarse law remains a strong MC/antithetic contender. Antithetic generation flips all Gaussian base variables (and complements mixture uniforms for AR); each marginal must remain unchanged.

Pilot allocation and any choice among MC/antithetic/MLMC policies use VAL histories only, actual measured sampling costs, and frozen pilot variances. Round/overhead rules must be recorded before EVAL, and actual wall-clock cost must accompany variance-based predictions. Include sampling, coarse calls, function evaluation, allocation pilot and amortization; distinguish per-query marginal cost from shared multi-query cost. Target fine-equivalent sample budgets128/512/2048, at least16 independent estimator replicates. Preserve raw function draws and allocation decisions.

Within-fitted-law high-sample reference and known-process truth are separate. The Gaussian carrier has exact Kalman conditional sampling; the nonlinear carrier needs particle filtering under the known process using only observed history, with a particle-count/repetition convergence audit. Do not condition truth on the hidden state that generated the EVAL trajectory. Record reference Monte Carlo/particle uncertainty. No new method-quality conclusion is allowed until both references and independent score verification exist.

## Decision and limits

The route merits further framework development only if a concrete capability remains beyond strongest ordinary controls, with useful gains on at least two nontrivial function families without a worse fine law. At this stage no global novelty, real-domain benefit or conference readiness is established. If ordinary controls suffice, do not add a consistency loss to rescue the route. Keep observation/dynamics uncertainty-interface learning as a distinct fallback; generic SSM expressivity alone does not invalidate a new learning recipe.
