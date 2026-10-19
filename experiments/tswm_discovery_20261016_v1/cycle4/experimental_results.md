# Cycle4 experimental evidence

Overall discovery objective: ACTIVE, NOT ACHIEVED. No accepted research direction.
All measured values below are from server execution. Prior cycles are preserved.

## A. Native MotionJEPA/LeWM objective adapter pilot: six fits

Unchanged official `train_step` and SIGReg, linear32->2 observation/difference
adapters, causal3-step linear predictor; original512-wide motion inverse head.
This is a new controlled objective audit, not a native visual-paper reproduction.
Clean physical-state NMSE was approximately8e-15 for all three configurations.
With30 independent observation nuisance dimensions, 2000-step NMSE was
0.497659(default LeWM),0.533266(matched static weight),0.544318(MotionJEPA).
Motion-minus-matched gap=0.011052468, below the frozen0.10 gate.
Physical signal directions are noise-free in this condition; the separate
`sensor_noise` generator branch was not run. Do not label this raw sensor noise.

## B. One fixed attribution extension: five fits

| Configuration | Step2000 NMSE | Step4000 NMSE | Step8000 NMSE |
|---|---:|---:|---:|
| lewm_native_weight_random | 0.497659 | 0.000378 | 0.000123 |
| lewm_matched_static_random | 0.533266 | 0.499065 | 0.478940 |
| motion_random | 0.544318 | 0.544892 | 0.135551 |
| lewm_matched_static_trainingCCA | 0.000651 | 0.000343 | 0.000283 |
| motion_trainingCCA | 0.116285 | 0.093504 | 0.085087 |

Same seed41, cohort9051, independent probe9052 and development9053. All final
checkpoints are fixed-step, without state-based selection. Training-only CCA
initialization uses no state labels. Three random-initialization2000-step prefixes
match every pilot state tensor exactly, maximum difference0. Default LeWM is
repaired by budget alone; matched-static LeWM is repaired by ordinary CCA.
Motion has a remaining initialization-sensitive deficit in these adapters.
This neither proves a native architecture failure nor passes the original gate.

## C. Native VJEPA component paths: three initializations, no training

All three native CNN forwards accept64x64 inputs. With the unchanged exp10 loss,
encoder/mean-predictor gradients equal VICReg-only gradients exactly in all three
checks; the variance head has a nonzero gradient. Perturbing variance-head
parameters leaves native mean-only CEM actions unchanged under identical RNG.
Original linear code: varying target-KL beta0->1 does not change encoder gradients
because target tensors are detached; changing the supplied future probe argument
does not change the result because it is unused. No future-probe leakage found.
Selected unchanged AST bodies are executed to avoid notebook/Colab side effects.
These are source-path facts, not learned calibration or planning-performance results.

## D. Risk-state compression: nine new controlled fits plus three moment controls

| Configuration | Risk-driver probe NMSE | Conditional Gaussian KL | Conditional mean MSE | Policy regret |
|---|---:|---:|---:|---:|
| detached | 0.948990 +/- 0.087645 | 0.060029 +/- 0.003472 | 0.001408 +/- 0.000027 | 0.014649 +/- 0.001699 |
| attached | 0.183615 +/- 0.295364 | 0.016468 +/- 0.019485 | 0.001404 +/- 0.000035 | 0.003650 +/- 0.004713 |
| full_observation | 0.000000 +/- 0.000000 | 0.332513 +/- 0.019181 | 0.001439 +/- 0.000000 | 0.020107 +/- 0.001247 |
| moment_control | 0.039036 +/- 0.000000 | 0.006788 +/- 0.000187 | 0.000979 +/- 0.000000 | 0.001368 +/- 0.000080 |

Numbers are means +/- sample SD across three training initializations on ONE
fixed cohort, not independent dataset replications or confidence intervals.
Matched detached/attached variants each have4549 parameters, identical inputs,
training pairs, initialization, minibatches, heads, optimizer and8000updates.
Only variance-to-encoder gradient differs. Mean residual is detached inside the
variance term for both variants. Full-observation control has6435parameters and
is an informational control, not a capacity-matched competitor.

Process: y_next=.8y+.3a+sqrt(.27(1+.8a*tanh(v)))*eta; v_next iid standard normal.
Randomized +/-1 actions; observations are an unknown orthogonal mixing of y,v
and30 iid nuisances.100burn-in steps precede sampled transitions. Targets project
future observations onto an action-response direction fitted by TRAIN-only OLS.
No physical labels train the encoder, heads or moment projection. Physical labels
fit only a frozen diagnostic probe after training and define evaluation truth.
Policy regret is exact expected quadratic next-state cost in this declared SCM.
Tail Brier refers to |projected future|>1.5. This is not an observational causal claim.

Frozen attached-repair gate passed all seeds: **False**. Per-seed details are
`outputs/risk_state_pilot/analysis.json`. In particular seed43 is not discarded.
Ordinary cross-moment dimension reduction is an existing-method control, not
claimed SAVE/GSAVE software reproduction. It adds a reported training-statistic
pass and freezes64encoder coefficients; its4485head parameters receive the same
8000updates. Full-observation heads overfit/misfit despite containing the state:
extra available information alone does not establish a fitted-model ceiling.

## Limits

Risk fits use a fresh observation-target regression objective, not the native
self-referential latent JEPA loss. Transfer to trained native JEPA is UNPROVEN.
No real-data risk evaluation, independent data confirmation or new-method claim.
Raw plans, logs, source pins, checkpoints and generator scripts remain available.
