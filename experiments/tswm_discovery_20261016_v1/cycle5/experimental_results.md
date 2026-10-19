# Executed experiments and metrics

## KUMA: complete released-code configuration

PEMS03 96-to-12, seed2025, full width256/FF256/one layer, batch32, Adam0.002,
learning rate halves each epoch, patience3/max20, workers0, no batch caps.
Training completed 14 epochs, best logged validation at epoch11.
The native engine reloaded its validation-selected checkpoint before testing.
Test MSE **0.063033100885**, MAE **0.166339368318**, 5135 windows.
One seed; a healthy reproduction, not a new mechanism result.
`outputs/kuma_native_pems03_affinity1/` stores raw log, argv, checkpoint, native
unrounded `results.jsonl`, process metadata and successful exit status.

## COFE: full native-model reproduction and implementation controls

All three use 638917 parameters, ETTh1, context24/horizon24, seed1, Adam0.001,
50 epochs x100 updates, batch48, clip0.5. Training-only global scaler; fixed
train/val/test endpoints8640/11520/14400; 30 validation and30 test windows,
native rolling stride96. Origins are sampled uniformly on native support192..8616,
a declared difference from the GluonTS sampler. Native loss bodies, native
100-sample standard-normal latent-prior generation and quantile scores retained.
Validation raw-unit19-quantile score alone selects the checkpoint; no test tuning.

| Variant | Selected epoch | Best validation native CRPS | Test normalised MSE | Test empirical CRPS | Test90% coverage |
|---|---:|---:|---:|---:|---:|
| released | 28 | 0.224694 | 0.441321 | 0.346482 | 0.900198 |
| scale_fix | 19 | 0.218883 | 0.463980 | 0.309550 | 0.872024 |
| scale_fix_noaux | 22 | 0.215862 | 0.514813 | 0.316373 | 0.871230 |

`scale_fix` applies softplus in normalized coordinates then multiplies the
context standard deviation and divides by absolute affine scale; it never adds
a context mean to dispersion. `scale_fix_noaux` additionally sets the two
auxiliary weights to zero, with unchanged architecture/data/update budget.
One seed and one dataset do not support a stable improvement or an ablation
claim that the auxiliary term never helps. MSE and probabilistic scores trade off.

The following stress adds a fixed normalized level offset to both history and
target after checkpoint selection. This is synthetic real-series stress, not a
causal intervention or full-pipeline unit-conversion claim. Original-unit
normalized CRPS has a changing absolute-target denominator under this stress;
use standardized empirical CRPS and coverage/width for location comparisons.

| Variant | Offset | Coverage90 | Interval90 width | Empirical CRPS |
|---|---:|---:|---:|---:|
| released | -3.0 | 0.217064 | 0.316582 | 0.370713 |
| released | 0.0 | 0.900198 | 2.610761 | 0.346482 |
| released | 3.0 | 0.998016 | 9.717147 | 0.780029 |
| scale_fix | -3.0 | 0.872024 | 1.715783 | 0.309550 |
| scale_fix | 0.0 | 0.872024 | 1.715786 | 0.309550 |
| scale_fix | 3.0 | 0.872024 | 1.715788 | 0.309551 |
| scale_fix_noaux | -3.0 | 0.871230 | 1.842687 | 0.316373 |
| scale_fix_noaux | 0.0 | 0.871230 | 1.842685 | 0.316373 |
| scale_fix_noaux | 3.0 | 0.871230 | 1.842674 | 0.316373 |

Three component seeds41/42/43 passed finite native forward/loss/backward checks.
Zeroing the unconstrained noise-estimator parameters leaves forecast loc/scale
exactly unchanged and reaches the auxiliary regularizer minimum. The core
gradient equals its no-auxiliary gradient after zeroing. This is also confirmed
on all three trained checkpoints using training samples only; released loss
1.020407438 ->1.020308495, distribution-parameter change exactly0. It undermines
mechanism attribution through this auxiliary objective; it is not a new method.
Native num_samples10 returns component locations, unlike100 observation draws.
The native default100 used above is unaffected by that special10 branch.

## Continuous partially observed counterfactual computation

18 frozen cases: d2/3/5 x T64/256 x independent system seeds61/62/63.
The two horizons are nested within each system, not independent replications.
Known two-action linear Gaussian transition/measurement model; one global
time-invariant R in O(d), both determinant components, identical factual and
interventional transition kernels across the restricted SCM family. Factual
observations are noisy/partial. Fixed open-loop alternative action plan; query
is conditional terminal squared error. This is not a bound over all SCMs.

The terminal query is affine in vec(R) with Gaussian random coefficients.
An ordinary augmented Kalman filter obtains their posterior mean/covariance.
Independent dense Gaussian conditioning agrees within 5e-15;
20 direct twin-state Kalman checks/case agree within 1.15e-14.
Both O(2) components have an independent all-critical-roots cross-check.
Existing Shor SDP with row/column orthogonality constraints plus feasible
multistart/polar rounding gives maximum absolute gap 2.48756313e-05, maximum
relative gap 2.48756313e-05; all36 extrema pass the frozen1% gate.
Maximum numerical constraint violation 6.94e-09. Tiny signed
gap reversals down to -9.13e-08 are solver tolerance effects, not formal
certificates. The SDP numbers are **not interval-arithmetic-validated bounds**.

Decision: **STOP_CUSTOM_SOLVER_IN_THIS_REGIME**. Existing tools already resolve this small
regime. No learned-world-model, nonlinear-system or globally sharp
nonparametric counterfactual claim is supported.

## Runtime and implementation failures

The first unpinned KUMA attempt exited3221225477 (Windows0xC0000005) after epoch12.
The first COFE attempt disappeared after epoch27; its precise exit code was not
captured. Neither incomplete attempt contributes to the completed comparisons.
Their logs/checkpoints are preserved. No contemporary exhaustion event was found;
host application-error events include native-library faults, without proving
the physical cause. No unrelated job or global system setting was changed.

Process-local affinity0xffff0000, one CPU thread, MKL sequential were used for
new independent retries. A45-second arithmetic diagnostic passed15672 loops.
The complete four fits then exited normally. The first 12 KUMA epochs
match the interrupted attempt's printed training/validation metrics exactly.
This is bounded evidence of a working profile, not proof of permanent stability.

The first counterfactual pilot failed its independent O(2) assertion because its
feasible optimiser's ascent/descent signs were reversed. Corrected before any
accepted row; scientific plan/seeds/stopping gate stayed fixed. The failure is
retained in `continuous_counterfactual_implementation_failure.md`.
