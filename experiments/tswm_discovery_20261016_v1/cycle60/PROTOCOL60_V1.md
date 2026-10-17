# C60: competent latent-state controls on two independent measured systems

Date 2026-10-18. Overall goal ACTIVE / UNACHIEVED; accepted directions 0. C59 executed a real N87 screen but did not compare a trained generic neural state-space model. This follow-up fills that missing control; no proposed new architecture is being tested.

## Source and protocol identities

Use the pinned actual torchid state update, output, LSTM initializer and simulator from forgi86/sysid-neural-estimator. Exact commit/blob identities are in source_selection60_v1.json. This is a cross-dataset adaptation of those components, not reproduction of the authors' complete paper table or training budget. Do not copy the native loader's full-record output normalization into a training-only comparison. Download the small raw pick-place MAT and exact Python sources only; arrays/checkpoints use the authorized G: subtree. No packages or drivers change, and no server result files synchronize locally.

MetaLIC's released benchmark evaluation uses actual output histories in every test window and encoder/readout prediction. Its released loss does not train the computed x_plus transition term. That source route is a valid one-step-history contract, not a ready free latent rollout comparator. This is an implementation/protocol eligibility observation, not a novel research gap or a judgment that the paper's declared task is invalid.

## Data and availability

1. N87: reuse hash-verified C59 raw 128x5008 measurements and the sealed 76/25/27 row split. Use prefix 100, matching the existing ordinary bank's short-history condition. The old heldout scores are already known, so this follow-up is exploratory baseline completion, not an untouched confirmatory test for a new method. Do not change the bank based on the new neural results.
2. Pick-place: pinned raw headdata_1v6.mat. Keep chronological split consistent with source: raw first 40000 fit, next 10000 validation, final remainder test; sample every tenth raw point as the source does. Audit actual t/u/y shape and units/clock before fitting. Use prefix 100 within each complete validation/test partition. Do not treat contiguous windows as independent devices.

All u/y normalization uses fit data only. The neural input is current u and its backward increment, both derived from the same known input trajectory. Past y is available only to the initializer. No future y, latent truth or test-selected hyperparameter enters inference.

## Ordinary neural control

Use native NeuralLinStateUpdateV2, 8 latent states, hidden width 64; native LinearOutput with one scalar response; native LSTMStateEstimator hidden width 32 with all available prefix u/y. For a known sampled input trajectory, the adapter applies one native residual state update then emits its output, interpreting the initial state as the last observed-boundary state. Qualify this output timing explicitly against a manual unroll; do not silently call it the native simulator's pre-update output convention. The adapter is an ordinary convention change, not a new method.

Train each system separately with seeds 2021,2022,2023, 5000 Adam updates, lr=.0005, batch 32, observed context 100 and free-running training future 128. Draw windows from fit data only. Clip gradient norm at 1. No late recipe changes after any test score. Validate fixed complete validation partitions every 250 updates including step 0, select the finite minimum validation MSE checkpoint, and retain the terminal checkpoint too. Preserve all seed outcomes. If a training loss becomes nonfinite, stop that fit and retain it; do not silently retry a seed. Validation divergence is recorded explicitly rather than serialized as an invalid JSON infinity.

Before real fitting, qualify native import/parameter hashes, eager/scripted post-update outputs and gradients, and resource-profile the identical synthetic workload on CPU/CUDA. Choose the execution device from measured synthetic step time, before model fitting. No benchmark outcomes may choose the device. The main purpose is a competent control, not a CPU/GPU performance paper.

## Evaluation and interpretation

Seal all selected checkpoint hashes before the new neural test-scoring step. Score complete heldout response tails without re-encoding future true outputs. Primary descriptive endpoint: MSE in fit-y-standard-deviation units, plus per-complete-record NRMSE. Report seeds separately; record-level conditional uncertainty is not a training-seed confidence interval. Native source and the adapter's derived u-increment input/output timing differences remain explicit.

Compare N87 with the immutable C59 static/play/EMA/hybrid/ARX rows, without asserting equal training compute. For pick-place, build the same fixed standard static/play/EMA/hybrid banks using training data and validation-only ridge selection, plus stable ARX. Do not require a single shared checkpoint or physical-state recovery: the desired research level is a reusable modeling rule, and both controls use the same code/recipe across systems with separate fitted parameters.

Poor finite-budget neural performance does not establish a new memory problem. First inspect validation convergence, initialization, horizon mismatch, state/order and output-family adequacy. If ordinary controls suffice, abandon a new-architecture necessity claim in the tested setting. Independent devices, broader systems and a distinct useful mechanism remain required before accepting a research direction.
