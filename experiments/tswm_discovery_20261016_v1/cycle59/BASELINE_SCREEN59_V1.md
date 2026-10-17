# Frozen deterministic baseline screen on the audited raw tutorial rows

Written after shape/provenance/core qualification and before any model fitting or predictive validation/test scoring. This is exploratory cross-row N87 response prediction, not an official MagNet benchmark reproduction, a new method, or evidence of cross-material transfer.

Data: all 128 raw N87_5 rows, 5008 samples, 25 C; verified original 62.5 ns clock. No circular shifting, whole-row H centering, synthetic history wrapping, row resampling or neighboring-window split. Exact duplicate B+T rows, if any, form indivisible groups. Sorted groups are permuted with seed 5900: floor(.6G) train, floor(.2G) validation, rest test. Raw row identities and group lists are sealed before fitting. Metadata auditing has read all raw arrays; the scoring receipt must not falsely claim the first ever test-array opening.

Two separately fitted history conditions: first 100 samples, or the first floor(L/4)=1252 samples. Input is observed prefix B,H plus the future B trajectory. Output is remaining H. All global feature/target scales derive only from training rows. No temperature transfer is measurable because temperature is constant.

Controls are ordinary deterministic constructions, not renamed contributions:

1. Static nonlinear input map: B, B^3, B^5, tanh(B), backward Delta B. No unknown H beyond the prefix is an input.
2. Static plus 16 standard play/stop operators, radii log-spaced .02 to 2 in train-B-scaled units.
3. Static plus 8 exponential relaxation states, physical time constants dt times [4,16,64,256,1024,4096,16384,65536].
4. Static plus both standard banks.
5. Stable ARX: the static map plus scalar autoregressive a in [0,.5,.9,.99,.999,.9999], observed last prefix H as initial condition, then free-running recursion. Teacher-forced training residual H[t]-a H[t-1], no future H in deployment.

For the first four controls, the prefix H mean calibrates output offset; subtract prefix feature means before applying learned weights. Play and relaxation banks are initialized from the first B sample and carried through the entire observed prefix. Unknown earlier magnetic state remains a limitation; poor predictions cannot by themselves prove a memory-structure failure. ARX uses the same available prefix, with its own state estimator.

Ridge regularization grid: [1e-4,.01,1,100] applied to mean squared training objective after train-only feature scaling. Pick each arm's regularization (and ARX a) by validation mean squared error in train-H-scaled units. Deterministic tie break is listed order. Eight bank models and two ARX models are selected; 80 total candidate fits across the two history conditions. Seal selected coefficients, initialization conventions, scales and validation decisions before computing any test prediction risk. No test-based retraining or selection.

Primary descriptive endpoint: test-row mean MSE divided by train-H variance. Also retain each row's NRMSE relative to its own future H RMS; that target-derived denominator is for evaluation only. Report all five arms for both prefixes, not only a selected winner. Paired row intervals are conditional on this data split, not neural-seed uncertainty. No new-method acceptance threshold is defined because no candidate new method is being tested.

If a static or ordinary dynamic control is already sufficient, lower the priority of a new-memory necessity story on this sample. If all controls have residuals, first examine unknown initial state, finite sample, basis/order, calibration, noise, waveform support and misspecification. Do not turn one material's residual into a general research gap. Proper nonlinear state-space / official ODE / expressive neural controls and independent systems remain necessary before a framework claim.
