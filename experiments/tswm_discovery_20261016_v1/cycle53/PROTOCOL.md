# C53 native TSWM subset reproduction and new-crop transfer audit

Goal: obtain source-grounded evidence from competent time-series world models before designing another method. This is a subset of the published protocol, followed by an explicitly new transfer audit. It is not a new method or a full eight-dataset/seven-backbone reproduction.

## Source and data

- TSWM official repository commit 2e5e48540ff0da48e8192163969a6c58df1d5a3e. Original Python model, data, codec, trainer, optimizer, selection and directional-supervision code execute unchanged. A wrapper supplies paths, serial job scheduling and additional retained diagnostics only.
- Third Autonomous Greenhouse Challenge DOI 10.4121/21960932.v1, C52 selective acquisition. First original experiment is primary TRAIN/VAL: five train teams / one validation team, data seed 0, context 256 / horizon 16 / stride 80, native accepted counts 815/163. The published paper Table 1 reports 5/1 systems and 163 held-out windows, agreeing with this batch. The original paper supplied no raw file hashes, so exact byte-level provenance of its reported run is not claimed.
- Second original experiment is an untouched new-crop evaluation source, retained as a separate batch. Repeated compartment/team identities do not constitute independent real domains or a new unseen-subject test. No observations, future targets, scores or metadata from this batch select hyperparameters, codec or checkpoints. All formal fits complete and checkpoint identities are sealed before transfer evaluation opens.
- Inputs follow the native protocol: observed target history and factual future action/exogenous schedules. Future schedules are declared model inputs; this is a conditional forecast, not a prospective plan-availability or causal-identification claim. No EEG/clinical data are used here.

## Native subset

Backbones fixed before outcomes: CrossLinear and PatchTST. Five original training seeds 0,1,2,3,4. Four arms: observation-space concat; frozen target-AE concat; frozen target-AE Gate; native Gate with directional supervision (rho 1.0, default prior split, Instant plan encoding). Thus 40 predictive fits, serial on one RTX 5090, plus one shared native AE.

Use the native launcher configuration: context 256, horizon 16, stride 80, batch 128, 100 maximum epochs, patience 20, Adam lr 1e-3, weight decay 0, MSE main loss, clip 1, ReduceLROnPlateau factor .5/patience 5, no train/eval batch truncation. Train and validation reductions/selection stay native. Target AE: d_model 32, no RevIN, seed/data seed 0, 100 epochs, patience 15, batch 128, lr 1e-3, native codec window stride (the codec CLI has no stride argument). AE pretrained once and shared, as in source Stage 0. Native codec preparation runs on CPU; predictive fits on CUDA. No new loss, architecture or test-selected budget.

Record every native training curve through read-only wrappers around epoch-return values, selected checkpoint hash/config, training/deployment parameter counts, wall time, epochs and cap flags. Native final/selected formats are preserved. Capture independent RNG/schedule hashes when possible without drawing random values or changing their order. A diagnostic wrapper must not replace an optimizer update or selection rule. Exact initialization equality across unlike architectures is not claimed.

## Evaluation and attribution

Primary native VAL: full-precision retained predictions and MSE/MAE/RMSE, overall/per-channel/horizon, plus the exact native mechanism-consistency routine where compatible. Original paper comparisons are reported as subset concordance, not exact headline equality. Second-batch results are a new transfer audit and are never called an original-paper reproduction.

Retain pointwise absolute/squared errors, native predictions/targets and full input provenance on the server. Check native full-window histories against the C52 contract. Independently recompute score arithmetic and aggregate counts in NumPy; reload selected models and replay predictions, checking input/checkpoint/source hashes. Report every seed/arm/backbone; do not select a favorable backbone or seed after transfer. Date-block descriptive uncertainty must respect overlap; five model seeds are not five independent plants. Multiple per-channel comparisons are diagnostic and cannot by themselves validate a novel framework.

A shift-related error is not automatically a research gap. Before proposing a new mechanism, compare simple persistence, a regularized direct supervised control with the same available inputs and an adequate existing model/control for the specific observed failure. Any follow-up method uses a newly frozen protocol and independent confirmation data, rather than tuning second-batch scores.

## Storage and publication

Read server-local STORAGE_LOCATION_V1.json or explicit environment overrides. Datasets/caches/checkpoints/large raw arrays use the user-authorized large-disk project subdirectory. Do not synchronize results to the local workstation. Publish screened portable code/configuration and compact reports to Gifoe/wm4health, preserving existing experiments. Original server source paths/byte manifests and public path-redacted manifests must be distinguished. No credential material, third-party private launchers, raw signal files or environments enter GitHub.
