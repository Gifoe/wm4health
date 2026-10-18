# Executed pipeline and artifact interpretation

This archive contains historical/versioned scripts, including unsuccessful diagnostics and superseded finalizers. They are retained for traceability, not all part of the current execution path. No final research direction was accepted.

## Actual core sequence

1. Use the pinned source/assets/dependency environment documented in `baseline_registry.md`, `SOURCE_PROVENANCE87_V1.json`, the C80 manifests and `native87_v1.py`. Public source OpenFIM cee2bb53; HF fim-ode00875c25. Native model12,968,068 parameters. The actual runtime uses Torch2.8cu128 on RTX5090 and the previously qualified task-local dependencies.
2. Freeze `PROTOCOL87_V1.md`; run `smoke87_v1.py` and `generate87_v1.py`. TRAIN/VAL only, no test opening. `verify_data87_v1.py` checks exact data pairing and independent physical references.
3. Run `train87_v1.py`:9 fits x3000 updates, all parameter gradients enabled, common pretrained initialization and paired minibatch/dropout RNG streams. Select by common VAL native-normalized field l1. All7 checkpoints per fit remain on the server. `ordinary87_v1.py` and `qualify_val87_v1.py` establish the ordinary weak-fit selector and normalization equivalence.
4. Before test opening, `STRONG_PRIOR_CONTROL87_V1.md` adds an ordinary prior-aware control. `empirical87_v1.py` learns priors from the same32 source field labels per mechanism without family/parameter labels and seals its VAL choices. The first waiting queue was stopped before opening; training was preserved.
5. `evaluate87_v2.py` is the executed held-out evaluator:14 models,2 families,3 cohorts, fresh seed2026875. `evaluate87_v1.py` is superseded. The actual neural inference batch is256 and loaded native code enables matmul precision high/TF32. Reported native simulations use RK4 with4 substeps per.05 interval.
6. `correct_affine87_v1.py` changes only the degree1 ordinary baseline's numerical integration to exact augmented matrix exponentials. It produces `EVALUATION_RESULTS87_V3.json`, retaining the original V2 results and recording every changed array/hash. No model, data, selected hyperparameter or native/empirical prediction changed.
7. `analyze87_v1.py` computes descriptive paired effects. `qualify_report87_v1.py` adds explicit numerical limitations to `POSTSTUDY_ANALYSIS87_V2.md`. Bootstrap intervals condition on the fixed source/test pools; they are not evidence of training-population robustness.
8. `close_numerics87_v1.py` records the unsuccessful supplemental learned-rollout checks. `verify87_v4.py` separately replays all63 source selections/RNG, direct native field arrays, physical truth, Gaussian conditioning, reported metrics and exact affine flow. Its status explicitly says learned-rollout qualification is incomplete. It is not a full numerical certificate.
9. `finalize87_v7.py` is the completed snapshot builder; earlier partial finalization is retained remotely. Source-only report/status helpers create the archival package. Figures are visually checked from server bytes in RAM; result/model/data files are not synchronized locally.

## Limits that must accompany reported numbers

Read `NUMERICAL_OUTCOME87_V1.md`, `PRECISION_NOTE87_V1.md` and `SUPERVISION_BOUNDARY87_V1.md`. The supplementary1% learned-rollout criterion failed on the first frozen-model cohort and did not reach adapted-model cases. Reported simulation metrics therefore describe the fixed computational protocol and are not certified continuous-time errors. Source field/metric replay is a distinct, completed check.

The empirical Bayes control is an existing procedure. Its strong reported performance and the inconsistent resampling effects do not establish a new method, nor prove every possible selection effect absent. The source training dataset of the released checkpoint is unavailable, so this archive cannot establish its causal pretraining defect.

## Storage and reproducibility

Raw generated pools, all weights, prediction arrays, original logs and intermediate/failed artifacts stay on the remote15TB storage. The published manifests retain their hashes and original server paths. Public repository content is source, configuration, summaries, metadata, diagnostic logs and figures; no raw arrays or model weights are uploaded. Reproduce the source data from frozen seeds and equations rather than downloading an unspecified corpus. Do not run a closed stage twice into the same immutable output directory; use an explicit fresh run root/version.
