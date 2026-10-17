# C53 complete baseline registry

C53 completed UTC 2026-10-17T17:07:30.920782+00:00. This is a native-source subset reproduction and a separately identified new-crop transfer audit: two backbones, four configurations, five seeds, 40 predictive fits and one shared AE. Selected predictive fits contain 15600 optimizer updates and 2591.94 native fit seconds; 0/40 predictive fits hit the 100-epoch cap. Codec training, qualification, failed attempts and plotting are additional cost. The native-driver failed attempt alone wasted at least 162 updates; unrecorded partial work cannot be counted exactly.

Source TRAIN/VAL has 815/163 accepted windows; the new crop has 803 windows from repeated compartment/team identities. All 40 checkpoints were fixed before transfer scoring. Five training seeds measure optimization variation, not five independent plants. Factual future action/exogenous schedules are supplied. The original paper's raw run-byte identity is unavailable, and PatchTST AE concat uses the disclosed effective-batch128/microbatch64 adaptation, so bitwise headline reproduction is not claimed.

Independent verification covered 84 retained forecast groups, 40,572 model-window predictions, 2,596,608 scalar predictions, all 40 loaded-forward/future-target-corruption checks and 80 native mechanism cells. Retained-array score arithmetic error was zero. Native mechanism functions are retained source; there is no independent intervention ground truth. Overall research goal remains ACTIVE_UNACHIEVED; accepted final directions: 0.

Pinned implementation: [TSWM source](https://github.com/HaochenZhang717/TSWM-Official-Repo/tree/2e5e48540ff0da48e8192163969a6c58df1d5a3e), [paper](https://arxiv.org/abs/2610.01842), [original data](https://doi.org/10.4121/21960932.v1). All 125 original Python source hashes were checked. C52 records acquisition and interpolation-contract audits; required_data_manifest_v1.json specifies the 26 public XLSX loader inputs. Twelve existing caches additionally matched native reconstruction from those raw files exactly.

The four native configurations are observation concat, frozen-target-AE concat, frozen-target-AE Gate and Gate with the original directional objective. Native hyperparameters, data roles, group split, random seeds and selection rule remain fixed in PROTOCOL.md. The paper-native configurations have different parameter counts and observed runtimes; this is not a new-method capacity/compute-matched study.

Persistence and direct ridge are original diagnostic controls with the declared inputs. Ridge alpha is 1000.0 selected exclusively by first-crop VAL equal-batch MSE from the frozen five-alpha grid. Controls have one fit, not five independent seeds. They were evaluated factually; no intervention-effect truth was added.

| Bank | Backbone | Configuration | MAE mean and training-seed SD | MSE mean | Seeds |
|---|---|---|---|---|---|
| source_val | CrossLinear | obs | 0.116913 +/- 0.03497 | 0.024184 | 5 |
| source_val | CrossLinear | ae_concat | 0.109255 +/- 0.00380 | 0.019490 | 5 |
| source_val | CrossLinear | ae_gate | 0.072710 +/- 0.00191 | 0.011187 | 5 |
| source_val | CrossLinear | ae_gate_soft | 0.083359 +/- 0.00344 | 0.014017 | 5 |
| source_val | PatchTST | obs | 0.097294 +/- 0.03089 | 0.017585 | 5 |
| source_val | PatchTST | ae_concat | 0.083144 +/- 0.02234 | 0.013657 | 5 |
| source_val | PatchTST | ae_gate | 0.066957 +/- 0.00095 | 0.009811 | 5 |
| source_val | PatchTST | ae_gate_soft | 0.066765 +/- 0.00135 | 0.009826 | 5 |
| source_val | ordinary_control | persistence | 0.054941 | 0.010724 | 1 |
| source_val | ordinary_control | ridge | 0.080786 | 0.012351 | 1 |
| new_crop_all | CrossLinear | obs | 0.183151 +/- 0.03222 | 0.063990 | 5 |
| new_crop_all | CrossLinear | ae_concat | 0.167731 +/- 0.00750 | 0.054857 | 5 |
| new_crop_all | CrossLinear | ae_gate | 0.151430 +/- 0.00179 | 0.047189 | 5 |
| new_crop_all | CrossLinear | ae_gate_soft | 0.167947 +/- 0.00868 | 0.053484 | 5 |
| new_crop_all | PatchTST | obs | 0.147521 +/- 0.02073 | 0.047121 | 5 |
| new_crop_all | PatchTST | ae_concat | 0.143968 +/- 0.02535 | 0.046427 | 5 |
| new_crop_all | PatchTST | ae_gate | 0.116617 +/- 0.00299 | 0.034836 | 5 |
| new_crop_all | PatchTST | ae_gate_soft | 0.116925 +/- 0.00260 | 0.034671 | 5 |
| new_crop_all | ordinary_control | persistence | 0.101019 | 0.042410 | 1 |
| new_crop_all | ordinary_control | ridge | 0.152577 | 0.056544 | 1 |

Complete training diagnostics, every source selection curve, all channel/horizon metrics and all compartment metrics are retained. The fresh-directory entry point passed staging/AST/raw-cache checks; its full 40-fit pipeline has not been rerun (FRESH_RUN.md).
