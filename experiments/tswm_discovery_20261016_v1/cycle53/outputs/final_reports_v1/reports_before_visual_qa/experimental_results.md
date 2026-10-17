# C53 complete experimental results

C53 completed UTC 2026-10-17T17:07:30.920782+00:00. This is a native-source subset reproduction and a separately identified new-crop transfer audit: two backbones, four configurations, five seeds, 40 predictive fits and one shared AE. Selected predictive fits contain 15600 optimizer updates and 2591.94 native fit seconds; 0/40 predictive fits hit the 100-epoch cap. Codec training, qualification, failed attempts and plotting are additional cost. The native-driver failed attempt alone wasted at least 162 updates; unrecorded partial work cannot be counted exactly.

Source TRAIN/VAL has 815/163 accepted windows; the new crop has 803 windows from repeated compartment/team identities. All 40 checkpoints were fixed before transfer scoring. Five training seeds measure optimization variation, not five independent plants. Factual future action/exogenous schedules are supplied. The original paper's raw run-byte identity is unavailable, and PatchTST AE concat uses the disclosed effective-batch128/microbatch64 adaptation, so bitwise headline reproduction is not claimed.

Independent verification covered 84 retained forecast groups, 40,572 model-window predictions, 2,596,608 scalar predictions, all 40 loaded-forward/future-target-corruption checks and 80 native mechanism cells. Retained-array score arithmetic error was zero. Native mechanism functions are retained source; there is no independent intervention ground truth. Overall research goal remains ACTIVE_UNACHIEVED; accepted final directions: 0.

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

| Bank | Gate backbone | MAE change vs persistence (%) | MSE change vs persistence (%) |
|---|---|---|---|
| source_val | CrossLinear | 32.343 | 4.317 |
| source_val | PatchTST | 21.871 | -8.513 |
| new_crop_all | CrossLinear | 49.902 | 11.269 |
| new_crop_all | PatchTST | 15.440 | -17.858 |

Positive changes in the preceding table mean larger error. The fixed recipe trains on MSE; aggregate MAE and MSE answer different questions.

| Bank | Backbone | Paired contrast | MAE reduction (%) | Descriptive seven-day interval (%) |
|---|---|---|---|---|
| source_val | CrossLinear | obs -> ae_concat | 6.550 | [-0.141, 13.304] |
| source_val | CrossLinear | ae_concat -> ae_gate | 33.450 | [26.436, 41.268] |
| source_val | CrossLinear | ae_gate -> ae_gate_soft | -14.646 | [-20.686, -9.105] |
| source_val | PatchTST | obs -> ae_concat | 14.543 | [9.511, 19.536] |
| source_val | PatchTST | ae_concat -> ae_gate | 19.469 | [14.497, 23.836] |
| source_val | PatchTST | ae_gate -> ae_gate_soft | 0.286 | [-2.057, 3.024] |
| new_crop_all | CrossLinear | obs -> ae_concat | 8.419 | [7.276, 9.127] |
| new_crop_all | CrossLinear | ae_concat -> ae_gate | 9.719 | [7.989, 11.199] |
| new_crop_all | CrossLinear | ae_gate -> ae_gate_soft | -10.908 | [-12.853, -8.991] |
| new_crop_all | PatchTST | obs -> ae_concat | 2.409 | [1.234, 4.042] |
| new_crop_all | PatchTST | ae_concat -> ae_gate | 18.998 | [17.018, 20.548] |
| new_crop_all | PatchTST | ae_gate -> ae_gate_soft | -0.264 | [-1.212, 0.705] |

Positive paired reductions favor the candidate. Intervals use five-seed-averaged paired window errors and synchronized seven-day blocks: seven occupied source bins and six new-crop bins, 2,000 draws. They are descriptive conditional on these crops/compartments. Three-day sensitivity and all ordinary-control contrasts are retained in paired_contrasts.json. There is no independent-domain inference.

| Bank | Backbone | Configuration | Declared relation | Prior role | Mean q+0.10 agreement | Defined seeds | Coverage range |
|---|---|---|---|---|---|---|---|
| source_val | CrossLinear | ae_gate | Tpipe->Tair | train | 0.99141 | 5 | 1.0000-1.0000 |
| source_val | CrossLinear | ae_gate | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.0123-0.0123 |
| source_val | CrossLinear | ae_gate | CO2dosing->CO2air | heldout_clean | 0.00000 | 5 | 0.9939-0.9939 |
| source_val | CrossLinear | ae_gate | VentLee->Tair | train | 0.44612 | 5 | 0.5215-0.6196 |
| source_val | CrossLinear | ae_gate | VentWind->Tair | train | 0.39014 | 5 | 0.4110-0.5951 |
| source_val | CrossLinear | ae_gate_soft | Tpipe->Tair | train | 1.00000 | 5 | 1.0000-1.0000 |
| source_val | CrossLinear | ae_gate_soft | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.0123-0.0123 |
| source_val | CrossLinear | ae_gate_soft | CO2dosing->CO2air | heldout_clean | 0.00000 | 5 | 0.9939-0.9939 |
| source_val | CrossLinear | ae_gate_soft | VentLee->Tair | train | 1.00000 | 5 | 0.6810-0.6994 |
| source_val | CrossLinear | ae_gate_soft | VentWind->Tair | train | 1.00000 | 5 | 0.5828-0.5951 |
| source_val | PatchTST | ae_gate | Tpipe->Tair | train | 1.00000 | 5 | 1.0000-1.0000 |
| source_val | PatchTST | ae_gate | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.0123-0.0123 |
| source_val | PatchTST | ae_gate | CO2dosing->CO2air | heldout_clean | 0.08343 | 5 | 0.9816-0.9939 |
| source_val | PatchTST | ae_gate | VentLee->Tair | train | 0.68881 | 5 | 0.3558-0.6810 |
| source_val | PatchTST | ae_gate | VentWind->Tair | train | 0.01846 | 5 | 0.5153-0.5521 |
| source_val | PatchTST | ae_gate_soft | Tpipe->Tair | train | 1.00000 | 5 | 1.0000-1.0000 |
| source_val | PatchTST | ae_gate_soft | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.0123-0.0123 |
| source_val | PatchTST | ae_gate_soft | CO2dosing->CO2air | heldout_clean | 0.00248 | 5 | 0.9877-0.9939 |
| source_val | PatchTST | ae_gate_soft | VentLee->Tair | train | 1.00000 | 5 | 0.6748-0.6994 |
| source_val | PatchTST | ae_gate_soft | VentWind->Tair | train | 1.00000 | 5 | 0.5951-0.6933 |
| new_crop_all | CrossLinear | ae_gate | Tpipe->Tair | train | 0.92134 | 5 | 0.9527-1.0000 |
| new_crop_all | CrossLinear | ae_gate | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.1644-0.1644 |
| new_crop_all | CrossLinear | ae_gate | CO2dosing->CO2air | heldout_clean | 0.00000 | 5 | 0.8132-0.8132 |
| new_crop_all | CrossLinear | ae_gate | VentLee->Tair | train | 0.62264 | 5 | 0.6874-0.7796 |
| new_crop_all | CrossLinear | ae_gate | VentWind->Tair | train | 0.51990 | 5 | 0.6638-0.9489 |
| new_crop_all | CrossLinear | ae_gate_soft | Tpipe->Tair | train | 1.00000 | 5 | 1.0000-1.0000 |
| new_crop_all | CrossLinear | ae_gate_soft | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.1644-0.1644 |
| new_crop_all | CrossLinear | ae_gate_soft | CO2dosing->CO2air | heldout_clean | 0.00000 | 5 | 0.8082-0.8132 |
| new_crop_all | CrossLinear | ae_gate_soft | VentLee->Tair | train | 1.00000 | 5 | 0.9215-0.9265 |
| new_crop_all | CrossLinear | ae_gate_soft | VentWind->Tair | train | 1.00000 | 5 | 0.9639-0.9689 |
| new_crop_all | PatchTST | ae_gate | Tpipe->Tair | train | 0.99975 | 5 | 1.0000-1.0000 |
| new_crop_all | PatchTST | ae_gate | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.1644-0.1644 |
| new_crop_all | PatchTST | ae_gate | CO2dosing->CO2air | heldout_clean | 0.10566 | 5 | 0.7945-0.8132 |
| new_crop_all | PatchTST | ae_gate | VentLee->Tair | train | 0.73774 | 5 | 0.4832-0.9240 |
| new_crop_all | PatchTST | ae_gate | VentWind->Tair | train | 0.13657 | 5 | 0.7435-0.9415 |
| new_crop_all | PatchTST | ae_gate_soft | Tpipe->Tair | train | 1.00000 | 5 | 1.0000-1.0000 |
| new_crop_all | PatchTST | ae_gate_soft | AssimLight->PARin | heldout_clean | 1.00000 | 5 | 0.1644-0.1644 |
| new_crop_all | PatchTST | ae_gate_soft | CO2dosing->CO2air | heldout_clean | 0.04450 | 5 | 0.8070-0.8132 |
| new_crop_all | PatchTST | ae_gate_soft | VentLee->Tair | train | 1.00000 | 5 | 0.9103-0.9303 |
| new_crop_all | PatchTST | ae_gate_soft | VentWind->Tair | train | 1.00000 | 5 | 0.9676-0.9801 |

This table displays q+0.10 for readability. Every prior, seed, positive/negative shift, original score total/count and unavailable entry is retained in the native JSON and companion CSVs. Lighting source upward coverage is 2/163; high agreement there is a narrow check.

| Bank | Role | Channel | Below fixed range (%) | Above fixed range (%) |
|---|---|---|---|---|
| new_crop_all | target | PARin | 0.000 | 4.024 |
| new_crop_all | exog | Tout | 0.000 | 9.644 |
| new_crop_all | exog | Iglob | 0.000 | 10.336 |
| new_crop_all | exog | PARout | 0.000 | 12.539 |
| new_crop_all | exog | AbsHumOut | 0.000 | 4.475 |

The source recipe's output Gate advantage over AE concat persists on both backbones and both crops, with positive paired mean gains in all five seeds for these contrasts. AE concat alone has substantial seed dependence. Native directional supervision raises the penalized temperature-response signs but produces a CrossLinear accuracy cost; its PatchTST factual effect is small and varies by seed.

Persistence has lower aggregate MAE than every learned configuration on both crops. PatchTST Gate has lower MSE than persistence on both crops, while CrossLinear Gate does not. All predictive models were trained/selected using the native MSE protocol. The criterion-dependent comparison prevents treating the MAE ranking as evidence of general dynamical-learning failure. No loss-swap or persistence-blend is proposed as a novel world-model method.

The held-out CO2 sign remains mostly wrong, while lighting q+0.10 agreement is perfect on a restricted subset (only 2/163 source windows can be counted for that upward lighting shift). Penalized and held-out relations, positive/negative shifts and their coverage must remain separate. These are model responses against declared sign priors, without observed counterfactual effect truth. The native code and paper already distinguish supplied and held-out signs; this outcome alone supplies no new research gap.

The new crop also clips some raw exogenous values at native fixed ranges, including outdoor temperature, irradiance and outdoor PAR, and 4.024% of future indoor PAR target entries exceed its native upper range. This is a protocol/shift confound to retain in attribution, not a demonstrated cause of all errors. Physical-unit errors obtained by undoing affine scaling refer to clipped native targets. Timestamp steps are 299.980-300.067 seconds; all accepted steps are within one second of the nominal five minutes, so strict equality's timing flags are immaterial subsecond variation.

Server evidence paths are logical, relative to the configured artifact root: `cycle53/native_v1/evaluation_v1/`, `analysis_v1/`, `mechanism_summary_v1/`, `data_diagnostics_v1/` and `data_diagnostics_v2/`. Compact public copies and explicit path-redaction provenance are prepared for GitHub; raw signals, prediction tensors, checkpoints and environment files remain on the server. Every numerical table here derives from the retained verified files.
