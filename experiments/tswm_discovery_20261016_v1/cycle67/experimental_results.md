# Real-task results and independent evidence

Primary metric is per-case Libre MAE divided by that case's observed-context sample std; average MC/cases within each participant, then average participants equally. Raw MAE and case RMSE are in mg/dL; RMSE is averaged by case/participant, not a pooled global square root. Each subset has9 participants. MC streams are repeated samples from the same frozen model, not independent fits. Source resampling provenance, different future information and target-domain training must be respected.

Validation35 cases,910 prediction records:

| method | information group | participant-mean normalized MAE | MAE mg/dL | mean case RMSE mg/dL |
|---|---|---|---|---|
| analog_k1 | known plan + TRAIN-domain labels | 1.1396861 | 26.501906 | 33.501443 |
| analog_k16 | known plan + TRAIN-domain labels | 0.77267234 | 18.169943 | 22.671761 |
| analog_k4 | known plan + TRAIN-domain labels | 0.87103523 | 20.200737 | 24.988548 |
| bayes_action_full_view | known plan; frozen backbone/context fit | 0.93325698 | 22.369328 | 28.462305 |
| bayes_action_past_view | known plan; frozen backbone/context fit | 0.93325698 | 22.369328 | 28.462305 |
| bayes_ignored | context transition; plan matching ignored | 0.97193278 | 23.714649 | 29.642885 |
| bayes_indexed | known plan; frozen backbone/context fit | 0.93281082 | 22.353824 | 28.443715 |
| bayes_raw_macro | known plan; frozen backbone/context fit | 0.89988096 | 21.838808 | 27.539319 |
| clamped_simulate | known plan; frozen backbone/context fit | 0.79531671 | 19.070637 | 23.99358 |
| context_mean | context only | 0.81339612 | 19.614463 | 24.692841 |
| last_value | context only | 1.3520731 | 35.391292 | 40.128038 |
| native_predict | context only | 0.96430029 | 22.791838 | 28.291713 |
| ridge_alpha1 | known plan + TRAIN-domain labels | 0.85773091 | 20.352677 | 25.542866 |
| ridge_alpha100 | known plan + TRAIN-domain labels | 0.73847504 | 17.585601 | 22.094319 |
| ridge_alpha10000 | known plan + TRAIN-domain labels | 0.77476765 | 18.339593 | 22.896293 |
| seasonal_day | context only | 0.99465214 | 24.444678 | 31.275805 |

Classical k16/alpha100 were selected and hashed before EVAL. Evaluation33 cases,726 records:

| method | information group | participant-mean normalized MAE | MAE mg/dL | mean case RMSE mg/dL |
|---|---|---|---|---|
| analog_k16 | known plan + TRAIN-domain labels | 0.83116069 | 18.375429 | 22.89063 |
| bayes_action_full_view | known plan; frozen backbone/context fit | 0.99167263 | 22.577355 | 29.147892 |
| bayes_action_past_view | known plan; frozen backbone/context fit | 0.99167263 | 22.577355 | 29.147892 |
| bayes_ignored | context transition; plan matching ignored | 0.98932588 | 23.079851 | 29.46938 |
| bayes_indexed | known plan; frozen backbone/context fit | 0.99813962 | 22.685396 | 29.270005 |
| bayes_raw_macro | known plan; frozen backbone/context fit | 0.99883691 | 22.748454 | 29.085634 |
| clamped_simulate | known plan; frozen backbone/context fit | 0.88181266 | 19.957055 | 25.34505 |
| context_mean | context only | 0.89341539 | 20.362146 | 26.017874 |
| last_value | context only | 1.4514963 | 36.307698 | 41.142668 |
| native_predict | context only | 1.0697421 | 23.887553 | 29.388736 |
| ridge_alpha100 | known plan + TRAIN-domain labels | 0.83653112 | 19.440399 | 24.046241 |
| seasonal_day | context only | 1.0977372 | 25.660429 | 32.998513 |

Main effect is mixed across subsets: full-view action re-encoding is0.048% worse on VAL and0.648% lower on EVAL than indexed native matching. All9 EVAL participant means improve, and the descriptive bootstrap interval is below0; do not call the effect absent or statistically zero. About82.82% of the EVAL aggregate absolute decrease comes from one participant. The small effect and prior-art overlap do not justify promoting this repair into a new framework or treating view mismatch as the dominant error source. Native clamped simulation is already a stronger context-only-fit control here. Analog/ridge use additional TRAIN-domain labels and must be grouped separately.

Every prelisted descriptive contrast,10,000 participant-bootstrap draws, seed67025; these are exploratory unadjusted intervals, not nine times the number of windows or a corrected family of confirmatory tests:

| subset | A minus B | normalized MAE delta | A better people /9 | descriptive bootstrap95% |
|---|---|---|---|---|
| VALIDATION | bayes_action_full_view minus bayes_indexed | 0.00044615564 | 4 | [-0.00194819, 0.00327554] |
| VALIDATION | bayes_action_past_view minus bayes_indexed | 0.00044615923 | 4 | [-0.00194818, 0.00327555] |
| VALIDATION | bayes_raw_macro minus bayes_indexed | -0.032929858 | 7 | [-0.05817573, -0.00343071] |
| VALIDATION | bayes_indexed minus bayes_ignored | -0.039121957 | 4 | [-0.12910971, 0.03467268] |
| VALIDATION | bayes_action_full_view minus bayes_action_past_view | -3.5892938e-09 | 7 | [-1e-08, -0.0] |
| VALIDATION | clamped_simulate minus context_mean | -0.018079411 | 6 | [-0.03269176, -0.00337698] |
| VALIDATION | analog_k16 minus bayes_indexed | -0.16013848 | 8 | [-0.22344476, -0.09463088] |
| VALIDATION | ridge_alpha100 minus bayes_indexed | -0.19433578 | 9 | [-0.24841598, -0.14452134] |
| EVALUATION | bayes_action_full_view minus bayes_indexed | -0.0064669877 | 9 | [-0.01723121, -0.0004819] |
| EVALUATION | bayes_action_past_view minus bayes_indexed | -0.0064669906 | 9 | [-0.01723121, -0.0004819] |
| EVALUATION | bayes_raw_macro minus bayes_indexed | 0.00069728614 | 5 | [-0.02377229, 0.02542269] |
| EVALUATION | bayes_indexed minus bayes_ignored | 0.0088137449 | 4 | [-0.0533968, 0.06854682] |
| EVALUATION | bayes_action_full_view minus bayes_action_past_view | 2.8551484e-09 | 4 | [-0.0, 1e-08] |
| EVALUATION | clamped_simulate minus context_mean | -0.011602734 | 6 | [-0.0268952, 0.00565213] |
| EVALUATION | analog_k16 minus bayes_indexed | -0.16697893 | 8 | [-0.25971656, -0.06419828] |
| EVALUATION | ridge_alpha100 minus bayes_indexed | -0.1616085 | 7 | [-0.25830411, -0.05785693] |

EVAL main contrast by participant:

| participant | full-view normalized MAE | indexed normalized MAE | difference |
|---|---|---|---|
| CGMacros-003.csv | 1.32263124 | 1.32265921 | -2.79762826e-05 |
| CGMacros-013.csv | 0.814025883 | 0.814363763 | -0.00033788042 |
| CGMacros-014.csv | 0.871698453 | 0.872623716 | -0.000925263434 |
| CGMacros-015.csv | 0.969480582 | 0.969624569 | -0.000143986731 |
| CGMacros-023.csv | 0.978406628 | 0.98135131 | -0.00294468226 |
| CGMacros-027.csv | 0.95955059 | 0.960231427 | -0.000680837001 |
| CGMacros-032.csv | 1.15228907 | 1.20049331 | -0.0482042451 |
| CGMacros-036.csv | 0.777621682 | 0.782526644 | -0.00490496145 |
| CGMacros-044.csv | 1.07934957 | 1.07938263 | -3.30570625e-05 |

Full-view versus past-view action controls differ by at most4.57763672e-05mg/dL over all retained forecasts. This establishes near-equivalence of measured outputs here, not a theorem about every internal representation or sequence length. Raw-macro matching improves VAL but does not preserve the improvement on EVAL. Conditioning's effect also depends on the reported metric; do not infer causal validity from an error difference.

Independent replay PASS: {"case_forecast_archives": 68, "classical_forecasts_replayed": 480, "forecast_metric_rows": 1636, "fresh_native_forecasts": 2, "processed_cases_from_source": 490, "raw_source_hashes": 45, "ridge_mask_group_checks": 3, "source_macro_rounding_bounds": 490, "subset_result_rows": 1636, "validation_only_selections": 2}. All490 retained cases were reconstructed from raw CSVs; all68 forecast archives and1636 metric rows were hashed/recomputed;480 classical forecasts replayed; three ridge matrices verified by an independent centered SVD solution; two fixed native EVAL calls matched bitwise. Maximum raw-MAE discrepancy=1.066e-14, ridge-coefficient discrepancy=3.059e-14. The first verifier stopped on a7.15e-7 float32 macro-sum versus float64-reference discrepancy. V2 kept strict history/metric tolerances and checked each macro feature against its explicit60*eps32*sum_abs arithmetic bound; no primary data/prediction/score changed.

Timings/call counts remain per row. The serial bundle caches conditioning views for ordinary controls, so these are not standalone equal-compute latency benchmarks. Raw numerical records and forecast arrays remain on G, with FORECAST_ARCHIVE_LEDGER67_V1.json listing all hashes. No physical counterfactual truth, uncertainty calibration, original-paper composite score or original-paper split reproduction was established. One independently specified dataset protocol is not a general framework validation.
