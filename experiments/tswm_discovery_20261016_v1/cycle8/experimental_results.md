# Completed real-data experiments

## Protocol

This cycle executes **new real-data control experiments**, not original WorldTS, TAWM, RP-GSSM or JEPA reproductions. All numerical fits and audits run on the remote research server; ordinary CPU ridge/gradient boosting is deliberately used before expensive neural training. Native source/configuration is pinned for the data-generating protocols and inspected firmware, not claimed to be the exact flashed board version.

Train: nine actuator-walk recordings2–10. Validation:11–12. Held-out test:13–16. Truncated recording1 has1016rows and is excluded by the inventory before any model metric. Ten independently recorded wave runs and two320k regime runs are separate OOD groups and select no hyperparameters. No window crosses a recording boundary. A16-step horizon is an observation-count horizon with variable physical duration, not a fixed-wall-time simulator certificate.

Future programmed actuator/sensor commands are known and allowed by the recorded open-loop protocol. Actual future timestamps and measured sensor values are not deployable inputs. The actual-future-dt arm is a privileged conditional comparator, **not an oracle lower error bound**: finite misspecified predictors can perform worse with more information. Predicted/past-time arms share the same fitted conditional dynamics weights and use only permitted histories/commands. Standard HGB can predict intervals but is not proposed as an innovation.

Ridge uses two RPM lags, polynomial command features and train-only scaling; six ridge penalties are chosen by validation-recording increment error. HGB direct-next-state controls are more flexible, not parameter/compute matched; their default capacity is frozen before the follow-up fit and is not tuned on test/OOD. Original fitted estimator score arrays are exported for five HGB fits; ridge validation-alpha grids are retained. Seven selected estimators are retained across two model bundles; counting twelve ridge candidates and five HGB fits gives17fits, not17independent replications.

All scores are on **measured RPM** in source units, normalized by3000². Neither RPM nor pressure is privileged physical latent ground truth. Equal recording-weighted means are descriptive, without row-wise confidence intervals or cross-environment significance claims. Ridge roots include rare regime-boundary windows; the HGB follow-up excludes crossing roots. Do not interpret that small protocol difference as a model effect.

## Prediction controls

| Group | Arm | Recordings | One-step MSE /3000² |16-step MSE /3000² |
|---|---|---:|---:|---:|
| held-out walks | persistence | 4 | 0.0003102647 | 0.001990526 |
| held-out walks | ridge_no_clock | 4 | 0.0002457619 | 0.0006906395 |
| held-out walks | ridge_oracle_future_dt | 4 | 0.0002446232 | 0.0006916213 |
| held-out walks | ridge_predicted_dt | 4 | 0.0002457142 | 0.0006921379 |
| held-out walks | HGB_no_clock | 4 | 0.00012591 | 0.000586428 |
| held-out walks | HGB_oracle_future_dt | 4 | 9.887079e-05 | 0.0005422691 |
| held-out walks | HGB_predicted_dt | 4 | 0.0001251101 | 0.0005113966 |
| fast waves | persistence | 5 | 0.0001173785 | 0.003509628 |
| fast waves | ridge_no_clock | 5 | 0.0001452621 | 0.0009880251 |
| fast waves | ridge_oracle_future_dt | 5 | 0.0001333065 | 0.0009186317 |
| fast waves | ridge_predicted_dt | 5 | 0.0001487533 | 0.001019945 |
| fast waves | HGB_no_clock | 5 | 0.0003514827 | 0.006508339 |
| fast waves | HGB_oracle_future_dt | 5 | 0.0003453023 | 0.007076884 |
| fast waves | HGB_predicted_dt | 5 | 0.0003597911 | 0.007391736 |
| slow waves | persistence | 5 | 5.58881e-05 | 0.001419878 |
| slow waves | ridge_no_clock | 5 | 7.286401e-05 | 0.0005190091 |
| slow waves | ridge_oracle_future_dt | 5 | 6.977573e-05 | 0.0004796051 |
| slow waves | ridge_predicted_dt | 5 | 7.473901e-05 | 0.0005520815 |
| slow waves | HGB_no_clock | 5 | 0.0004036022 | 0.006343171 |
| slow waves | HGB_oracle_future_dt | 5 | 0.0004190705 | 0.006655033 |
| slow waves | HGB_predicted_dt | 5 | 0.0004357452 | 0.00698145 |
| regime_jumps_single | persistence | 1 | 0.003160319 | 0.008972988 |
| regime_jumps_single | ridge_no_clock | 1 | 0.002035741 | 0.002507204 |
| regime_jumps_single | ridge_oracle_future_dt | 1 | 0.002030196 | 0.002489111 |
| regime_jumps_single | ridge_predicted_dt | 1 | 0.002036542 | 0.002517118 |
| regime_jumps_single | HGB_no_clock | 1 | 0.002158964 | 0.002619387 |
| regime_jumps_single | HGB_oracle_future_dt | 1 | 0.002143467 | 0.002549541 |
| regime_jumps_single | HGB_predicted_dt | 1 | 0.002137665 | 0.002579762 |
| regime_jumps_multi | persistence | 1 | 0.00673292 | 0.01271285 |
| regime_jumps_multi | ridge_no_clock | 1 | 0.004276349 | 0.004549586 |
| regime_jumps_multi | ridge_oracle_future_dt | 1 | 0.004264274 | 0.004528028 |
| regime_jumps_multi | ridge_predicted_dt | 1 | 0.004277602 | 0.004557187 |
| regime_jumps_multi | HGB_no_clock | 1 | 0.004677111 | 0.004804978 |
| regime_jumps_multi | HGB_oracle_future_dt | 1 | 0.004668464 | 0.004730734 |
| regime_jumps_multi | HGB_predicted_dt | 1 | 0.004660769 | 0.004765228 |


Full per-recording results, measured-unit endpoint errors and interval-time errors are in outputs/clock_rpm_controls/results.json and outputs/strong_transition_controls/results.json. The ridge-only past-time comparator is also retained there and in real_controls_summary.json. Validation recordings are reported separately and never pooled into the test mean.

The more flexible HGB improves held-out one-step prediction, while ordinary ridge can generalize better on wave rollouts. This is evidence against assuming greater nonlinear capacity is automatically a stronger simulator. It is not a novel comparison or an architecture population claim. Timing is not a uniform explanation of OOD failure.

## Failure localization

| Recording | Low-resolution blocks /32 | Fraction of evaluated rows | Fraction of HGB squared error |
|---|---:|---:|---:|
| regime_jumps_multi | 10 | 31.2516% | 97.5550% |
| regime_jumps_single | 5 | 15.6258% | 93.2791% |


Low-resolution blocks produce a disproportionate share of measured-label squared error. Reference-voltage, oversampling and hatch changes should not be lumped into the same physical-dynamics shift. Grouped metrics are retained, not hidden by removing difficult test cases.

| Regime | Sensor | Effective period grid with >99.5% coverage | Unique RPM values | Top8 atom mass |
|---|---|---:|---:|---:|
| 3 | out | 1 ms | 101 | 49.470% |
| 12 | in | 4 ms | 33 | 85.250% |
| 14 | out | 4 ms | 31 | 87.120% |
| 21 | in | 8 ms | 24 | 96.620% |
| 31 | in | 8 ms | 20 | 94.720% |


Inverted coarse-RPM periods concentrate on increasingly coarse1/4/8ms grids. The inspected source computes `float(now)-float(last_tick)` before the division. Afloat32 numeric example at70,000,000ms turns a true10ms difference into8ms; subtracting unsigned timestamps before conversion preserves10ms. This is a source-consistent explanation, **not proof that this exact firmware binary generated the archive** and not recovery of the true rotor trajectory.

## Data qualification

Both official ZIPs match publisher MD5 and full transport SHA256. All38CSVrecordings have finite numeric data and strictly increasing within-recording timestamps. The recorded targets of all64regimes match the pinned protocol target lists. There are16walk files rather than an assumed ten-seed dataset; one is short. Timestamps have a shifted origin, so no cross-file Unix-time alignment is assumed. These facts qualify permitted analysis, not a causal-identification theorem.

## Execution and uncertainty

Remote base-control runtime 20.85s; nonlinear-control runtime 122.09s. One deterministic split/fitting protocol; no multiple-initialization robustness claim. Model bundles, original retained per-iteration training scores, validation grids, raw inventory and raw per-recording metrics are retained. Terminal process exits are recorded. Full stdout remains in the tool transcript; no reconstructed text is falsely presented as an original raw stdout file.
