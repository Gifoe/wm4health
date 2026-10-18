# experimental results

All24 new aggregate/108 record rows are in outputs/aggregate_metrics76_v1.csv and outputs/record_metrics76_v1.csv.

| Method | EMPS identity | EMPS affine | EMPS nonlinear | F16 identity | F16 affine | F16 nonlinear |
|---|---:|---:|---:|---:|---:|---:|
| source_unadapted | 0.023877072 | 0.031211532 | 0.040125129 | 0.290841012 | 0.467700348 | 0.521722905 |
| privileged_known_sensor | 0.023877072 | 0.023877072 | 0.024260188 | 0.290841012 | 0.290841012 | 0.299313628 |
| source_monotone_inverse | 0.021224948 | 0.021224948 | 0.023289393 | 0.299780269 | 0.343349901 | 0.409961400 |
| calibration_arx64 | 0.021546380 | 0.018188129 | 0.016905994 | 0.304560297 | 0.306719600 | 0.361241490 |
| calibration_narx64_seed2026751 | 0.010242922 | 0.010242922 | 0.010562858 | 0.352836703 | 0.352836703 | 0.349425181 |
| calibration_narx64_seed2026752 | 0.012312144 | 0.012312144 | 0.014706616 | 0.298645176 | 0.298645176 | 0.327292385 |
| calibration_narx64_seed2026753 | 0.017301432 | 0.017301432 | 0.020304213 | 0.314418462 | 0.314418462 | 0.342957464 |
| simulation_affine_global | 0.022387481 | 0.022387481 | 0.026581568 | 0.289476773 | 0.289476773 | 0.337985391 |
| simulation_monotone_selected | 0.021224948 | 0.021224948 | 0.023289393 | 0.289476773 | 0.289476773 | 0.298646491 |
| simulation_monotone_terminal_global_affine | 0.016863629 | 0.016863760 | 0.015054902 | 0.288994647 | 0.288993003 | 0.298745528 |
| simulation_monotone_terminal_C75_innovation | 0.016863557 | 0.016863565 | 0.015054810 | 0.288996845 | 0.288997684 | 0.298744752 |

Independent verification reconstructed893,952 forecast values, all targets/metrics,216 recursive forecasts and six affine global optima. DEV never chooses the selected model.

See SIMULATION_FINDING76_V1.md for complete evidence and limits.
