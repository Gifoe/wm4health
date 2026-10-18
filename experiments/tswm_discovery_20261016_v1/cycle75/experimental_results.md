# experimental results

All54 aggregate and243 record rows are in outputs/aggregate_metrics75_v1.csv and outputs/record_metrics75_v1.csv. The full table follows.

| Method | EMPS identity | EMPS affine | EMPS nonlinear | F16 identity | F16 affine | F16 nonlinear |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0.092456623 | 0.092456623 | 0.093548220 | 1.114733882 | 1.114733882 | 1.157819070 |
| source_unadapted | 0.023877072 | 0.031211532 | 0.040125129 | 0.290841012 | 0.467700348 | 0.521722905 |
| privileged_known_sensor | 0.023877072 | 0.023877072 | 0.024260188 | 0.290841012 | 0.290841012 | 0.299313628 |
| source_affine_inverse | 0.023806867 | 0.022854184 | 0.026916084 | 0.289519463 | 0.358936589 | 0.453677788 |
| source_monotone_inverse | 0.021224948 | 0.021224948 | 0.023289393 | 0.299780269 | 0.343349901 | 0.409961400 |
| calibration_arx64 | 0.021546380 | 0.018188129 | 0.016905994 | 0.304560297 | 0.306719600 | 0.361241490 |
| calibration_narx64_seed2026751 | 0.010242922 | 0.010242922 | 0.010562858 | 0.352836703 | 0.352836703 | 0.349425181 |
| calibration_narx64_seed2026752 | 0.012312144 | 0.012312144 | 0.014706616 | 0.298645176 | 0.298645176 | 0.327292385 |
| calibration_narx64_seed2026753 | 0.017301432 | 0.017301432 | 0.020304213 | 0.314418462 | 0.314418462 | 0.342957464 |

All2,011,392 forecast values, targets and metrics independently verified;486 independent recurrence checks. All records finite. Raw arrays/weights remain server-only.

See CALIBRATION_FINDING75_V1.md for full limits and provenance.
