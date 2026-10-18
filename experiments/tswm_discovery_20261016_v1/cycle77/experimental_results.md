# experimental results

All48 aggregate/216 record rows and support/contrast/grid tables are published.

| Destination | Calibration source / family | Identity | Affine | Nonlinear |
|---|---|---:|---:|---:|
| EMPS | unadapted | 0.023877072 | 0.031211532 | 0.040125129 |
| EMPS | known_sensor | 0.023877072 | 0.023877072 | 0.024260188 |
| EMPS | EMPS / C75_innovation_monotone | 0.021224948 | 0.021224948 | 0.023289393 |
| EMPS | EMPS / C76_simulation_affine | 0.022387481 | 0.022387481 | 0.026581568 |
| EMPS | EMPS / C76_simulation_monotone | 0.021224948 | 0.021224948 | 0.023289393 |
| EMPS | F16 / C75_innovation_monotone | 0.024056542 | 0.028587997 | 0.035202438 |
| EMPS | F16 / C76_simulation_affine | 0.023667112 | 0.023667112 | 0.028716387 |
| EMPS | F16 / C76_simulation_monotone | 0.023667112 | 0.023667112 | 0.021319733 |
| F16 | unadapted | 0.290841012 | 0.467700348 | 0.521722905 |
| F16 | known_sensor | 0.290841012 | 0.290841012 | 0.299313628 |
| F16 | EMPS / C75_innovation_monotone | 0.771450664 | 0.771450664 | 0.851031892 |
| F16 | EMPS / C76_simulation_affine | 1.046674847 | 1.046674847 | 1.377807109 |
| F16 | EMPS / C76_simulation_monotone | 0.771450664 | 0.771450664 | 0.851031892 |
| F16 | F16 / C75_innovation_monotone | 0.299780269 | 0.343349901 | 0.409961400 |
| F16 | F16 / C76_simulation_affine | 0.289476773 | 0.289476773 | 0.337985391 |
| F16 | F16 / C76_simulation_monotone | 0.289476773 | 0.289476773 | 0.298646491 |

Main1,787,904 forecast values and432 scalar recurrences verified. Offset profiles add324 rows,9,642,240 rechecked values and2916 scalar recurrences. No official-test output or new model fit.

See COMPOSITION_FINDING77_V1.md for all limitations and provenance.
