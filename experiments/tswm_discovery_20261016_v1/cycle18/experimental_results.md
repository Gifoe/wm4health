# Native DynaMix three-case demonstration: completed

One unchanged10,256-parameter public DynaMix model generated nine10,000-step trajectories on the actual released Lorenz63,Chua,SprottM arrays. Context lengths2000,1028,512 exactly follow the original notebook. Seeds2021/2022/2023 vary native stochastic state-attention, not model-training seeds. All inference was CPU using an existing compatible Python3.13.9 environment; no new training or environment change. Scientific worker104.416s,exit0.

| System | Native geometry KL | Native spectrum distance | True-denominator MASE10 | Context-replay geometry KL | Context-replay MASE10 |
|---|---:|---:|---:|---:|---:|
| lorenz63 | 0.781781 ± 0.044414 | 0.085848 ± 0.016093 | 0.380582 ± 0.054332 | 1.903002 | 10.256993 |
| chua | 2.031320 ± 0.178232 | 0.084340 ± 0.013739 | 0.247555 ± 0.037744 | 4.815721 | 18.122721 |
| sprottM | 3.171791 ± 0.073204 | 0.097949 ± 0.007498 | 0.222527 ± 0.011761 | 4.769093 | 4.732538 |

Means±sampleSD are only three stochastic draws of one fixed model. The context-only controls are diagnostic, not a strong-model leaderboard. The native model improves both geometry and MASE over replay in these cases; spectrum distances remain small. These released demonstrations are not a reproduction of the54-system paper aggregate, an independently selected OOD suite, or evidence every dynamical property is correct. Ground-truth bounding-box escape is below0.52% in the nine draws; native histogram retention is92.86–98.89% because its boundary-bin rule excludes additional in-range points.

The current notebook reverses the MASE function's arguments, making its denominator depend on generated increments. Both exact notebook-call and true-denominator values are saved. Persistence notebook-call MASE reaches40–93 million while the true-denominator values are3.71–9.40; this is a metric-call artifact. Native-model values differ only modestly under the two conventions. This finding concerns the Python example, not the paper's Julia implementation or published tables.

Independent NumPy recomputation checks all15 model/control metric rows; maximum relative arithmetic error4.03e-7. Same-seed64-step Lorenz replay matches the saved10,000-step prefix exactly. Original22 source files plus4 demo artifacts remain hash-identical. All three NPZ result files and preflight arrays have been mirrored and hashed. Full result, original logs and verification: outputs/native_demo/. Figure: outputs/analysis/native_dynamix_demo.png.

Earlier synthetic interface preflight passed strict native state loading, finite predictions, replay and horizon-prefix checks; it was not an accuracy result. Research decision: competent original autonomous dynamics baseline established on these examples; no new failure or method claim accepted.
