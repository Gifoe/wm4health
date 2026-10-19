# Native cross-system meta-ID gate: completed

One released5.548M-parameter model, unchanged weights,256 calibration systems and256 independent probe systems. Original WH generator,930 samples/system,400context +400skip +30initial +100forecast. No new model training. This is a source-native held-out-system evaluation, not a numerical claim to reproduce a particular paper table. The full generator uses future-containing normalization; physical/prospective interpretation is excluded.

| Probe predictive distribution | RMSE | Marginal NLL | Joint NLL / horizon | Marginal95% coverage | Total functional95% coverage |
|---|---:|---:|---:|---:|---:|
| Native independent | .131670 | -1.869444 | -1.869444 |95.082%|50.781%|
| Scalar temperature | .131670 | -1.868570 | -1.868570 |95.602%|51.953%|
| Calibrated bias + diagonal | .131682 | -1.863896 | -1.863896 |95.359%|51.953%|
| Same calibrated bias + full covariance | .131682 | -1.863896 | -2.014477 |95.359%|92.188%|

The diagonal and full rows have identical means and marginal variances. The total functional is sum(y)/sqrt(100); rescaling it to the sum gives the same coverage. Full-minus-diagonal per-system joint NLL difference=-.150581, system-bootstrap95% interval[-.188168,-.115990],71.875% systems lower. Scalar temperature=1.060787 does not repair dependence. Ledoit–Wolf shrinkage=.135711; all covariance/bias parameters fit only the separate calibration group. These pilot intervals are descriptive and do not establish cross-family generality.

Native zero-forecast RMSE1.010480, persistence1.366289, wrong-system-context1.406205 versus native.131670: a nontrivial reusable system-conditioned baseline is present. Wrong-context is a negative control, not a competitive algorithm. Native marginal calibration group coverage94.168%, NLL-1.899626.

Full covariance substantially repairs selected functionals but does not establish exact calibration: half contrast90.234%,DCT1=87.891%,DCT2=95.312%,alternating97.656%. Do not say all joint behavior is solved, or attribute the remaining gap to a new mechanism. The data generator and ordinary functional/generative methods still need to be excluded.

Independent SciPy density/functional re-evaluation matches all metrics within2.96e-14. Native generator regenerates first8 systems in each group exactly. CPU replays4 probe systems within5.96e-6 mean /4.02e-7 std of saved GPU predictions. Native forward loss matches SciPy within3.54e-8. All14 source files unchanged. Two dependency-import failures occurred before sample generation, with full logs preserved; isolated numba/llvmlite/odeintw were added only inside this cycle. Scientific run exit0,10.92s; independent verification exit0,5.92s.

All arrays, source/weight references, frozen protocol, exact seeds and raw logs are retained. See PROTOCOL.md and outputs/native_gate_attempt2/{results.json,independent_verification.json}. Figure: outputs/analysis/native_metaid_gate.png. No accepted novel direction; overall goal unachieved.
