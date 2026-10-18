# C77: frozen component composition and coordinate-offset attribution

No new research gap is accepted. F16's ordinary simulation-calibrated nonlinear map transfers to the EMPS dynamics with observed NRMSE0.021319733, compared with0.024260188 for the privileged true-sensor fitted-source diagnostic. The reverse transfer is poor, but already poor for identity sensing. Its large apparent nonlinear-composition deficit therefore does not isolate a new failure mechanism.

## Task and information boundary

This is a retrospective diagnostic using existing C75/C76 components, not a prospectively untouched or independently confirmed test. Each transfer arm uses a frozen destination source ARX64 model and a frozen map fitted on the calibration-source system. It never receives the destination's new-view calibration labels or map. Source/destination identity is supplied. The experimenter has already seen every system/condition, and diagonal maps exist as information-privileged references. No new fit, architecture, knot, source model, optimizer or DEV-selected family is introduced by the composition matrix.

The shared sensor function is artificially defined in the separately source-standardized scalar coordinates of EMPS position and F16 acceleration. These physical units are different. This is a controlled mathematical factorization over real trajectories, not evidence that an actual physical sensor is shared between the two devices. There is no within-trajectory plant change, unknown-task identification or causal intervention claim. Supplied recorded future inputs remain an explicit conditional simulation contract, especially for feedback-collected EMPS. Original official test MAT outputs remain numerically closed.

Each forecast gets the destination's128 observed history points and128 supplied future input points on the exact C75 DEV origins. No future output enters the transport function. C75 calibration-fit x SD is used only as a reporting denominator and is unavailable to that function. Privileged canonical-coordinate diagnostics invert the known synthetic generator after prediction; they do not become training signals or operational latent-state claims.

## Complete frozen matrix

Observed NRMSE, all records included. Map families are C75 innovation-monotone, C76 global-affine simulation, and C76 internally selected monotone simulation. Diagonal rows reconstruct prior predictions; off-diagonal rows are frozen transport without destination calibration. No family/source is selected using destination DEV.

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

All48 aggregate rows,216 record rows,96 support-stratum rows,36 canonical contrasts and18 fixed-grid diagnostics accompany this report. The three observation conditions and two systems are not pooled into one score.

With C76 monotone maps, EMPS→F16 identity canonical RMSE is0.951288667 and nonlinear canonical RMSE0.957353893. Its canonical MSE excess over the known-sensor diagnostic is0.776326834 already for identity and0.787903183 for nonlinear sensing; the nonlinear increment beyond identity is only0.011576350. Thus a large overall transfer error should not be relabelled as evidence of nonlinear observation-composition failure. In the reverse direction, the same ordinary family transfers successfully at this task scale. These are two physical systems, not enough independent replication to claim a general law.

History-only support checks retain all forecasts. In EMPS→F16,12.0955% of observed history values exceed the source map's outer knots, and352 of564 origins have their full history inside. Nonlinear canonical RMSE on those352 is still0.930442487 versus0.957353893 overall. This shows a failure among history-in-range windows; it does not exclude later forecast extrapolation, since no future-target-based filtering was performed. F16→EMPS has every observed history in range. On the fixed canonical grid[-2,2], the selected EMPS nonlinear inverse has coordinate bias-0.583440012 and RMSE0.606273629; F16's map has bias-0.007167726 and RMSE0.064145180. Correct within-system prediction is not proof that a calibration recovered a unique transferable coordinate.

## Post-discovery offset intervention

After reading the matrix, a separately frozen addendum profiles offsets[-1,-0.5,0,0.5,1] for every pinned inverse map. No optimizer runs, and original maps and main results stay unchanged. If h is replaced by h+delta, updating its forward inverse consistently gives an exact observation-prediction change governed by z_base+delta*(R1-1), where R1 is the homogeneous ARX response to an all-one output history. This is direct linear algebra for the fitted model, not a new algorithm or an identifiability theorem for the real plant.

EMPS's RMS(R1-1) over128 steps is0.006316588; F16's is1.315448158, about208 times larger. The fitted EMPS is near an offset symmetry, with1-sum(a)=-0.000618663; it is not exactly symmetric. Small source sensitivity is not proof of exact structural nonidentifiability or the sole explanation of source bias. A separate pure-integrator mathematical fixture verifies the exact unit-sum special case without changing either actual fitted model.

The following rows profile the selected EMPS nonlinear map. The final row is an explicitly privileged intervention: delta0.583440012 cancels its mean inverse-coordinate error on the fixed synthetic truth grid. It is unavailable to learners and is never selected as a method or from DEV.

| Offset diagnostic | EMPS internal NRMSE | EMPS DEV NRMSE | F16 DEV NRMSE |
|---|---:|---:|---:|
| fixed_-1.0 | 0.007250984 | 0.024093575 | 1.589099301 |
| fixed_-0.5 | 0.006144088 | 0.023353426 | 1.250898834 |
| fixed_0.0 | 0.006006053 | 0.023289393 | 0.851031892 |
| fixed_0.5 | 0.006895310 | 0.023903946 | 0.352307560 |
| fixed_1.0 | 0.008495159 | 0.025144911 | 0.658155954 |
| privileged_grid_offset | 0.007123214 | 0.024069513 | 0.316017477 |

The privileged correction worsens source internal NRMSE from0.006006053 to0.007123214 while improving destination F16 DEV from0.851031892 to0.316017477. That tradeoff is consistent with a source-fit calibration offset that compensates source-model error and transfers poorly to a model with very different offset sensitivity. It does not prove that all residuals are offset, that the real sensor is unidentifiable, or that a new world-model mechanism is required. The fixed+0.5 diagnostic is also retained, not substituted as a chosen result. Every one of the324 profiles is reported.

## Verification and execution provenance

All1,787,904 main forecast values, targets, support masks/metrics and contrast arithmetic reconstruct. Independent scalar ARX/interval-map replay covers432 first/last-window forecasts, maximum absolute disagreement2.842170943040401e-14. All18 diagonal rows reconstruct original stored predictions. The actual imported calibration library is hash-checked against its prior published snapshot, and an AST check confirms no new fit/optimizer/update calls in the main runner.

The offset addendum stores and rechecks9,642,240 forecast values across324 rows and54 array artifacts. Independent scalar replay covers2916 profile windows, maximum difference4.399258735077183e-14; the complete translation identity matches direct shifted parameters to1.1768364061026659e-14. Metric reconstruction differences are exactlyzero.

Three engineering failures are retained and excluded from scientific interpretation. Main V1 expected C74 per-case keys in a C75 archive and stopped before any cross-system result; V2 validates the actual pred/record_ids/origins schema. Gauge V1 imported a second process-affinity configuration and stopped before main; its command was observed but had no separate contemporaneous captured PID/log. Gauge V2's artificial unit-sum F16 fixture accidentally created a1.1797485833 unstable secondary mode, amplifying rounding error despite an exact-arithmetic invariant trajectory; it stopped before any profile. Gauge V3 uses a pure AR1 integrator only for that mathematical check. Actual model roots, masks outside the current process, source data and scientific conditions never change. All source versions, correction notes and captured terminal logs remain.

## Prior art and research decision

The primary [Bai and Reyland Wiener identification paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587721/) discusses identifiability assumptions and internal-coordinate scaling ambiguity. Its stable-system/iid-input assumptions are not asserted here, and it is not cited as a proof of our offset-specific finite-data result. The offset response above is derived and checked directly. The primary [Wiener identification project](https://www.it.uu.se/research/syscon/Ident/Wiener_Id.html) already includes nonlinear sensing and blind adaptation. Existing compositional/continual world-model boundaries remain in C71/NOVELTY_BOUNDARY71_V4.md and C76/NOVELTY_AND_DISPOSITION76_V1.md. Neither a generic coordinate ambiguity nor an unadapted transfer failure supplies sufficient novelty.

The next discriminating control is ordinary history-only offset/gain alignment or state estimation, using the same128 already observed destination history samples and no future target or extra calibration segment. If that repairs the transfer, retire this scalar composition premise as well. If not, amplitude support, source-model approximation, stochastic observation effects, coordinate conventions and complete shared/state-space/replay baselines still need resolution before a general research gap can be accepted. Do not increase F16 model capacity or tune the reported score. The independent C71 original-budget/free-simulation assessment continues. No final recommended direction or new mechanism is accepted; full goal ACTIVE/UNACHIEVED.
