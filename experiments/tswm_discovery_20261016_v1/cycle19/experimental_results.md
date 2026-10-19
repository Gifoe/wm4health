# Completed cycle19: exploratory path-law audit with falsification controls

The data support a localized event-law discrepancy in one released Chua example. They do not establish an unsolved general world-model problem. Ordinary fitting substantially weakens architecture necessity.

## Protocol and estimand

Three original released systems, context lengths2000/1028/512, future10000 sample indices. Reuse all nine native cycle18 draws. Coordinate1 event: start below context median and hit its context90th percentile within10/50/100 samples, including the start. Primary landmarks have stride100 inside500-sample blocks. The all-offset supplement uses401 full100-sample windows per block; it was added after the first scoring attempt failed on a zero denominator. Rank diagnostics use each full path's own evaluation quantiles, not deployable model calibration. No physical sampling interval is inferred.

Block bootstrap intervals are descriptive: one reference per system, no verified mixing/coverage, and stochastic draws are not independent physical/training replications. Completed-return CDF is secondary and censor-biased. Empty conditional halves remain null with eligible counts, never a zero probability.

## Pilot and attribution

- Original Chua native all-offset CDF differences at100 are+0.1803 to+0.2243 with fixed context thresholds; rank differences+0.1491 to+0.1838. Reference rank CDF is0.392956. These are exploratory fixture-specific estimates.
- Sprott's coarse-grid apparent discrepancies shrink to rank max errors0.0120–0.0141 under all offsets, showing serious phase/grid sensitivity. The reference second-half primary cohort has zero eligible landmarks. Do not call the original31–55pp coarse discrepancy a physical failure.
- Lorenz rank max errors vary0.0549–0.1606 across native draws. There is no uniform three-system path-law failure.
- Zeroing native attention noise worsens geometry on all three systems: D_stsp1.92765/7.99066/7.39117 versus native means0.78178/2.03132/3.17179. It improves point MASE but does not repair Chua path fidelity. Exploration is purposeful in the released model.
- At common Chua origin2000, longer context improves global geometry, including when the original short-context normalization is held fixed. Rank event residuals depend on origin/history; the fixed-coordinate short wrapper reproduces the native64-step output exactly. This is attribution on the same fixture, not independent confirmation.

## Ordinary vector-field control

Twelve specifications per task are selected only on past64-step validation rollouts, then refitted on all available context. No future-based specification changes. All five10000-step rollouts remain finite. Native metrics below use the retained official metric implementation; MASE uses the true series denominator.

| Task | D_stsp | D_H | MASE10, true denominator | Rank event-CDF difference at100 |
|---|---:|---:|---:|---:|
| lorenz63_original | 0.321315 | 0.040274 | 0.084179 | +0.054386 |
| chua_original | 1.063348 | 0.041368 | 0.056040 | +0.078442 |
| sprottM_original | 0.325015 | 0.028061 | 0.012709 | +0.000000 |
| chua_common_short | 1.532648 | 0.081146 | 0.061994 | +0.167522 |
| chua_common_long | 0.623928 | 0.020496 | 0.056333 | +0.022841 |

Original-system library geometry0.3213/1.0633/0.3250 and spectra0.0403/0.0414/0.0281 improve on native means for every demonstration. At common Chua origin, the long-history library fit reduces rank-CDF difference at100 from0.1675 in its short-history fit to0.02284; its geometry improves1.5326 to0.6239. The short-history library fit is worse than native for that event statistic. History and ordinary model choice matter; no blanket superiority or universal repair is claimed. Per-context fitting and a frozen pretrained model have different budgets.

## Verification and failures

The first event job exited1 during strict JSON serialization of an empty conditional half. Its computed analogue arrays were retained. Attempt1 serializes unavailable values as null; all three control arrays are byte-identical to the failed attempt.

Independent event indexing verifies96 retained-path cohorts,90000 past-only analogue transitions, zero event-count differences and maximum reversal-moment arithmetic error8.33e-17. Attribution verifies12 array hashes,48 event cohorts and six exact native64-step prefix replays. The ordinary control verifies five context-only selection/refit replays,20 independently counted event cohorts and exact64-step replay. Linear-flow integration max error7.56e-11; regression coefficient error2.78e-17; largest ridge normal-equation residual1.88e-16. These checks establish arithmetic/provenance, not novelty or physical uncertainty coverage.

Attribution scientific job146.37s; ordinary control33.48s wall time. Existing env/source/checkpoint unchanged. All23 new raw trajectory NPZ files (including failed/retry duplicates) are mirrored and SHA256 compared across hosts, total5706190 bytes.

The three-state time-reversal witness has identical scalar autocovariances but different directed first-step probabilities0.8/0.1 and MFPT1.36986/2.32877. Ordinary transition likelihood already distinguishes it (KL1.45561). This classical witness checks metric insufficiency; it is not a new theorem, benchmark contribution or method.

![Exploratory controls](outputs/analysis/event_law_controls.png)

Decision: no new architecture or accepted research direction. Stop optimizing these three already inspected test fixtures. Original primary statistics and post-pilot analyses remain separately labeled.
