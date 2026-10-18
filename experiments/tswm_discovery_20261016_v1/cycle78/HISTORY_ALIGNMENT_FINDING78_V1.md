# C78: ordinary same-history alignment repairs the frozen composition deficit

The scalar composition deficit does not justify a new world-model mechanism. With the source dynamics and base observation maps frozen, a standard two-coordinate ridge regression on the already supplied128 destination-history points reduces EMPS→F16 nonlinear NRMSE from0.851031892 to0.308192768. The privileged known-sensor fitted-source diagnostic is0.299313628. No additional new-view calibration segment, future output, true sensor function, neural layer or new knot is provided to the estimator. Retire this scalar calibration/composition premise as the leading research idea, rather than tune its remaining small residual.

## Information, task and adaptation

This is a retrospective falsification control on C75-C77's already inspected discovery DEV, not an untouched test confirmation or general theorem. Two different physical systems are normalized separately using their original first40% source observations. The three identical mathematical sensor functions are synthetic in those normalized scalar coordinates; they are not a physical sensor common to motor position and aircraft acceleration. System identity is supplied. There is no within-trajectory physical-mechanism change, continual forgetting or causal policy-effect evaluation. EMPS recorded future force is supplied conditionally; no unknown future policy action is inferred. Original official test MAT outputs remain numerically closed.

Every C77 map/source/destination/condition is retained. Each prediction fits nuisance coordinates anew from its128 observed u/x history. It forgets that fit after this origin. Source ARX64 and the18 base map artifacts remain unchanged. Let h be the base inverse, and h_corr=(h+r)/q. The forecast is the exact old forward map applied to R_h+q*d+r*(R1-1). All128 future outputs are recursive, with no future-output reset. Past information is identical to that supplied to frozen C77 predictors. Future inputs enter forecasting only after fitting; the fit API has no future-plan, future-label, generator or destination-calibration argument.

Three ordinary recipes were frozen before C78 scores. Offset innovation ridge fixes q=1 and estimates r from64 one-step historical equations. Affine innovation ridge fits q,r to the same equations using box-constrained least squares. Their regularizer is the destination source ARX's canonical first40% training residual MSE, independently rechecked; it includes model error and is not called a calibrated noise/posterior variance. The third recipe fits q,r by64-step observed-history simulation, initializing from the first64 of the128 supplied history points, with fixed history-SD normalization and0.001 quadratic shrinkage. Bounds q[0.02,50],r[-2,2], solver budgets and tolerances are fixed. Corrected overall slopes may leave the original base-map slope bounds; only its affine coordinate correction changes. All three recipes are reported, with no DEV-selected arm.

There are31,428 per-origin nuisance fits:10,476 for each recipe, across108 aggregate cells. These are small inference-time regressions, not new neural training runs or independent physical replications. All solvers report success and all predictions remain finite. Bound-hit counts and parameter quantiles are retained, including weakly constrained EMPS offsets that reach bounds. Nonlinear solver success is not a global-optimum proof. No equality of inference computation with the frozen baseline is claimed.

## Complete method/source/destination tables

Observed NRMSE uses the unchanged evaluation-only C75 calibration-fit x SD. Three base-map families, all conditions, both directions and diagonals follow. Frozen rows are referenced from C77, not rerun as fresh fits. No map family, method or seed is selected on DEV.

### C75_innovation_monotone

| Destination | Map source | Alignment | Identity | Affine | Nonlinear |
|---|---|---|---:|---:|---:|
| EMPS | EMPS | frozen | 0.021224948 | 0.021224948 | 0.023289393 |
| EMPS | EMPS | offset_innovation_ridge | 0.020563962 | 0.020563962 | 0.022170192 |
| EMPS | EMPS | affine_innovation_ridge | 0.014326197 | 0.014326197 | 0.014761635 |
| EMPS | EMPS | affine_past_simulation | 0.016895130 | 0.016895130 | 0.016727413 |
| EMPS | F16 | frozen | 0.024056542 | 0.028587997 | 0.035202438 |
| EMPS | F16 | offset_innovation_ridge | 0.020304599 | 0.022051395 | 0.029638737 |
| EMPS | F16 | affine_innovation_ridge | 0.011843462 | 0.011949317 | 0.013266182 |
| EMPS | F16 | affine_past_simulation | 0.013961146 | 0.012947832 | 0.014070934 |
| F16 | EMPS | frozen | 0.771450664 | 0.771450664 | 0.851031892 |
| F16 | EMPS | offset_innovation_ridge | 0.319362423 | 0.319362423 | 0.329298951 |
| F16 | EMPS | affine_innovation_ridge | 0.298649358 | 0.298649358 | 0.308192768 |
| F16 | EMPS | affine_past_simulation | 0.315805058 | 0.315805058 | 0.328524469 |
| F16 | F16 | frozen | 0.299780269 | 0.343349901 | 0.409961400 |
| F16 | F16 | offset_innovation_ridge | 0.310344573 | 0.362709389 | 0.438489681 |
| F16 | F16 | affine_innovation_ridge | 0.290744613 | 0.292187395 | 0.315710568 |
| F16 | F16 | affine_past_simulation | 0.313020077 | 0.314300310 | 0.338715277 |


### C76_simulation_affine

| Destination | Map source | Alignment | Identity | Affine | Nonlinear |
|---|---|---|---:|---:|---:|
| EMPS | EMPS | frozen | 0.022387481 | 0.022387481 | 0.026581568 |
| EMPS | EMPS | offset_innovation_ridge | 0.020331488 | 0.020331488 | 0.023088140 |
| EMPS | EMPS | affine_innovation_ridge | 0.010845004 | 0.010845004 | 0.011148659 |
| EMPS | EMPS | affine_past_simulation | 0.014930809 | 0.014930809 | 0.014642878 |
| EMPS | F16 | frozen | 0.023667112 | 0.023667112 | 0.028716387 |
| EMPS | F16 | offset_innovation_ridge | 0.019356904 | 0.019356904 | 0.023636647 |
| EMPS | F16 | affine_innovation_ridge | 0.011812670 | 0.011812670 | 0.012618158 |
| EMPS | F16 | affine_past_simulation | 0.013765608 | 0.013765608 | 0.013997200 |
| F16 | EMPS | frozen | 1.046674847 | 1.046674847 | 1.377807109 |
| F16 | EMPS | offset_innovation_ridge | 0.303666830 | 0.303666830 | 0.370744132 |
| F16 | EMPS | affine_innovation_ridge | 0.287117417 | 0.287117417 | 0.317614398 |
| F16 | EMPS | affine_past_simulation | 0.309261400 | 0.309261400 | 0.340842054 |
| F16 | F16 | frozen | 0.289476773 | 0.289476773 | 0.337985391 |
| F16 | F16 | offset_innovation_ridge | 0.301441886 | 0.301441886 | 0.355931372 |
| F16 | F16 | affine_innovation_ridge | 0.287087703 | 0.287087703 | 0.317665148 |
| F16 | F16 | affine_past_simulation | 0.309260130 | 0.309260130 | 0.340834557 |


### C76_simulation_monotone

| Destination | Map source | Alignment | Identity | Affine | Nonlinear |
|---|---|---|---:|---:|---:|
| EMPS | EMPS | frozen | 0.021224948 | 0.021224948 | 0.023289393 |
| EMPS | EMPS | offset_innovation_ridge | 0.020563962 | 0.020563962 | 0.022170192 |
| EMPS | EMPS | affine_innovation_ridge | 0.014326197 | 0.014326197 | 0.014761635 |
| EMPS | EMPS | affine_past_simulation | 0.016895130 | 0.016895130 | 0.016727413 |
| EMPS | F16 | frozen | 0.023667112 | 0.023667112 | 0.021319733 |
| EMPS | F16 | offset_innovation_ridge | 0.019356904 | 0.019356904 | 0.017467480 |
| EMPS | F16 | affine_innovation_ridge | 0.011812670 | 0.011812670 | 0.014166309 |
| EMPS | F16 | affine_past_simulation | 0.013765608 | 0.013765608 | 0.015587290 |
| F16 | EMPS | frozen | 0.771450664 | 0.771450664 | 0.851031892 |
| F16 | EMPS | offset_innovation_ridge | 0.319362423 | 0.319362423 | 0.329298951 |
| F16 | EMPS | affine_innovation_ridge | 0.298649358 | 0.298649358 | 0.308192768 |
| F16 | EMPS | affine_past_simulation | 0.315805058 | 0.315805058 | 0.328524469 |
| F16 | F16 | frozen | 0.289476773 | 0.289476773 | 0.298646491 |
| F16 | F16 | offset_innovation_ridge | 0.301441886 | 0.301441886 | 0.313049843 |
| F16 | F16 | affine_innovation_ridge | 0.287087703 | 0.287087703 | 0.297664497 |
| F16 | F16 | affine_past_simulation | 0.309260130 | 0.309260130 | 0.319695175 |

The known-sensor reference for identity/affine/nonlinear is respectively0.023877072/0.023877072/0.024260188 on EMPS and0.290841012/0.290841012/0.299313628 on F16. That reference uses the same approximate fitted source dynamics and a privileged exact sensor map; it is not the true plant, a universal lower bound or an operational method. An adapted local predictor can outperform it by compensating model error. All108 aggregate/486 record rows and complete canonical diagnostic metrics are attached, including cells where adaptation worsens a strong diagonal baseline.

For the C76 monotone source maps, all three ordinary recipes substantially improve the problematic EMPS→F16 nonlinear transfer: offset innovation0.329298951, affine innovation0.308192768, past-simulation affine0.328524469, versus frozen0.851031892. The affine recipe's median estimated r is0.601375765, obtained entirely from the provided history without the previous privileged true-grid correction. In the reverse F16→EMPS nonlinear transfer, the three recipes score0.017467480/0.014166309/0.015587290 versus frozen0.021319733. These are discovery-window results, not independent multi-system confirmation or a universal ranking of estimators.

The simpler one-step affine regression performs adequately; developing a new architecture for this measured deficit is unnecessary. The remaining difference to the privileged F16 reference does not distinguish source-model approximation, finite-history estimation, support or noise. No additional hyperparameter, layer, source refit or new knot is introduced to pursue it.

## Independent verification and retained artifacts

All4,022,784 stored future predictions/targets and all aggregate/record metrics reconstruct. Independent scalar ARX/interval-map replay covers972 first/last origins, max absolute difference2.3092638912203256e-14. All20,952 innovation solutions are independently reconstructed from historical sufficient statistics by univariate least squares or complete two-variable box-face enumeration; max parameter difference2.446931546273845e-13. Every10,476 nonlinear past-simulation objective is independently reconstructed via corrected-map historical recursion; maximum cost difference3.410605131648481e-12. Source residual scales, analytic Jacobians and map/response equivalence are qualified. No future-label sentinel is claimed from API inspection alone.

All per-origin q/r arrays, solver metadata, predictions and raw data remain on configured server G, with full paths/hashes and code to reproduce them. Compact publication includes all record metrics, parameter distributions, bound/solver counts and artifact hashes. Actual retained terminal log SHA256 and exit0 are recorded. This cycle had no numerical execution failure. Separately, legacy C76/C77 publication receipt leaf names had retained a75 suffix; correct scoped aliases were added by exact-hash copy, preserving originals, scientific artifacts and Git history. The alias receipt is included for provenance.

## Research decision

The original failure has now survived an engineering check, a fixed-capacity simulation objective control, frozen cross-composition/identity/support tests, a declared privileged offset intervention, and ordinary same-history alignment. Those checks support a mature calibration/coordinate explanation, with no distinct new mechanism established. This is negative evidence for the proposed lead, not failure of the broad world-model research goal and not a claim that all nonlinear/multivariate/partially observed/continual world modeling is solved.

Stop the normalized scalar sensor line. Continue the full time-series-world-model objective outside this repaired setup, using current primary literature, actual source and independent mechanism evidence. Recent spectral-representation and partial-trajectory papers are screened as prior-art boundaries in NOVELTY_AND_SCOPE78_V1.md, not adopted as untested ideas. The C71 original-budget three-seed/common-history free-simulation assessment remains unchanged and runs to completion. No final research direction, paper-level innovation or new method is accepted. Overall goal ACTIVE/UNACHIEVED; accepted final directions0.
