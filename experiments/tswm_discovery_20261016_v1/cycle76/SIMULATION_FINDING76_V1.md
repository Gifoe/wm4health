# C76: a mature calibration control removes the apparent F16 deficit

The residual observed in C75 is not sufficient evidence for a new world-model mechanism. Keeping the source dynamics, input information,17-knot inverse-map capacity, split, observations and DEV protocol fixed, ordinary simulation-error calibration reaches F16 nonlinear-observation NRMSE0.298646491. C75's one-step innovation calibration was0.409961400; the privileged known-sensor source-model diagnostic is0.299313628. This closes the observed predictive discrepancy at this discovery setting without a new architecture. Generic scalar sensor recalibration is retired as the leading research idea. No final research direction is accepted.

## What changed and what did not

This is a post-discovery mechanistic control, not an independent test confirmation or paper reproduction. C75's source-only normalization and ARX64 coefficients are hash-verified and frozen. The same nine EMPS/F16 original TRAIN recordings, disjoint40%/40%/20% source/calibration/DEV segments, within-calibration fitting/validation split, three explicitly synthetic sensor functions and128-history/128-future-input recursive task are retained. Physical systems remain separate episodes; no changed plant dynamics, genuine replaced sensor, causal policy intervention or continual forgetting is evaluated. Original official test MAT outputs remain numerically closed.

Calibration receives only observed(u,x), the frozen fitted source model and calibration-only knot/initializer information. Source/calibration raw trajectories do not overlap. Known sensor functions and canonical calibration labels remain outside learner APIs. Training windows are wholly inside calibration-fit, stride128:61 EMPS and1812 F16 windows per condition. The identical C75 internal-validation origins choose checkpoints; DEV is never used to choose an initializer, iteration, slope, penalty, knot or model. No source refit or root stabilization is introduced.

The affine sensor control has a global solution within its fixed positive-slope family. For h(x)=a*x+b, q=1/a and r=b/a yield observed prediction R_x+q*d+r*(R_1-1). Bounded least squares over calibration simulation outputs solves for q,r. The monotone control keeps exactly17 inverse-map parameters and the original positive slope bounds. Frozen ARX latent responses are affine in those parameters, allowing exact precomputation; the forward sensor is the exact piecewise-linear inverse. Standard float64 L-BFGS-B minimizes128-step observed simulation error without C75's identity-slope prior. Two starts are fixed: learned global affine and C75's internal-selected innovation map. This changes both loss horizon and prior; it establishes sufficiency of an ordinary calibration recipe, not a unique causal attribution to either change individually.

Six affine fits and12 monotone optimizations completed. Eight monotone attempts report convergence, four reach the prespecified300-iteration cap; all statuses, projected gradients, internal checkpoint scores and terminal maps are retained. There is no claim of a nonlinear global optimum. No additional iterations or altered tolerances were granted after inspecting DEV.

## All condition-level results

NRMSE uses C75's calibration-fit observed-x SD. Every record is included. C75 controls and all three neural seeds are shown for comparison, not newly rerun here. They retain their original budgets and no best seed is selected.

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

All24 new method/system/condition rows and108 record rows are in the attached CSVs. All predictions are finite. The two terminal maps are diagnostic rows, not DEV-selected alternatives. On EMPS, internal validation selects the C75 initializer at iteration0 in all three conditions. Its nonlinear DEV NRMSE therefore remains0.023289393, despite terminal maps having lower DEV errors around0.01505. Substituting those terminal maps would be test-like DEV selection, so it is not done. On F16 nonlinear, internal validation selects the innovation-start run's iteration50. The global-affine versus innovation-start terminal nonlinear scores are similar, but this is not an independent replication or global-optimality proof.

F16's selected nonlinear simulation calibration is essentially at the privileged fitted-source diagnostic in this finite-window task, and below all three C75 nonlinear refit seed errors0.349425181/0.327292385/0.342957464. That is evidence against requiring a new mechanism for this deficit. It does not prove all world models or all observation changes solved, identify a unique physical sensor/state, establish equal-compute superiority, or cover noninvertible/multivariate sensing, new physical mechanisms, policies, long-term stability or retention. EMPS's source AR root slightly exceeds one as already reported in C75; no infinite-horizon stability claim is made.

## Independent verification

Before fitting, scalar recurrence checks qualify the exact response decomposition on calibration-fit windows. Torch/NumPy map values and central finite-difference gradients are checked at interior parameters. After fitting, an independent recurrence and interval implementation replay216 first/last DEV forecasts across all records/conditions/methods, maximum absolute discrepancy2.842170943040401e-14. All893,952 saved forecast values, targets, aggregate metrics and record metrics reconstruct; metric differences are exactlyzero. Six affine global solutions are independently rebuilt with separate homogeneous/input response recurrences and least squares; maximum parameter discrepancy8.30002733209767e-13. Internal selection and its full validation scores independently reconstruct, with no DEV reselection. Model, array, code, source, protocol and terminal-log hashes are retained server-side or in compact metadata.

## Research decision

The explanation requiring a new mechanism has failed this ordinary-control test. Both C75 neural refitting and C76 fixed-capacity calibration repair substantial prediction error. The remaining generic observation-change story lacks a distinct innovation boundary, and elementary nonlinear sensing already belongs to mature Wiener system identification. See NOVELTY_AND_DISPOSITION76_V1.md for primary-source boundaries and the newly checked component-forgetting benchmark. Do not turn this repair into a dataset-specific optimization paper or continue changing the same calibration model until an advantage appears.

The broader problem of sequential unpaired observation/dynamics reuse is not disproved; it was never evaluated by these stationary-plant scalar tests. It also has no validated new mechanism here. The next research decision must be informed by the independent C71 original-budget common-history/free-simulation results and a concrete setting beyond these repaired controls. No final direction, paper claim or independent method validation is accepted. Data, arrays and models stay on configured server G; compact authored evidence is published to GitHub without local result/data/model payload files. Overall goal ACTIVE/UNACHIEVED; accepted final directions0.
