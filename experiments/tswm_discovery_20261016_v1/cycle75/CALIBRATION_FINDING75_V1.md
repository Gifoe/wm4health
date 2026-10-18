# C75: ordinary calibration/refit under unpaired constructed observation changes

No final research direction is accepted. This experiment weakens the premise that a new world-model mechanism is needed merely because an unadapted predictor fails after a sensor change. Ordinary calibration and refitting recover substantial predictive quality. The frozen one-step calibration remains imperfect on F16; that residual is not yet attributed to a new mechanism failure.

## Task and provenance

This is a newly designed diagnostic using nine real EMPS/F16 original TRAIN records from C74. The declared observation functions are artificial, invertible scalar transformations in each system's source-standardized coordinates: identity;0.35+1.4*y;0.25+1.2*y+0.8*tanh(1.5*y). These are not actual replaced sensors, joint changes in physical dynamics, or a source-paper reproduction. Physical units and sampling rates remain separate across systems. EMPS's recorded feedback motor force is supplied as a future input condition; no new-policy causal effect is estimated. F16 is the official default single acceleration channel, not the whole SIMO system.

Source is the first40% of each record. Calibration is the next40%, with its own first80% for fitting and final20% for internal selection. DEV is the original final20%, using the same C74 complete origins after its256-sample gap. Source-only pooled system scaling and two source ARX64 models are newly fitted: no C74 first80% coefficients/scaling are reused. Source and calibration trajectories are disjoint. Learners receive observed calibration(u,x) only, with no paired canonical calibration y or sensor generator. The known-sensor row is a privileged diagnostic. All methods receive128 observed history samples and the same supplied128 future inputs. Every future output is recursive, without target resets.

Ordinary inverse identification uses positive-slope affine or17-knot piecewise-linear functions. Its objective is a frozen-ARX innovation equation. Four fixed identity-prior weights are selected only by recursive observed-x error on internal calibration validation, not DEV. All48 candidate fits and scores are retained. Bounds are never active in any selected adapter. Its first beta parameter is the canonical value at the first knot, not a free physical offset at x=0. A returned map is not evidence of unique physical-state identification.

Calibration-data refits comprise ARX64 and three33,281-parameter residual NARX64 MLPs per condition/system. Each neural seed runs the full2000 one-step plus500 recursive16-step updates; checkpoints are chosen using only internal128-step calibration validation. There are18 neural fits and45,000 actual neural updates in total, plus two source and six calibration ridge fits. No test-based tuning, prediction clipping, failed-record filtering, or newly proposed architecture is used. Adapter arms additionally have their fitted source model; no equal-compute, sample-efficiency or capacity-superiority claim is made.

## Complete DEV results

NRMSE is divided by pooled observed-x SD from calibration fitting only. Systems and observation conditions are not pooled. All54 method/system/condition rows and all243 record rows are retained. These are repeated windows, not independent physical-system replications.

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

All three neural seeds are shown; no best-seed claim or significance test is made. Aggregate and record-level raw RMSE/MAE, selected checkpoints, all internal checkpoint scores, all inverse-map candidates and artifact hashes accompany this report. The privileged known-sensor source model is not a universal lower bound: a refit can improve its imperfect source dynamics.

On EMPS nonlinear observation, frozen unadapted NRMSE is0.040125129, ordinary monotone calibration0.023289393, calibration ARX0.016905994 and the three nonlinear refits0.010562858/0.014706616/0.020304213. Thus this setting does not support a necessity claim for a new latent reuse mechanism.

On F16 nonlinear observation, frozen unadapted NRMSE is0.521722905, ordinary monotone calibration0.409961400 and the three nonlinear refits0.349425181/0.327292385/0.342957464. The privileged source-model diagnostic is0.299313628. Frozen one-step calibration is inferior to ordinary nonlinear refitting here; this could be an objective/noise/model-bias issue. No simulation-error-trained sensor adapter, uncertainty-aware state estimator, full nonlinear state-space/replay control or genuinely changed physical dynamics is evaluated by this cycle. These omissions prevent promotion of the residual to a general world-model research gap.

## Independent qualification

Every saved target reconstructs exactly from hash-verified C74 canonical arrays and the C75 transformation. All2,011,392 forecast values and all aggregate/record metrics reconstruct exactly. An independent scalar recurrence and NumPy Linear/SiLU implementation replay486 first/last-window forecasts across every record, condition and method. Maximum float64 disagreement is1.7763568394002505e-14 and maximum Torch-float32 versus NumPy disagreement2.1330354433857224e-6. Inverse-map innovation equations and internal-only lambda selection also reconstruct. No learner forecast helper is reused in this replay.

The EMPS source AR characteristic radius is1.0001247463 and its offset innovation multiplier1-sum(a)=-0.00061866265. It is not a stable infinite-horizon physical simulator, and the near-unit offset sensitivity requires care in interpreting calibration coordinates. F16's radius is0.9933500055 and multiplier0.3200745115. The finite128-step task is reported as specified; no root stabilization was applied after seeing DEV. Source-array, coefficient, model, prediction, protocol and raw-log hashes remain traceable. The history-only API is an information boundary, not a future-target sentinel experiment.

## Prior art and decision

Unknown static observation nonlinearities around learned dynamics are an established system-identification setting. The primary [Bai and Reyland Automatica paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587721/) analyzes assumptions, scaling ambiguity and identification under nonlinear outputs. The [Uppsala Wiener identification project](https://www.it.uu.se/research/syscon/Ident/Wiener_Id.html) discusses nonlinear measurements, blind adaptation and validation. Their guarantees are not asserted to apply to these feedback/colored-input real records. This cycle is an ordinary baseline instantiation, not a faithful reproduction of either algorithm.

The primary [ReDRAW L4DC2026 abstract](https://proceedings.mlr.press/v331/lanier26a.html) reports latent-dynamics residual adaptation for sim-to-real visual world models. Only its official abstract/bibliography was inspected here; no code was executed or full theorem checked. It blocks a generic claim that latent dynamics adaptation is new, but does not prove this exact sequential unpaired sensor/dynamics reuse task solved. Earlier SplitJEPA, JEPA-TTT and compositional-continual boundaries remain in C71/NOVELTY_BOUNDARY71_V4.md. No absence-of-results inference is treated as positive novelty evidence.

The next discriminating control, before increasing architecture complexity, is an ordinary sensor adapter trained by multi-step simulation error on the same calibration records, with state initialization and noise treatment qualified. Compare it with the existing inverse-innovation baseline and nonlinear refits under frozen information and budgets. If it closes the F16 residual, reject the need for a new mechanism on this setting. If not, separate approximation, excitation, calibration ambiguity and measurement noise before constructing a sequential joint-change test. This is a next-control decision, not a final recommended paper direction.

Official test MAT outputs remain numerically closed. The independent C71 original-budget three-seed campaign is unchanged. All arrays/weights/raw data remain on configured server G storage; only selected authored code, compact evidence and reports are published. Overall goal ACTIVE/UNACHIEVED; accepted final directions0.
