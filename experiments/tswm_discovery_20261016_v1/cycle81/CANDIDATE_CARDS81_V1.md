# C81 candidate evolution: no qualified final direction

This cycle narrows a research question without treating prior-art existence as proof that every practical deployment problem is solved. Three distinct routes remain at different evidence levels. The selected experiment is a falsification screen, not an endorsement.

## 1. Posthoc adaptation directions in a general dynamics checkpoint

Parent: C80's qualified native FIM results and heterogeneous ordinary-control performance. Operator: refine + instrument. Question: can a released general vector-field model expose a small, useful adaptation basis without retraining its encoder or relying on independent future labels for checkpoint selection?

Why it matters: a useful world model should adapt its transition law from available observations and then simulate a different initial state; matching the observed path alone is insufficient. The minimal mechanism is an external regularized linear correction in frozen decoder features, fit to weak trajectory constraints. It is deliberately ordinary. Its scientific value depends on whether pretrained features carry transferable information beyond generic alternatives.

Closest overlaps are JFR, ALPaCA, neural-ODE function encoders, FORM, WENDy, and Bayesian mixed-effect/weak-form ODE inference. Their combination cannot itself support novelty. The narrower unresolved claim is posthoc feature usefulness in an existing general checkpoint under the same observation/selection budget. Status: needs evidence; no innovation claim established.

Strongest rescue: if matched ordinary features fail while the pretrained basis improves independent new-IC predictions, investigate which directions were transferred and whether a context-only procedure can select them. Strongest threat: basis scale/rank, weak-regression noise, or an ordinary affine correction explains all gains. C81 freezes those simple controls and a development gate. A pass would still require trajectory-loss/JFR, errors-in-variables and cross-benchmark controls.

## 2. Dynamical support contracts rather than scalar forecasting scores

Parent: the difference between observed-orbit reconstruction and new-IC simulation in C80. Operator: invert + instrument. Question: which observable context properties predict whether a query trajectory is identified well enough to simulate?

This is important, but passive nonidentifiability is textbook material and C10 already illustrated it. MOD/UQNU, GP-ODE, active identification, neural ODE processes, and operator inference occupy substantial territory. A new contribution would need a specific operational contract and useful empirical decision rule beyond Hessian nullspaces, generic uncertainty, or proximity to observed states. No such mechanism is yet supported here. Status: needs mechanism and additional literature; not a fallback claim derived from one failed adaptation experiment.

## 3. Reusable dynamics under unknown observation changes

Parent: the user wants general time-series world models rather than one-dataset optimization. Operator: transfer + combine. Question: can a learned transition law be reused when measurement coordinates change and the new measurements are only partially informative?

Prior affine/PCA-anchor controls addressed restricted, sometimes privileged conditions; they do not solve general nonlinear sensor transfer. Conversely, an arbitrary unknown observation map is not identifiable without explicit assumptions. A viable route must first specify available synchronization, shared trajectories, controls, or identifiable observation families, then compare ordinary system identification and observer adaptation. Existing observer, invariant-representation and identifiable-latent-dynamics work is direct competition. Status: potentially broader, but no sufficiently concrete new mechanism or executable evidence contract yet. It is not promoted merely because route 1 fails.

## Current decision

Route 1 supplies the cheapest decisive experiment and is selected for development screening. Routes 2 and 3 remain unqualified alternatives. None is a recommended final research direction. If route 1 fails the frozen gate, report the negative evidence and do not retune the same development systems until a positive result appears.
