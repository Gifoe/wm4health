# Distinguish environment-estimator bias from remaining transition error

Execute only after complete independent verification of context resampling. This is another ordinary attribution control, with the same Meta-iLaD checkpoint and original11 context episodes, all20 pairs and160 query roots. No network weights or physical labels are optimized/provided. Query futures remain unavailable to inference and iteration selection. Freeze this protocol before refinement scores.

## Existing-method boundary

[T2-z, Section3.1.2](https://arxiv.org/html/2610.22364v1#S3.SS1.SSS2) already optimizes an environment latent from observed interactions while freezing a world model, then reuses it for other action outcomes. Direct latent optimization is prior art. Here we use ordinary image-space joint latent fitting on the existing Meta-iLaD model, rather than reproducing T2-z's5B flow-matching model or proposing a new method.

## Fixed context information and controls

Use the registered native11 contexts for each of the38 distinct environments. Every context is from an independent episode and excludes all registered queries/twins. Infer each initial context state from its first3-frame native encoder mean; infer its shared environment code from the original mean context embedding. All40 context frames are available, as in the native information contract.

Run two equal100-update context-fitting controls, each using Adam lr0.01, gradient-norm cap10 and full40-frame pixel-sum MSE. Control A optimizes only the11 context initial states with c fixed at its amortized estimate. Control B optimizes those initial states and one shared2D c per environment. Freeze model weights in both. Retain each environment's best context-only iteration, including0; in B retain its c and11 states from the same iteration. Do not select using query error, change budgets or regularize after seeing results. Report fit traces, code displacements, source-prior scales and held-out query errors. This is maximum-likelihood latent fitting, not a calibrated posterior.

For held-out queries report old and refined c with the unchanged one-pass source state; then report old prefix-fitted source state and a newly refitted source state under refined source c. The latter uses the exact earlier60-update prefix-only recipe, same initial encoder mean and per-row best prefix criterion. Its transplant uses refined target c. Include a target-prefix privileged diagnostic under refined target c; it remains extra target information. Context-only root fitting with c fixed changes no query c, so its role is an inference-budget/reconstruction control, not a different query predictor.

## Attribution and decisions

Store all actual context/query forecasts, latent estimates, optimization traces and data/source hashes on G. Report200 context-estimation updates plus60 query-state updates separately from zero weight-training fits. Independently reconstruct chosen iterations and query metrics.

Measure source/target factual competence, strict future3..39 MSE, response errors against the same structural twins and pair-level ratios. Keep all20 pairs. If the remaining residuals disappear under this same-information ordinary control, context/state amortization explains them sufficiently for this route; a new dynamics architecture is unjustified. If they persist, finite-context variance and simple latent optimization are insufficient explanations, but incorrect root coordinates, context-code bias, transition approximation and optimization quality remain competing hypotheses. A lower context reconstruction loss alone never proves better world-model semantics. No novelty or final research direction is automatically accepted.
