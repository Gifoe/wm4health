# Native belief-update falsification, cycle24

Prospective execution of cycle23/NEXT_EXECUTABLE_GATE.md. No new measured belief-update outcomes at writing. No new neural training, hyperparameter selection or new-method claim.

## Pre-data correction to the context sampling design

Use **contiguous prefixes** of the same50-step native rollout at cardinality0,2,5,9, rather than arbitrary disjoint subsets. The original random-subset wording would make observed starting states informative about alpha through unobserved previous transitions, invalidating the planned simple “exact physical posterior”. A full contiguous prefix has an alpha-independent initial state and exogenous inputs; its conditional transition likelihood is the exact likelihood for the available information. Nested prefixes share physical systems and are paired diagnostic conditions, not independent datasets. This correction is fixed before any new data are generated or any outcomes inspected. The method inputs and context-size support remain native. New audit design, not original-paper task/leaderboard reproduction.

## Fixed workload and provenance

- Unchanged author corrected sourcef0baf6262523d9c5abaf8c0381b4db97818f7c34 and six native noisy action-squash checkpoints selected by original validation, three training seeds each. Locate via cycle23/output manifests and require all hashes. Two actuation variants of one plant, not multiple independent real systems.
- Native alpha Uniform[-1,1], A=[[.8,.2],[-.2,.8]], state x0~N(0,I), actionUniform[-2,2], additive transition noise std.01.64 fresh systems per variant, seeds20262401/20262402.50 native CPU transitions; preserve noise RNG and independent NumPy reconstruction checks.
- Context prefixes0,2,5,9; add one probe transition for posterior, so maximum10 matches training support. Native toy context_dim1 enables deterministic quadrature.
- Probe state0, u=0 or1.5. Query state0 and informative u (0 for squash1,1.5 for squash2), fixed before/after evidence. Independent physical probe/query noise. No intervention on future labels or normalization.
- Native update / no update / Bayesian likelihood reweighting of the SAME native decoder / known-alpha-physics posterior. The latter two are established inference/privileged controls, never called innovations. Native noise likelihood uses both observed state coordinates. Physical posterior uses all observed prefix transitions and alpha-independent x0.

## Diagnostics and numerical gates

- Prior and post-encoder physical moments:128-node Gauss-Hermite;64-node cross-check on the same evidence. One-dimensional latent Bayes reweighting uses8193 standard-normal grid points over[-7,7], refinement16385 on first8 contexts; alpha posterior grid8193 over[-1,1], refinement16385. Densities are full two-output Gaussian mixtures; do not substitute Gaussian moment scores. Record truncation/integration limits.
-256 model-generated probes per context drawn from the continuous native latent Normal and its full observation likelihood. Native posterior re-encodes each outcome; integrate query first and second moments. Ordinary reweighting conditions the same prior decoder law. Quantify per-context Monte Carlo standard error and bias-corrected squared first/second-moment drift; bootstrap independent physical contexts and report signed and unsigned summaries. No-update passes drift by construction, so usefulness must also be tested.
- Real probes/queries generated from native physical truth, all methods score the same outcomes. Metrics: full predictive mixture NLL; physical query mean RMSE; variance; marginal95% coverage. Exact physical posterior here means numerical integration of a known model's correct likelihood, subject to documented grid error; it is privileged and not a learned baseline. Native decoder integration and source compatibility checked independently.
- Save per-context query mixture parameters, raw real and synthetic probe outcomes, post-update moments, physical trajectories, config/checkpoint/source hashes, scripts, logs, and timing. One model per published checkpoint seed; no test-based checkpoint selection. Physical samples shared across the three seeds are not tripled independent evidence.
- Numerical discrepancies are investigated before a model-failure conclusion. Predetermined refinement checks may trigger integration repair in a separate attempt, not scientific hyperparameter search. There is no requirement to get a positive result.

## Falsification / acceptance boundary

Abandon this broad lead if native physical updates are coherent/useful, or discrepancies are numerical, unsupported, or inconsequential. A known likelihood update fixing a failure is a baseline explanation and feasibility control, not proof of algorithmic novelty. Any future framework must show a new useful tradeoff beyond old Neural Processes, martingale methods, Bayes/SMC and active system identification. No final research direction is accepted by this protocol.
