# Research scope after the user's clarification

The target is a time-series world-model framework: infer useful state from historical measurements and available conditions, learn conditional evolution, and support reusable future prediction/simulation. A single dataset's tuned scores are not the research contribution. Pretraining, actions, Loop Transformers and multimodal inputs are optional mechanisms, not required destinations.

WorldTS is a scope exemplar, not a mandatory architecture. Its public implementation learns covariate-conditioned latent dynamics and then a frozen-dynamics observation decoder across numerical, textual and image-covariate tasks. References: https://github.com/decisionintelligence/WorldTS and https://arxiv.org/abs/2609.31162 . Its forecasting evidence does not itself establish identified intervention or counterfactual capability. The eventual paper must state its own supported capability precisely.

## Current candidate and what it does not yet establish

Cycles20–21 investigate whether compact self-predictive representations discard observed incoming variables which alter physical transitions. This is a mechanism probe for state learning, not the final framework or a universal world-model failure claim. Action queries are one experimental instrument for identifying relevant outputs. If a successful mechanism needs actions, the resulting capability must be explicitly limited to appropriately grounded conditions; it cannot silently claim action-free generality.

The present pilot uses invertible linear observation mixing, independent trajectories and known synthetic causal truth. It has no pretraining or cross-system parameter sharing. Larger representations repair the primary failure. Active-subspace reduction with an action-response output target also repairs it. Therefore, a new neural compression architecture is not justified by these experiments.

One plausible development route is to discover appropriate predicted outputs separately from the information needed in the conditioning state. The target-selection ingredient, not the mature compressor, would need a precise novelty argument. Existing CDL, predictive-state/control theory, derivative-informed model reduction, and modern two-space/conditional world models are close boundaries. An established conditional-state model may already be sufficient.

## Decisive missing comparator

WorldTS and ordinary conditional state-space models can receive identified exogenous covariates separately from the predicted time series. The mixed-observation toy deliberately does not supply that factor assignment. A failure on it is not evidence that WorldTS fails on its stated named-covariate interface. Before proposing a general framework, test a competent separate-condition encoder/transition baseline when the data expose those roles. If role separation alone solves the issue in realistic tasks, narrow the research to discovering unknown input/output roles or abandon the purported general gap.

No future unavailable covariate may be supplied as observed. A sensor setting, actuator command, forecast and realized future measurement require separate availability records. Observational conditional prediction is not identified causal intervention prediction.

## Advancement criteria

1. Finish the fixed independent-cohort experiment before reading its test results; preserve failures and ordinary successful baselines.
2. Separate target selection, compressor choice, teacher quality, input role information and training budget. Do not select the new method on test performance.
3. Require a source-native or realistic independent time-series setting and a non-linear observation/control check before recommending the candidate as a general framework direction.
4. Add genuinely multi-step simulation/online-state tests appropriate to stochastic future inputs, not only one-step point error. Unknown future drivers must be integrated over, not secretly supplied.
5. Retain at most three final directions and recommend one only after a precise closest-work comparison and evidence of improvability. Current accepted final directions:0.

The overall research goal remains active and unachieved. This scope note changes the evaluation standard, not the frozen ongoing scientific protocol.
