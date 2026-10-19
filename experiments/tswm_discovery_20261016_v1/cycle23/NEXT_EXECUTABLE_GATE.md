# Fixed next native workload, before any new belief-update outcomes

The new research focus is a reusable time-series world model that preserves the meaning of one uncertain system across branches and evidence updates. This is a candidate capability to investigate, not an accepted innovation. No new architecture is selected and no dataset-specific loss tuning is authorized by this protocol.

## First, inexpensive falsification on a strong native implementation

Use six already downloaded, author validation-selected `Explore the Context` models: noisy `action_squash_1` and `action_squash_2`, published seeds1,2,3 (run IDs3/7/5 and6/8/9). Each was trained for50k updates, context_dim1, maximum context cardinality10. Use the existing preflight manifest to locate weights, not a new test score. Other18 checkpoints are available but must not be called tested for belief updates yet. These two variants share a two-state plant and differ in actuation nonlinearity; they are **not** evidence from two independent real physical systems.

Generate64 fresh physical parameter draws alpha~Uniform[-1,1] per variant using seed20262401/20262402, native initial state N(0,I), native A=[[.8,.2],[-.2,.8]], native action U[-2,2], process noise std.01, and50 observed transitions. Select disjoint context subsets of cardinality0,2,5,9, fixed before evaluation. Use source-native environment kernels and record every transition; next added observation keeps cardinality<=10. Do not repeat cycle17's temporal covariance calibration.

Probe at the native calibration initial state x=0 with u=0 or1.5. This yields an informative/uninformative contrast reversed by the two native actuation maps. Query a separate fixed virtual branch from x=0 with the informative action for that map. The question concerns uncertainty about the persistent system parameter, so the query initial state remains fixed when evidence is added. Do not accidentally compare different current physical states.

## Compare distinct belief-update mechanisms

1. Native amortized context encoder plus unchanged persistent-latent decoder.
2. No-update control. This can be internally coherent while discarding useful evidence; low drift alone is not success.
3. Ordinary Bayesian likelihood reweighting of the **same native decoder's** one-dimensional latent prior. Use full transition noise likelihood. This is an old inference control, not a proposed new method.
4. Known physical alpha prior/likelihood posterior, a privileged positive control. It must not be represented as a general learned model.

Measure physical query first and second moments before a probe and after averaging updates over model-generated probe outcomes. Use256 probe draws per context with fixed seeded noise, confidence intervals across independent physical contexts, and Monte Carlo standard errors per context. The native one-dimensional latent permits deterministic integration; start with128-node Gauss-Hermite for physical prior/post-encoder moments. Likelihood reweighting may need a dense standardized latent grid because noise.01 is narrow; predeclare8193 grid points over +/-7 prior standard deviations and an integration check with16385 points on the first8 contexts. Also check64vs128-node native physical moments on the same fixed evidence. Numerical accuracy is a gate, not a method hyperparameter selected for a better scientific result.

For actual held-out probe outcomes, compare full physical predictive mixture log scores, mean error and coverage against no-update, reweighting and privileged alpha posterior. Preserve raw mixture parameters and actual query outcomes. Any alpha-grid approximation must pass a grid-refinement check; do not silently replace mixture scoring by Gaussian scoring and report them as the same density. Three checkpoint seeds are replications of training, not three independent datasets. Report group and per-context effects; signed averages can cancel.

## What would be learned

- If native updates are accurate and coherent within numerical/MC uncertainty, abandon the broad belief-consistency lead.
- If a violation appears only under unsupported context lengths, duplicated evidence, changed input normalization or a different state initial condition, reject that interpretation.
- If ordinary likelihood reweighting repairs the phenomenon, it establishes a known inference explanation and a competent baseline. It does **not** establish a novel framework. Any subsequent proposal must offer a demonstrable generalization, memory or inference-cost advantage over this baseline and known NP/martingale methods.
- If an internal discrepancy does not affect actual held-out prediction or information decisions, lower its practical priority. Do not call a model-generated-data artifact a consequential research gap.

This workload is an audit newly designed here, not an exact reproduction of the published active-calibration result. It has not yet been executed. Final accepted research directions remain zero. Broader16-dimensional latent pendulum/mountain-car tests are contingent on a useful residual in this low-dimensional gate, not an automatic expansion quota.
