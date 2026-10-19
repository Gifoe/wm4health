# Closest-work and interface boundary

Source facts and research inferences are distinguished below. Primary snapshots, extracted reading text and immutable-revision code bytes are archived in outputs/literature. The WorldTS source directory is an archive checkout without .git; two inspected files were independently fetched through the GitHub connector at the recorded revision and matched byte-for-byte and by Git blob SHA1. Direct local HTTPS to raw/API GitHub failed; it was not used as a successful verification.

## WorldTS: existing role separation

Primary: https://arxiv.org/html/2609.31162v1 and https://github.com/decisionintelligence/WorldTS/tree/269426ea6217d7152a244da0e209d922acd46d4f .

In the numerical implementation, worldts_model.py lines154–169 construct separate target and condition encoders. Lines504–542 split the named target/covariate channels, encode future targets from target observations, and feed historical/future condition encodings into the state predictor. latent_state_loss.py supervises h_pred against h_target; it does not require predicting the next condition representation as part of h_target. The project's stated forecasting interface and multimodal evidence are relevant scope examples.

Inference: a toy which mixes current gain with all predicted observation channels does not refute this interface. The unresolved question can only concern learning useful roles/targets when those roles are unavailable, or an independently observed limitation of the separate-condition solution. A future comparison must preserve equal input information and distinguish unavailable role annotations from unavailable signal values.

## Denoised MDPs: coupled exogenous factors are already recognized

Primary: https://proceedings.mlr.press/v162/wang22c.html and https://arxiv.org/html/2206.15477v6 .

Section2.1 explicitly says that being unaffected by actions is insufficient to drop a variable, using wind during sailing as an example. Its removable independent factors do not influence the controllable state. Thus the broad claim that physically relevant uncontrollable inputs should be kept is established, and our gain-modulation construction does not contradict that stated assumption. Source-native training has not been run here.

## Existing sensitivity and balanced reduction

Primary: https://arxiv.org/abs/1304.2070 and https://www.jmlr.org/beta/papers/v27/25-1861.html . The latter is screened at abstract/metadata level here: derivative-informed input/output subspaces and reduced-basis neural operators already exist. Cycle20 actually executed the mature gradient-covariance input compressor with two output-target choices. This establishes an ordinary implementation route once the target is supplied.

Further screened boundaries: CoBRAS, https://epubs.siam.org/doi/10.1137/22M1513228 ; nonlinear balanced truncation, https://arxiv.org/abs/2604.23044 ; empirical nonlinear balancing, https://arxiv.org/abs/math/0606430 . They use state/gradient covariance or controllability/observability ideas for nonlinear reduced dynamics. These are not executed or fully read in this cycle. Neither general sensitivity weighting nor keeping dynamically important low-variance features can be claimed as new.

Research inference: cycle20's successful action-output active-subspace arm shares the automatic target-choice ingredient. It disproves a need for a new neural compressor on that pilot, while the novelty and realistic utility of target choice remain unresolved. The term 'new target selector' is a hypothesis label, not a validated novelty claim. Before a final research direction, compare with the assumptions and actual mechanisms of CDL, balanced reduction, and ordinary conditional state-space models.

## Additional primary screening after the fixed run

Action-Controllable Factorization, https://arxiv.org/abs/2510.02484 , already learns independently controllable factors from high-dimensional observations using action contrasts; its context-dependent effects and assumptions need a full source comparison before any unknown-role claim. The broader idea of recovering controllable factors by interaction goes back at least to https://arxiv.org/abs/1802.09484 . These were primary-paper screening results, not executed controls in this cycle.

https://arxiv.org/html/2610.16644v1 studies anchored action-response geometry under temporal drift, including removal of action-independent levels and profiling amplitude. It targets decision-relevant geometry, not full latent-state physical simulation. It is nevertheless a close boundary for any claim that action contrasts/response targets themselves are new. Only its problem/target sections and abstract were screened here; no native execution or full proof audit.

## Current decision

One development candidate, no accepted final novel direction. A general framework must survive independent realistic time-series and multi-step tests; a collection of modified synthetic prediction heads is insufficient. Existing successful alternatives should determine the final simplest mechanism rather than be excluded to protect a new architecture.
