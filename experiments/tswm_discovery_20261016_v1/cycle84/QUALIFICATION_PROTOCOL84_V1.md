# C84: qualification of the proposed adaptation question

Frozen before running the numerical qualification. This is an original contract/identifiability audit, not a reproduction of ILA, CSR, AdaJEPA or Sandwich-Residuals, and not a new method or a benchmark result.

## Question and competing explanations

The proposed direction asks a model to distinguish an observation change from a transition change and update the correct module. First establish that this label is identifiable and operationally necessary. An unknown invertible observation map can be moved into a conjugated transition model. A self-supervised predictive loss cannot distinguish input-output-equivalent parameterizations. Known action commands do not, by themselves, name every latent coordinate.

H0: component labels depend on latent coordinates; both repairs can produce the same outcomes for every allowed action. H1: these two repairs necessarily disagree on controlled predictions. We test H1 directly. A separately named latent-coordinate intervention is outside the common action interface and must not be substituted for an action-response test.

## Systems and maps

Two controlled continuous-time systems, both with B=(0,1):

* Linear: q'=-0.3q+0.6v; v'=-0.2q-0.4v+u.
* Duffing: q'=v; v'=-0.8q-0.35v-0.2q^3+u.

The source representation and observation are identity. The target observation-only world has unchanged f and y=g(x). The alternative dynamics-only world has identity observations and vector field f_g(z,u)=J_g(g^{-1}(z)) f(g^{-1}(z),u). No paired target states are given to a fitted baseline. These equivalent models are mathematical witnesses, not learned models.

Three invertible g: (i) fixed invertible linear mix M=[[1.1,0.35],[-0.25,0.9]]; (ii) M applied after (q+0.3 tanh(v),v); (iii) (q+0.25 tanh(q),v). The third preserves B exactly and fixes the origin. Thus a known actuation direction and a single common zero reset do not remove every possible coordinate ambiguity. A known physical kinematic equation can exclude this transformation; no assertion is made that the ambiguity survives such an anchor.

Seeds 2026841 through 2026844, eight independent blockwise-uniform action plans per seed, 128 steps of dt=0.05, block length eight, u in [-0.5,0.5], initial state exactly zero. This gives 192 paired action-controlled trajectories. Save all reference/alternative arrays. Main recurrence uses RK4; no clipping and no future target replacement.

For every case also set the first *named internal coordinate* to 0.8 at step 64 in each world and continue four action plans. This gives 96 pairs. These are different physical operators after changing coordinates. Report their discrepancy only as a semantic qualification, never as a failure on the common physical action interface.

## Ordinary baseline and external check

For the linear system with the linear observation map, fit a full affine one-step transition by ordinary least squares from 64 or 256 noiseless target transitions, independently for four seeds (eight fits). This learns the induced observation-space A, B and intercept, without classifying the repair location or using target physical states. Evaluate each fit on eight independent 128-step action plans. The exact parametric family is favorable to this ordinary baseline; results cannot establish arbitrary nonlinear or partially observed adaptation performance.

Independently integrate original and transformed continuous vector fields using DOP853, rtol=1e-10, atol=1e-12, for the first two action plans of the first seed in every system/map (12 pairs). Verify analytic map Jacobians against central differences at 100 independent points per map and inverse round trips at those points.

Frozen checks: main input-output equivalence max absolute error <=1e-10; independent ODE equivalence <=2e-6; Jacobian finite-difference error <=1e-7; inverse error <=1e-10; ordinary noiseless linear rollout max error <=1e-8. Failure is a qualification/implementation failure, not evidence for a gap.

## Interpretation and stop rule

If both representations agree on allowed controls, abandon the claim that a model must first recover the true repair location to predict those controls correctly. This does not establish that existing learned methods actually generalize after shift. The conjugacy argument is established state-space mathematics, not our innovation. No novel research direction is accepted from this audit. A follow-on direction must state physical anchors, its allowed interventions, and a failure that survives strong existing adaptation and joint-identification baselines. No new architecture or long training is justified solely by a component-classification score.
