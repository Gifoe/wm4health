# Rejected explanations and stopped directions

1. **COFE forecasting improvement demonstrates successful scale disentanglement.**
   Not established. Native auxiliary noise estimates can be set to zero while
   leaving predictive distributions unchanged, both at random initialization
   and trained checkpoints. A decoder dispersion error is an alternative
   explanation for calibration under level shifts. Do not treat these source
   findings as a new architecture contribution or universal identifiability result.
2. **The source variance repair is a novel research mechanism.** Rejected.
   Location/scale equivariance is ordinary probability calculus. The single-seed
   real-data comparison demonstrates a concrete implementation repair only.
3. **COFE has an invalid posterior/prior mismatch.** Rejected as a diagnosis.
   Context-conditioned variational latents and standard-normal generative priors
   can be valid. The native sampler was retained; no unsupported prior replacement.
4. **The special10-sample branch invalidates native100-sample results.** Rejected.
   We checked the branch, but evaluated100 samples as configured.
5. **KUMA fails as an exact fixed Koopman operator.** Inadmissible strawman.
   Its released README explicitly calls the mechanism input dependent and
   Koopman inspired. Full released training is healthy, MSE0.063033101.
6. **Continuous partial observability necessarily needs a new counterfactual
   solver even for the small shared-Gaussian-alignment family.** Rejected in
   the frozen18 cases. Existing Gaussian inference and generic SDP satisfy the
   precision gate; maximum relative gap2.48756313e-05. Do not repeatedly
   tune this solved toy to create a new-method story.
7. **Smoothing once across causal models or using PSD moment bounds is new.**
   Rejected by Haugh/Singal2023 and established SDP/moment methods. Generic
   WorldKernel discussions overlap with PSD partial identification too.
8. **A speculative triangular coarse/fine transform refutes COFE's theorem.**
   Not established or executed. Initial-law, learned-class and conditional
   density assumptions must be reconciled with the unavailable full proof;
   a thought experiment is not a theorem counterexample.
9. **The runtime crashes demonstrate model weakness or out-of-memory.**
   Rejected. They are native-process execution failures, retained separately.

No accepted research direction results from these rejections. This file records
negative scientific decisions; it does not declare the autonomous goal complete.
