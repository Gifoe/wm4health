# Pre-test native sampler attribution

The native50-step CSDI sampler fixes the reverse variance to the DDPM posterior variance. Even with the exact conditional-mean noise predictor, a finite discrete chain need not reproduce the true clean conditional law. Attribute this separately before calling a learned-model error an unresolved research gap.

For the linear system, propagate the sampler's affine Gaussian moments analytically with exactly the native alpha/beta schedule, initialization N(0,I), mean update and reverse variance. Compare block conditioning and sequential-prefix calls, using each training corpus's frozen normalization. Report Gaussian-law discrepancies as a supplemental oracle diagnostic, separately from the original finite sliced empirical metrics. No sampler setting is tuned or replaced in the formal experiment.

Qualify the propagation by running the unchanged native impute method with an explicitly privileged analytic Bayes-noise module in place of the learned denoiser, using8192 samples at normalization seed2021 and both fixed queries. Check means/covariances against the propagated laws with the fixed6-standard-error diagnostic. This oracle is not a learned CSDI model or a proposed method; it adds no formal neural fits and no performance claim on the nonlinear system.
