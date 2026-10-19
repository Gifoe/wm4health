# Frozen native operator discriminator

Freeze before any numeric result. Source MTS3 revision `ab971c0ec43df7664c4b727ca69d5b8dd0cab5da`, native `Predict` worker and HIPRSSM branches. This is one shared implementation, not two independent source reproductions or a trained-model result.

Hypothesis: repeated native moment propagation agrees with independently redrawn context uncertainty, but differs from a single persistent uncertain context. Main paper MTS3 Eq5 and Appendix Eq14 use a shared slow descriptor within a fast window. Test the implied joint process explicitly.

Admissible simple cell: two fast coordinates and two context coordinates, latent_obs_dim=1, A=aI2, C=cI2, zero means, initial fast covariance .05I2, context covariance sI2, independent Gaussian process noise .01I2, no new observations. All dimensions within the fast state have exact representable covariance, isolating cross-scale dependence from spatial covariance truncation. Prescribe parameter values in native modules; do not call this learned behavior. No actions are required for this module-level test. Maximum horizon64 is inside native mobile-robot MTS3 slow-window length75.

Factorial grid: a in {0,.5,.95,1,-.8}, c in {0,.4,1}, s in {0,.05,.5}. Inspect horizons1,2,4,8,16,32,64 for each of worker/HIPRSSM. Zero context variance, zero context coupling and a=0 are negative controls. One-step propagation should agree in all cases. Negative a checks cancellation, so the hypothesis is not restricted to underdispersion.

Controls: (1) exact augmented-state Gaussian propagation retaining state/context cross-covariance, (2) closed-form persistent-context moments, (3) closed-form independently resampled context moments. Full joint Kalman propagation is a standard control, not a proposed invention. No observations are assimilated during these open-loop tests.

Monte Carlo: 50,000 particles for six preselected cells: (a,c,s)=(1,1,.5),(.95,1,.5),(-.8,1,.5),(0,1,.5),(.95,0,.5),(.95,1,0). Seed2021+case index. Sample context once per rollout for persistent control, independently every step for fresh-context control. Save endpoints at inspected horizons, empirical variances and coverage. Sampling is used only for checking known process semantics; no estimator/hyperparameter is selected on these samples.

Report native-vs-exact variance ratios, true-Gaussian expected NLL excess and nominal95% marginal coverage. Check native arithmetic versus its fresh-context formula to1e-8, augmented versus persistent closed form to1e-8. Monte Carlo checks use finite-sample uncertainty; do not make failure claims from a single empirical fluctuation. All float64 arithmetic, prescribed transforms/noise, native source hashes and runtime recorded.

Decision gate: a mismatch establishes only a native moment-propagation limitation under this declared process. It does not establish real-data miscalibration, learned parameter relevance, a failure of every hierarchical world model, or novel mathematics. Next gate needs a faithfully trained native model and strong ordinary persistent sampling/augmented-filter controls. Relevant prior art includes PETS TS-infinity, state augmentation, Schmidt/consider filtering and Rao-Blackwellized particles. No new architecture is accepted from this test alone.
