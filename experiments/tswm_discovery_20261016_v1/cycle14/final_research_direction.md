# Research decision — objective still active

**Final recommended research direction: not yet established. Accepted directions: 0.**

**Research Question.** Can a general time-series world model maintain reusable predictive state with useful conditional responses and uncertainty? This is an umbrella objective, not an innovation claim. The narrower remaining intake question is whether existing amortized mechanism-posterior methods already satisfy calibrated multi-query prediction under partial observation without reset-group supervision at practical finite cost.

**Evidence.** Four unchanged trained native WorldTS checkpoints show normalization-coordinate drift on shared patches; fixed-reference controls give exactly equal states. Separate component timings show approximately 19%-24% of native forecast wall time for history encoders and statistics. These results do not demonstrate a forecast failure or a useful new transport method.

**Existing Methods.** Causal normalization directly addresses cache reuse; neural processes/Bayesian identification already represent persistent latent mechanisms; OneWorld already couples branches by shared mechanisms; probabilistic JEPA already handles coherent stochastic transitions. Earlier deconfounding models already separate action evidence and controlled transitions. No claim that these methods are universally sufficient or mutually interchangeable is made.

**Proposed Idea.** None accepted. The surviving posterior-reuse question requires a concrete residual beyond those methods before proposing a new mechanism. Combining standard components without such a residual is insufficient.

**Falsification.** An ordinary existing posterior model that matches a candidate's quality, calibration, information access and compute falsifies its claimed advantage. A different input contract or an underfit Gaussian strawman is not acceptable evidence. Unidentified observational counterfactuals are excluded.

**Feasibility.** The RTX 5090 server and frozen native WorldTS checkpoints are available. Small controlled linear/nonlinear partially observed families could validate inference correctness before a broader cross-domain study; they would be newly designed experiments, not original-paper reproductions. No expensive run should start merely to generate more result rows.

**Next Experiment.** First freeze the exact data/inference contract and reproduce an existing Bayesian/latent neural-process control's conditional path construction. Only if a finite-resource residual remains should the next discriminator compare that control with matched recurrent/particle alternatives on independent held-out systems. These findings alone do not justify new long training or a dataset-specific fix. Ordinary research remains authorized; no further confirmation is requested.

This file is a cycle decision record, not a claim that the user's research objective has been completed.
