# C81: Frozen foundation features as an adaptation basis — development screen

This is a new, exploratory experiment, not a paper reproduction or a qualified novel method. The parent observation is C80: a released FIM-ODE checkpoint and ordinary context-only weak identification have heterogeneous new-IC errors. The C80 tutorial systems must not be tuned again.

## Question and prior-art boundary

Do features extracted posthoc from the frozen FIM-ODE decoder provide useful adaptation directions, beyond ordinary dictionaries, random features, affine residual correction, and the same features with no pretrained readout prior? Learned-feature Bayesian readouts (ALPaCA), posthoc Jacobian adaptation (JFR), neural-ODE function encoders, weak identification (WENDy), FORM, and Bayesian weak/collocation models already cover the component ideas. No novelty, Bayesian calibration, causal, latent-state, or general world-model claim follows from this screen.

## Data, eligibility, and opening

Use the C80-pinned OpenFIM commit, base checkpoint and ODEBench revision without further downloads. Select eligible two-dimensional systems solely by public structural metadata: exclude C80 systems 26, 42, 28; require at least two author-marked successful solutions for parameter set zero, each with 512 points. Sort eligible IDs. Development systems are the first six of eligible[0::3]; reserved evaluation systems are the first six of eligible[1::3]. Save their exact IDs before loading inference features or scoring trajectories. Other systems are not part of this screen. The source JSON was parsed in C80; these are previously unscored systems, not an assertion of cryptographically unopened data or absence from pretraining.

For each development system, use native load_odebench_system: parameter index zero, regular 50% subsampling, 3% multiplicative Gaussian noise, RNG seed 2026810 reset per system. Only successful trajectory zero supplies 256 context states/times. Context data are fully observed; there are no actions or hidden sensors. Independent initial-condition trajectory one is the main new-IC outcome. Clean trajectory zero is reconstruction, not forecasting. Neither clean trajectory enters fitting, candidate selection or feature construction. Known query initial state and time grid are permitted inputs. Public equations are not used for fitting or source-specific dictionary choices.

## Feature extraction qualification

Keep all checkpoint parameters fixed. Capture the input of the native decoder's final Linear layer. Verify its affine readout and native normalization reproduce native cached vector fields on context queries within float32 tolerance (atol=rtol=2e-5). Verify the cached field agrees with a direct native model_forward call; verify feature context is only the declared prefix. Qualification uses context only and must pass before new-IC scoring. Hash checkpoint and source again after the experiment.

## Estimators, same observations and solver

All correction models use the same weak observed-path constraints. For context-only mean m and coordinate standard deviation s, z=(x-m)/s. Form trapezoidal feature integrals and state increments over every window of widths 8,16,32. A residual model fits B=delta(z)-integral(f_native(x)/s); the zero-centered model fits B=delta(z). Scale columns by their weak-design RMS, solve ridge in float64, and retain every fit/solver failure. This is an ordinary estimator under a working regression model, not an exact Bayesian trajectory posterior. No numerical differentiation or future labels are supplied.

Methods:

- Unmodified native FIM vector field with cached context encoding (same mathematical field as model_forward).
- Native hidden features plus bias; weak ridge correction to native field.
- The same native features fitted from zero rather than the native readout.
- Gaussian random Fourier features plus bias with exactly the same feature count as the native hidden features plus bias; frequencies N(0,1), phases uniform, RNG 202681, evaluated on z; correction to the native field.
- Small affine correction using [1,z0,z1,f_native0/s0,f_native1/s1].
- Ordinary weak polynomial degree-three and polynomial-plus-sin/cos dictionaries, using the C80 frozen 12-candidate recipe. The common trigonometric library is source-informed from earlier C80 work; do not describe it as blind equation discovery.

For each correction family, select ridge from {0.001,0.1,10} by trajectory MSE on noisy context points 191:256. Fit only points 0:192; recompute context encoding, state normalization, features and integral windows using this prefix alone. Validation starts at observed state 191. After selecting by (MSE, ridge) among successful candidates, rebuild all feature/statistics from the full context and refit. The ordinary dictionary follows the same split and C80 grid {1e-6,.001,1} x thresholds {0,.03}. No candidate may use validation observations to construct its encoder or normalization. If all candidates fail, report the family as failed; do not silently substitute another method.

Use RK45 rtol=1e-4, atol=1e-6 for all methods, at the original clean time grid. Declare 20,000 RHS calls and absolute state 1e6 as failure guards, without clipping or selective time truncation. Store all selected models, contexts, predictions, candidates and raw solver diagnostics on the server data disk. Report failures separately; do not average only successes to claim superiority. Report NRMSE using clean reference pooled standard deviation for scoring only, per-system outcomes, adaptation and rollout cost, and correction parameter counts.

## Decision rule and limits

This is a six-system development falsification screen; no hyperparameter changes after opening. A candidate can justify a later independently frozen evaluation only if pretrained features improve new-IC error relative to native FIM on at least four of six systems, retain complete finite trajectories for every system where native succeeds, and beat the strongest ordinary correction/weak-dictionary control on at least four systems with at least 10% lower median paired NRMSE ratio. These are exploration gates, not statistical significance or a publication guarantee. Otherwise retire this particular readout-only route without concluding all world-model adaptation is solved. Reserved evaluation remains closed in this cycle.

Even a pass requires JFR/trajectory-loss adaptation, WENDy errors-in-variables, matched effective-rank controls, a genuinely different benchmark, and multiple context noise seeds before any mechanism or generalization claim. Full FIM reconstruction and converged controls must be measured before a latency advantage is accepted. No new architecture or long training is authorized by a mere development-screen pass.
