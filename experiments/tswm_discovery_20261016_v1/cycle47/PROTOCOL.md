# C47 — endogenous observation / stochastic-state training interaction MVP

Goal remains a reusable time-series world-model framework. C46 simple factorization is deprioritized; its EVAL is not reused. This is a new controlled method-falsification screen, not a HyperNSDE/TESAR paper reproduction or a final research direction. Prior goal turn=PROGRESS. Accepted final directions0.

## Source-grounded question and concrete candidate

Pinned HyperNSDE9d3db33ccfcf79e134d69381c6284b2692a5aef6: split training feeds only `pred_z` (deterministic component) to its event rate; residual `pred_r` is trained separately. Its unsplit code already feeds the full SDE path into the rate. Thus a full stochastic clock is an existing capability, and the split choice cannot be called an implementation bug. TESAR-CDEa57a940310ee877cb9305b588af759d572e2be2e addresses SAR/intensity weighting; it is not an assumption-free latent-SNAR solution.

Candidate learning interaction: train the joint marked-observation law while analytically marginalizing event/mark sampling conditional on the latent path. For nonnegative test function f, the ordinary conditional-Poisson identity gives

`E[exp(-sum_events f(t,y)) | z] = exp(-integral lambda(z_t) * (1-E_y|z exp(-f(t,y))) dt)`.

This is a classical Laplace-functional/Rao–Blackwell identity, not claimed new mathematics. A random finite feature bank yields a weighted minimum-distance objective; such transform matching, kernels and Rao–Blackwellization are also established ideas. The proposed recipe is provisional: its possible value is joint clock/innovation learning without a sampled-event gradient. Exact novelty remains uncertain after the targeted search. A fit or a formula alone cannot validate it.

Scope: conditionally Poisson/Cox observation process with independent Gaussian marks and administrative fixed horizon. Not a general history-dependent/Hawkes clock, informative EOS process, causal treatment model or arbitrary MNAR identification result.

## Controlled carriers and information

Two independently fitted mechanisms, same learning protocol: OU drift `-.6*z`, diffusion.7; bistable drift `1.2*tanh(1.7*z)-.75*z`, diffusion.55. Initial N(0,.8^2). True clock `.3+3.7*sigmoid(2*z)`. Observation noise sigma.2 known to all learners. Passive observation does not change physical dynamics.

Use a65-state fixed grid[-3.5,3.5],dt=.125,T64. A transition is a Gaussian Euler step quantized to the grid with endpoint tails; this finite-grid carrier is the actual ground-truth Markov model, not an exact continuous SDE. Its exact forward recursion removes particle/discretization-inference errors from the objective comparison. A positive result here still needs continuous/high-dimensional latent-state learning and independent real-domain evidence.

Marks are quantized into eight fixed bins with edges[-inf,-1.5,-1,-.5,0,.5,1,1.5,inf]. Every trajectory has a scheduled noisy initial-bin measurement, distinguished from random clock events. Given current grid state, each time/mark-bin count is independent Poisson(dt*rho*lambda(z)*Pr(mark-bin|z)); then the state transitions. No true hidden states, coefficients or dense future targets enter training/selection.

TRAIN512 independent physical trajectories/case, rho in{.5,1} equally; VAL256 same regimes; sealed EVAL256 unique physical trajectories/case, each observed under rho{.25,1,2} with the same physical path/initial measurement. EVAL histories at different rho are paired observation conditions, not independent physical trajectories. Evaluation physical scheduled future observations use fresh independent noise. All source/data/seeds/schedules retained.

## Four methods, two initialization seeds2026471/2026472, sixteen fresh fits

All share learned neural drift, learned scalar diffusion, identical neural clock, known observation link/noise and initial prior. Author `MLP_standard` class definitions are executed unchanged as selected source components; this is not the original full model/training pipeline. Drift is a stable linear term plus bounded neural residual; clock bounded positive. No warm starts or C46 weights.

1. Mean-clock likelihood: rate depends on the unconditional deterministic state mean given the scheduled initial bin, not current latent innovations. This is an explicitly source-inspired ablation, not a complete HyperNSDE reproduction.
2. Full-clock likelihood: exact joint observed-count likelihood, including empty intervals, marginalized over the full learned Markov state. Strong ordinary baseline, same observations and architecture.
3. Full-clock Laplace-feature objective: analytically integrated event/mark features, exact Markov recursion here, no sampled-event surrogate gradient.
4. Full-clock likelihood plus Laplace-feature objective: concrete interaction test, no added parameters.

TRAIN-only feature bank512, test functions on one/two temporal windows and selected mark bins plus optional initial-bin weights; nonnegative bounded weights. Empirical feature means/variances from TRAIN only. Minibatch64 features, weighted squared mean discrepancy `10*mean((Lmodel-Ltrain)^2/(var_train+.02))`. Hybrid adds this to observed NLL per65 clock/initial steps. Feature construction and weights never use VAL/EVAL. Finite features do not guarantee latent identifiability or equality of all observation laws.

Adam1e-3,clip10, batch32 trajectories for likelihood; paired per-seed schedules shared across methods. Minimum1200 updates, evaluate every200 on complete VAL exact joint NLL per65 steps for every method. Select minimum VAL NLL. Stop after three checks with relative-best improvement below.1% after minimum. Soft cap3000, extend6000 if best is at cap or either last two checks improves at least.1%; same stopping priority as source code records. Record selected/last/cap/curves and cost; do not claim training exhaustion. Qualification chooses CPU/CUDA by actual discarded training throughput. Compare quality against total measured compute; extra hybrid compute is not treated as free.

## Qualifications and evidence

Before formal fitting: compare exact Laplace recursion against direct Poisson/Markov Monte Carlo, numerical gradients against finite differences, ordinary filtering/likelihood against an independently coded NumPy recursion, scheduled-initial and future-boundary conventions. Discard qualification weights. No new full pipeline installation or global changes.

After all selections: EVAL joint observed NLL, conditional32-step scheduled-future energy/sqrt32, sample-mean MSE and90% marginal coverage using64 joint futures/origin. History uses first32 random-observation cells plus initial bin; filter transitions to state32. First future output is state32, then state33,etc. Future random clock observations are not supplied to forecasting. All methods use their actual full or mean clock during history filtering, with known rho. Known generating model provides a history-conditioned reference, never its hidden state as forecast input.

Measure clock/state coupling Cov(Nhistory,Z32 | initial-bin), using auditor hidden states only for diagnostics, and recover learned drift/diffusion/clock curves. Compare full likelihood and Laplace/hybrid quality/cost; an attractive loss value alone is insufficient. Independently score raw arrays. Two synthetic mechanisms/two seeds are a viability experiment, not real-domain or population novelty proof.

Falsification: if the complete full-clock likelihood baseline already covers the utility and the candidate has no quality/cost benefit, do not package it as a needed new world-model mechanism. If apparent gains come only from the mean-clock restriction, finite feature overfit, changed information, weak likelihood optimization or a sampling/API error, reject that attribution. A positive method result would require a fresh independent protocol and broader evidence before acceptance.
