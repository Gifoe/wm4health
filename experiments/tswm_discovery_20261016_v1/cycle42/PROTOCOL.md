# C42 — Unknown-parameter compositional learning eligibility screen

Frozen before comparative effects. This is a new synthetic mechanism audit, not an original-paper reproduction, a full candidate method, or a benchmark leaderboard. The framework objective is reusable time-series world modeling. C41 established only an information boundary and ordinary positive controls. Here all learned models must estimate unknown observation and transition laws.

## Question and ordinary equivalence

Does joint observation/dynamics factorization help finite-data unseen-combination prediction when both components are learned? The 11 model below is **an ordinary jointly fitted structured HMM**. It is also the exact discrete limiting case of the currently proposed factorized interface. There is no separately named new method. If it works, the result establishes a known-component inductive bias, not an unresolved research gap. If it fails, inspect optimization and identifiability before rejecting factorization.

## Data and information contract

Two independently drawn four-state worlds, four categorical sensor views and four dynamics IDs. Transitions are asymmetric, stable stochastic matrices; emissions are independently permuted noisy symbols, noise=.2. Initial latent distribution is uniform. Latent states, true matrices and future observations are never passed into training or checkpoint selection. Each sequence has 24 observations. Seen temporal cells are (v,s) with s=v or s=(v+1) mod4: eight edges, connected bipartite graph. The other eight temporal combinations are held out.

Per seen cell use nested N=16 or128 TRAIN sequences, independent64 VAL sequences. A separate32-row simultaneous four-view calibration recording under system0 is available to every model, without latent labels or transitions. Some held-out temporal cells thus have valid static calibration evidence; we explicitly test **unseen temporal combinations with anchors**, not an absence of every possible observation. No other paired-view data or labels are supplied. Score256 fresh sequences per each of16 cells using prefix8 and future16. Synthetic evaluation split is opened exactly once per selected fit; prior real-data TEST remains sealed. No hyperparameter selection using synthetic held-out cells.

## 2×2 and fairness limits

All four models are four-state, exact-inference categorical HMMs. Observation logits C(v,s) and transition logits A(v,s) use separate ordinary 8→32→16 tanh conditioners on concatenated one-hot view/system IDs. To modularize C, zero its system ID input; to modularize A, zero its view ID input. Thus every model can predict an unseen combination through shared neural weights; none uses a randomly untrained per-cell lookup at evaluation. All receive identical matrices of IDs, observations and anchors; the controlled change is conditional dependence restriction.

Nominal widths, latent size, two-layer networks, exact filtering, joint maximum-likelihood loss, optimizer and training schedules are identical. Masked input columns in modular models have zero gradients, so effective active parameter count differs; report both nominal and active counts. This is not a complete parameter/compute matched neural framework comparison. Do not use it to establish a new architecture's efficiency. No primitive recombination, nonlinear continuous latent system, or real-domain transfer is tested.

## Training and qualification

Two initialization seeds per world, two data budgets, four configurations =32 fresh fits. Batch32 uniformly drawn from pooled seen sequences; same minibatch schedule for all four configurations of each world/budget/init. Adam lr=.01, cosine to.001, 1500 updates, clip norm5, no weight decay. Exact anchor likelihood weighted by actual calibration/sequence counts, with no tuned auxiliary loss. Every250 updates select lowest **seen-cell conditional future NLL** on VAL; initial checkpoint is eligible. Store curves, all final and selected parameters, data, schedules and raw scores. Expose selection at cap and last-interval improvements; this budget is a screening cap, not convergence evidence.

Before formal fitting: exact forward likelihood compared with brute-force latent enumeration on small sequences; autograd finite-difference check; anchor formula checked by independent enumeration; future-corruption invariance; valid row sums; CPU/GPU 30-update timing qualification only, weights discarded. Choose faster device from these timings, not from predictive scores. No server-global changes. CPU is allowed if this small exact calculation is faster.

## Endpoints and falsification

Primary: conditional joint future negative log likelihood per future symbol, grouped seen vs held-out. Privileged known-parameter oracle uses the exact same observed prefix. Also record next-symbol distribution and KL to oracle; no latent-coordinate distance. Report all32 fits and initialization variation; only two data worlds, so do not claim robust cross-world statistics. Descriptive interaction I=(R10-R11)-(R00-R01), direction positive means observation factorization increases the benefit of dynamics factorization. 11 being best alone does not establish interaction or novelty. Unknown true matrices are available only to oracle and post-training diagnostics.

Failure of generic conditioners on combinations does not prove all world models fail. Strong11 success means ordinary structured modeling may suffice. If ordinary11 absorbs the whole candidate, reject a **new-method** claim at this stage; a broader framework still requires a concrete additional interaction plus complete close baselines and multiple real domains. Do not add modules or tune test noise to rescue a positive story.
