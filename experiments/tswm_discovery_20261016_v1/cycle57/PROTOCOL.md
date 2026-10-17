# C57: existing-normalizer controls in a nonlinear sensor world model

## Question and novelty boundary

Do competent existing reference-normalizer controls repair learned dynamics and plan-conditioned conditional means after the C56 true-point gradient finding? This is a mechanism experiment with adapted baselines, not an original-paper reproduction, a new method, or an accepted research direction. Generic contrastive gradient correction, running normalizers, normalizing Kalman filters and stop-gradient EM are existing ideas. A positive result here can eliminate a candidate gap; a negative result alone cannot establish originality or rule out optimization error.

## Controlled systems and data

Use a two-dimensional stationary controlled Gaussian state: z[t+1]=rho*R(angle[action[t]])*z[t]+sqrt(1-rho^2)*epsilon, rho=.95, action angles [-.2,.3], observation y=z+sqrt(.3)*eta. Actions are independent randomized Bernoulli(0.5). z[0] is standard Gaussian. Every action matrix preserves the same unit invariant marginal. This avoids an action-dependent marginal-reference mismatch while retaining genuine selectable future plans.

Two paired sensors observe the identical states, noises and actions: linear x=y, nonlinear x=asinh(y). The inverse is not supplied to the neural recognition models. The latter is nonlinear sensing of controlled linear latent dynamics, not arbitrary nonlinear latent dynamics. Dimensions are deliberately small enough to qualify exact inference. These are synthetic mechanism cases, not real-data or across-domain evidence.

Fixed data seed 5700, independent train/validation/held-out sequences of lengths 32: 512/128/256 sequences. Store input arrays separately from physical-state diagnostic labels. Training reads only train inputs/actions. No validation or held-out selection: every model is the fixed terminal model. Held-out inputs and labels remain unopened by training and scoring until all model and train-only map seals exist.

## Model and controls

All neural controls share an MLP recognition mean (2-64-64-2, ReLU), constant positive diagonal site precision and the same controlled prior family. Learn rho in (.05,.995) and two rotation angles; process covariance is (1-rho^2)I, initial prior N(0,I). Initial rho=.5, angles=[0,0], site precision=[1,1]. This restricted prior family contains the true dynamics, provides the same structural information to every control, and is an adaptation rather than the paper's unrestricted action network. Every neural method has exactly matched initialized weights per sensor and seed.

Five neural controls: native same-batch reference loss (B=32); independent reference B=32; running-normalizer adaptations with gamma=.9 and .1 and independent reference B=32; full 512-sequence training reference. Running estimates are maintained in log space per training sequence/time, initialized on first visit and detached in the gradient coefficients. Gamma=.9 is the inspected SogCLR source default; both values are fixed in advance and all results are retained. Target IDs are sampled without replacement within each update; independent reference IDs are sampled separately. The independent-reference comparator matches the running variants' encoder/reference work. Full references cost more and are an attribution upper control, not an equal-compute method claim.

Use the existing SogCLR principle and source boundary, Optimization-AI/SogCLR ef289b52ed72d22d8b3c4e8a76d698c117f02a44, sogclr/builder.py. This adaptation is not an execution of the authors' image pipeline. Detaching exp(log_kernel-log_running_mean) yields the stochastic-compositional gradient; gamma=1 must match the independent-reference loss gradient exactly.

Add a proper-likelihood control using a learned coordinatewise invertible sensor map sinh(k*x)/k, k in (0,2), initialized at .2, and learned diagonal Gaussian noise. This instantiates the existing normalizing-Kalman principle with a parametric sensor family suited to these cases. It is a strong structured control, not an unrestricted neural baseline or an original-paper NKF reproduction. It receives no hidden states or true inverse parameters in fitting.

## Execution and qualification

Use seeds 2021,2022,2023: 30 neural fits at 10,000 Adam updates, lr=.001, B=32, full differentiation, source reporting jitter 1e-5. Retain every failed fit; never replace a failed terminal run silently. Proper-likelihood controls use full-train L-BFGS, max 200 iterations/250 evaluations, strong-Wolfe search, tolerances 1e-7/1e-9, same prior initialization, no seed/model selection. Report its distinct capacity, objective and compute budget.

The exact Gaussian algebra is ported to the existing CUDA PyTorch runtime for the single RTX5090. This is a native-loss-matched port, not the original JAX Trainer. Before any formal update, qualify joint/marginal moments, loss and all parameter gradients against unchanged pinned RP-GSSM source a75225d5edceb972bb872dc9a335608733567dd4 in the existing JAX CPU environment; both full and fixed-posterior conventions are checked. Respect the native recomputed adjacent cross covariance after reporting jitter. Qualify proper likelihood against an independent joint Gaussian calculation and its flow Jacobian. Check CUDA float32 against the qualified float64 implementation, and the running-estimator gradient at gamma=1. Failure stops execution and is preserved. No package, driver, upstream or old experiment mutation.

## Evaluation and decisions

After all terminal model hashes are sealed, fit one affine physical-state mean diagnostic from each model's train smoothing means to train z, then seal all maps before any held-out scoring. This privileged diagnostic is identical across methods and measures recoverable state means; it is not an observation decoder, gauge proof or covariance-calibration metric. Do not map posterior covariance through this regression.

Use the causal prefix x[:12] and preceding actions only. Report physical filtered-state R2, factual realized-state MSE, and conditional-mean error at h={1,5,10,20}. For the latter, compare to the exact true filtering conditional mean propagated under known plans. Compare factual, all-zero, all-one and alternating future plans. Also report changed-plan minus factual conditional-mean effect error against the simulator truth, not individual counterfactual outcomes. No future observations enter a prefix state estimate. All physical truth and true inverse use is confined to diagnostics/evaluation.

Primary endpoint: h=20 changed-plan effect conditional-mean MSE averaged over the three changed plans. Report all seed outcomes, means and between-seed spread, plus paired sequence uncertainty within each fixed fitted model. No checkpoint, hyperparameter, method or representation selection on held-out scores. Distinguish seed variation from sequence uncertainty. Also report learned rho error, train-only state-mean fit, train objectives with a common full reference, runtime, updates, parameters, encoder/reference counts and peak CUDA allocation. Running surrogate values are not comparable training objectives.

If existing normalizer controls approach the proper-likelihood/oracle diagnostics and remove the native deficit, drop generic finite-reference correction as a candidate innovation. If they do not, inspect tracking/optimization and representation capacity before claiming a gap. No final direction is accepted from two small synthetic systems alone; a broad reusable framework still needs independent nonlinear/real-system and prior-art evidence.

Sources and compact logs stay in the new cycle57 directory. Data, checkpoints and larger arrays use the configured G: omnieeg project subtree. Publish evidence to wm4health with prior directories preserved. No local result-file synchronization. The overall research goal remains active and unachieved.
