# Closest-work update, checked after held-out opening

Literature update only; no model, threshold, hyperparameter, selection or evaluation change. Primary papers verified2026-10-19 (Asia/Shanghai). No claim of running their implementations.

## Bias-aware amortized inference is already an explicit method topic

Arruda et al., *Overcoming Selection Bias in Statistical Studies With Amortized Bayesian Inference*, arXiv:2604.18319v1,20April2026. Primary: https://arxiv.org/html/2604.18319v1 . Their framework includes selection in the simulator and trains neural posterior estimation for population-level parameters, with calibration/bias diagnostics across epidemiological selection settings. This is closer methodological overlap than only generic selection-bias references. Neither selection-aware amortization nor putting a selection operator inside a simulator is a novel proposal here.

The targets still differ: their stated conditional target includes the study-selection mechanism, whereas our candidate concerns a task-level prior distorted by whole-ODE rejection and deployment under an unselected/new-IC context population. C87 does not reproduce their settings or demonstrate that their method solves this exact native-world-model problem. Conversely, changing the application to world models does not establish a new statistical mechanism. A valid remaining claim would need a specific unresolved pretraining interaction, decisive native-model evidence and a distinct solution/evaluation contribution. Those are not established.

## Shared Bayesian mechanism priors are established system-identification machinery

Huang and Michailidis, *PAC-Bayesian Meta-Learning for Few-Shot Identification of Linear Dynamical Systems*, arXiv:2609.24117v1,21September2026. Primary: https://arxiv.org/html/2609.24117v1 . It learns a transferable dynamics prior across LTI systems and performs Bayesian adaptation for new short trajectories, with dependence-aware analysis. Its linear setting is not our nonlinear-family benchmark. Nonetheless, using a learned shared prior and closed-form coefficient updates cannot be claimed new. Our cubic-library empirical Bayes comparator is an ordinary control, with an approximate weak-equation likelihood.

## Do not confuse temporal truncation with selection rejection

Tsampourakis and Elvira, *Truncated Neural Likelihood Estimation for Simulation-Based Inference in State-Space Models*, arXiv:2605.21805v1,20May2026. Primary: https://arxiv.org/html/2605.21805v1 . It addresses sequential likelihood inference and temporal scalability. The title's truncation is not evidence that it studies whole-mechanism rejection; no such equivalence is asserted here.

## Consequence

The broad method framing is crowded. C87 remains a falsification experiment, not a newly validated direction. Native errors alone cannot justify a new method when a same-supervision ordinary prior-aware comparator is strong. A positive sampler result would still need fixed-supervision/coverage attribution and a narrower differentiation claim. A null or ordinary-sufficient result should end this scoped method route, rather than trigger threshold tuning or another renamed architecture.
