# Training-qualified stochastic world-model falsification

Previous goal turn: PROGRESS. C89 trained native CSDI but its short3000-update pilot failed physical-support qualification. Its polynomial mixture overflowed. C90 first resolves those ordinary confounds; it does not invent a new architecture or accept novelty.

## Fixed source, data and information

Use the unchanged C89 source-pinned native CSDI core, 413569 parameters, four64-channel residual layers, eight heads,50 quadratic diffusion steps. Reuse the exact C89 TRAIN8192/CAL256 arrays, TRAIN normalization and forecasting mask. Hidden world IDs are never read by the generator. No source, preprocessing, architecture, time horizon or noise schedule is tuned against C89 EVAL. This is the same controlled cross-task transfer, not the original CSDI benchmark reproduction. The original pilot ended while CAL denoising loss was still declining; a fixed full-budget fit tests the training-sufficiency explanation.

Fit from scratch seed2026901 for exactly20000 Adam updates, batch64, lr1e-3 with multipliers.1 at15000 and18000, no early stopping. Select among steps0/1000/3000/6000/10000/15000/20000 by the unchanged fixed-RNG CAL denoising MSE averaged over all50 diffusion times on the same64 CAL trajectories per family. Float32, TF32 disabled process-locally, two CPU threads. No global environment or hardware settings change.

After selection, evaluate128 native samples on each of the first32 CAL action sequences per family, constructing all nine exact physical possibilities for reference. The model receives only the same eight initial zero states, controls and family indicator. Use the C89 prospective competence screen: energy score must beat the exact deterministic posterior mean in each family and average nearest-oracle RMS/spread ratio must be<=.5. Screening is CAL only. If seed1 fails either family, do not run seeds2/3 or open EVAL; record the insufficiency rather than launch an architecture search. If seed1 passes, complete exactly20000 updates each for seeds2026902 and2026903 with the same frozen settings and independent initializations. CAL loss alone selects each checkpoint. Do not extend budgets or modify the model based on screening outcomes.

The new EVAL seed2026905 is sealed: generate16 independent four-plan query groups per family only after the models and ordinary control are frozen and qualified. C89 EVAL and its abandoned fresh queries cannot select C90 parameters. An actual multi-action failure claim requires qualified models and closest existing controls. A good compatibility classifier alone is insufficient. No new method is accepted here.

## Ordinary mixture repair, limited to an empty-component safeguard

Frozen C89 CAL diagnosis finds one exploding component with probability~1e-6 and effective TRAIN count~.004. Other components are finite on all CAL rollouts. Correct the ordinary mixture by dropping *all* components whose expected TRAIN membership is<1 observation, then renormalize the remaining probabilities. This rule uses fitted TRAIN probabilities and4096 trajectories per family only; it is not selective removal of unfavorable EVAL samples. No coefficient, variance, feature degree or trajectory is changed or clipped. Abort if total removed probability exceeds1e-4 in either family or if any remaining component has nonfinite CAL rollout. Record the original failed result and exactly which components were removed. This is an ordinary EM support safeguard, not innovation, and it does not establish global rollout stability.

Evaluate the repaired mixture's exact finite predictive law on the same CAL queries. It must pass the same energy/physical-support screen. The full query distribution, not a selected subset of samples, is retained. Known component count and polynomial/fully observed Markov bias remain stronger assumptions than native CSDI; do not claim a matched-capacity or matched-information leaderboard win.

## Research decision

The working question remains whether qualified time-series world models need a new mechanism to produce coherent action alternatives without altering individual probabilities. Shared-latent mixtures and marginal-preserving couplings are established controls. If ordinary constructions solve the present controlled problem, retire a new-architecture claim for this setting. A possible evaluation contribution still needs differentiated prior-art analysis, actual qualified native-model failures and decision/risk relevance. Neither synthetic success nor failure alone is sufficient for the full research goal.
