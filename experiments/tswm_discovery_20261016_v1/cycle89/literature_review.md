# Literature and novelty boundaries

The broad problem of jointly plausible action branches is occupied. [OneWorld](https://arxiv.org/abs/2609.30946) learns shared-mechanism compatibility across action-conditioned video futures. Its prior-corrected exact evidence-ratio construction can recover a shared joint distribution with correct marginals under its assumptions. The learned compatibility training objective and the native flow guidance must be distinguished from terminal Gibbs reweighting of an oracle-ID critic. No released native OneWorld model was run here.

[PAWBench](https://arxiv.org/abs/2608.27345) studies probabilities of repeated futures under a fixed initial observation/action. [EDDY](https://arxiv.org/abs/2605.06553) studies marginal-preserving diffusion-sample interaction, and [Couple to Control](https://arxiv.org/abs/2605.11311) studies initial-noise couplings. [OSCC](https://arxiv.org/abs/2609.33543) explicitly constrains observation-safe, marginal-preserving branch coupling for policy-gradient estimation in imperfect-information games. Marginal preservation, common random numbers, Bayesian mixtures, IPF and optimal transport are not new primitives.

The remaining working intersection is probabilistic coherence of *different hypothetical action queries* in learned time-series world models. Adding plans supplies no new real observation. Single-query marginal preservation alone is weaker than full projective consistency of all overlapping branch subsets. Source-scope differences do not establish novelty, and this cycle does not claim a verified new problem.

[CSDI](https://arxiv.org/abs/2107.03502) is a probabilistic time-series imputation/forecasting model. This cycle transfers its native conditional diffusion core to controlled action-conditioned trajectories; it does not reproduce its original benchmark tables. Native source commit: `ermongroup/CSDI@7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`.

[DALI](https://arxiv.org/abs/2508.20294) and [Meta-iLaD](https://proceedings.mlr.press/v306/ye26p.html) already concern contextual/identifiable latent dynamics. Source inspection is feasibility evidence, not model reproduction. The DALI RSSM does pass supplied context into its transition; an earlier tentative ignored-context suspicion is not supported by source.

Final boundary: no new architecture or theorem is justified by the present evidence. A competent stochastic model plus closest existing coupling methods is still required before a method contribution can be argued.
