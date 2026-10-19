# Literature and novelty boundary

Three independent source-search perspectives are retained in research/modern_frontier.md,research/strong_prior_art.md,research/adjacent_mechanisms.md. Their citations and publication/preprint status are explicit. Root rechecked the exact two Falasca manuscripts and released response function. Source evidence is not numerical reproduction.

[Falasca2025 exact v4](https://arxiv.org/html/2506.22552v4), [Falasca–Zanna2026 exact v5](https://arxiv.org/html/2602.13847v5), [pinned source](https://github.com/FabriFalasca/Physics-and-causally-constrained-neural-models/tree/398082d6dcee2e9d42832dfcd39febfa48c383dd).

Falasca2025 covers stationary-versus-response mismatch,full versus partial observation,multiplicative noise and temporal averaging. Falasca–Zanna2026v5 goes further with structured discrete maps and two stochastic residual-memory layers (Sec5.4);broad structured-state-plus-memory proposals therefore lack differentiation. Its public CdV source implements paired control/forced rollouts with endpoint dt_phys forcing;at native Euler step that is appropriate. The source tree and hash manifest are outputs/source_manifest.json and upstream/falasca_zanna.

DySLIM [ICML2024](https://proceedings.mlr.press/v235/schiff24b.html) already trains invariant measures along with prediction;neural SDEs [ICML2021](https://proceedings.mlr.press/v139/kidger21b.html) already model stochastic paths;CoDA [ICML2022](https://proceedings.mlr.press/v162/kirchmeyer22a.html) already provides shared structure plus environment contexts. These defeat generic primitive claims but do not prove every interaction or practical finite-data regime solved.

Next-nearest control: [Otto/Peitz/Rowley actuated Koopman generators](https://arxiv.org/abs/2209.09977),[official implementation](https://github.com/samotto1/KoopmanGeneratorEM). Partial observations,hidden-state inference and bilinear input-modulated generators have existing direct solutions. A future transfer claim must compare this class,GRU/NARX and learned stochastic state-space models using the same histories/action data. Newer frontier notes identify action-memory and multistep latent constraints already present in BeliefGraph-JEPA/SG-JEPA;do not rename those ideas.

Current scientific synthesis: failure attribution is a prerequisite to selecting a new framework. This cycle rejects an interface confound as architecture evidence. The still open candidate is an empirical finite-data transfer interaction,not a claimed theorem or accepted Research Gap.
