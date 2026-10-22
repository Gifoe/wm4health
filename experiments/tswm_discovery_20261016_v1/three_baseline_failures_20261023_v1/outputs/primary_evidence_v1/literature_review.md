# Primary literature and novelty boundaries

- [WorldTS](https://arxiv.org/abs/2609.31162): conditions latent-state formation/prediction on covariates and trains the observation decoder in a second stage. Its task is conditional forecasting; it does not automatically supply causal identification or an arbitrary autonomous simulator.
- [LatentTSF](https://arxiv.org/abs/2602.00297): frozen observation AE with latent forecasting/alignment, accepted at ICML2026. A frozen reconstruction representation is an important contrast to jointly learned predictive states. Use its actual released recipe rather than treating all latent losses as identical.
- [TSWM](https://arxiv.org/abs/2610.01842): eight-dataset benchmark and plan-fusion/directional-supervision recipe. The paper already reports divergence between forecast accuracy and declared directional consistency, including limits of supervised directions. Rediscovering that divergence or renaming directional supervision is insufficient novelty.

Checked official papers and repositories on 2026-10-23. Existing exact pins and current-main metadata are recorded by the execution audit. SOTA is protocol-specific; no unsupported universal rank is assigned. The three papers motivate a common state/dynamics question, but comparisons still require matching inputs, metrics, and task definitions.

No new mechanism is claimed yet. A replication discrepancy, stride-sensitive metric or runtime repair is not itself a novel time-series world-model framework. Prior work for a specific new hypothesis will be searched once a failure and its competing explanations are identified, before investing in a repair method.


Current experiment establishes a recipe-level failure and a training-budget confound; no novel mechanism has been proposed or verified.
