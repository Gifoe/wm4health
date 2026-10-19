# C48 literature and development boundary



- [NDDE, Data-Centric Engineering 2026](https://doi.org/10.1017/dce.2026.10053) learns multiple delays for non-Markovian closure. Article-linked software archives are [model code](https://doi.org/10.5281/zenodo.20113438) and [torchdde](https://doi.org/10.5281/zenodo.20113429). Exact archive checksums and Git blob correspondence have been verified, not merely inferred from matching version labels. Model commit `9b573c4bed5efee9fe9c6b765044dbd506b5bf07`; solver commit `4888c63dc84e966f6c6553277adc3851a2cca4fc`.
- [Time- and state-dependent neural DDEs](https://proceedings.mlr.press/v255/monsel24a.html) already extend beyond constant delays. Neither variable delay nor a neural memory kernel alone would be a new framework.
- [Associative-memory latent world models, 2026-10-14](https://arxiv.org/html/2610.19683v1) retrieve learned key/slot parameters from the current latent. This is useful prior capacity, but the displayed retrieval equations do not themselves implement an episode-specific recurrent history store. Historical context can still be present in the base predictor; no universal no-memory or paper-bug claim follows. Static learned priors and online state memory should be distinguished experimentally.
- [OSSM, 2026-09](https://arxiv.org/abs/2609.33566) already separates correction from autonomous propagation. [Adaptive World Models](https://arxiv.org/abs/2411.01342) and [JEPA-TTT](https://jepa-ttt.github.io/) are close boundaries for an adaptation lead. These works are reviewed here, not executed as comparative reproductions.
- [DireSMC, 2026-10-06](https://arxiv.org/abs/2610.08652) already tackles rare-event querying of diffusion surrogates. A generic guided-sampling wrapper is not an unoccupied alternative to C44.

## Interpretation contract

This round is a cross-benchmark adaptation of existing memory mechanisms, not an original-paper headline reproduction and not a proposed new method. Two real processes do not validate a universal world model. Conditional mean forecasting, delay sensitivity, numerical convergence and actual history use can locate a concrete residual; low scores alone cannot establish one. No EVAL-driven retuning, invented new architecture, forced action/causal requirement or unprecedented-operator requirement is imposed.


Read original selected source and numerical solver; archive hashes and all98 Git blobs verified. This evidence changes baseline choice, not the novelty verdict.
