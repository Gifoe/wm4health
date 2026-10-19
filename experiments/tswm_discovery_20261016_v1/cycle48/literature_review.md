# C48 source and innovation boundaries

# C48: broaden failure discovery, then execute a memory-mechanism comparison

Goal: a reusable time-series world-model framework at WorldTS contribution scope. C47 is completed progress: its fixed-feature Laplace/hybrid recipe lost to complete likelihood. It is not retuned. No final direction is accepted.

## Candidate map and lineage

| Candidate | Parent / operation | Concrete capability and possible mechanism | Prior-art boundary | Current action |
|---|---|---|---|---|
| Joint physical state / observation clock | C47 / refine | Efficient coupled stochastic-state and timing learning | HyperNSDE unsplit already couples clocks; current loss has adverse future quality | Deprioritized; no new fits |
| Temporal-support-aware world state | observation-interface work / transfer | Reuse dynamics when measurements are interval functionals, not instantaneous samples | Integrated/overlapping-measurement GPs and field-layer JEPA already cover basic operators and resolution alignment | Preserve as a lead; a distinct learning interaction is missing |
| Explicit memory versus recurrent state | new NDDE source / instrument | Learn history-dependent evolution that remains useful during autonomous continuation | Learned constant delays, state-dependent delays, latent ODE and recurrent state are established; adding a delay is not novel | Execute actual author kernels and solver on independent real processes, with strong ordinary controls |
| Selective adaptation after mechanism change | continual-world-model work / refine | Distinguish revised dynamics from transient state/noise, reuse prior mechanisms | Adaptive World Models, online filtering/change-point models and JEPA-TTT already cover substantial machinery | Unselected; no claim that test-time updates are new |

Selection is exploratory, not numeric idea scoring. The memory route is chosen for information gain: earlier rounds did not execute this source-grounded explicit-memory baseline on real processes. It is not chosen because a weak memoryless model is expected to lose. GRU and full-history ridge controls are mandatory. A generic model's expressivity is not an automatic veto; useful finite-data learning behavior is the empirical question.

## New primary-source facts

- [NDDE, Data-Centric Engineering 2026](https://doi.org/10.1017/dce.2026.10053) learns multiple delays for non-Markovian closure. Article-linked software archives are [model code](https://doi.org/10.5281/zenodo.20113438) and [torchdde](https://doi.org/10.5281/zenodo.20113429). Exact archive checksums and Git blob correspondence have been verified, not merely inferred from matching version labels. Model commit `9b573c4bed5efee9fe9c6b765044dbd506b5bf07`; solver commit `4888c63dc84e966f6c6553277adc3851a2cca4fc`.
- [Time- and state-dependent neural DDEs](https://proceedings.mlr.press/v255/monsel24a.html) already extend beyond constant delays. Neither variable delay nor a neural memory kernel alone would be a new framework.
- [Associative-memory latent world models, 2026-10-14](https://arxiv.org/html/2610.19683v1) retrieve learned key/slot parameters from the current latent. This is useful prior capacity, but the displayed retrieval equations do not themselves implement an episode-specific recurrent history store. Historical context can still be present in the base predictor; no universal no-memory or paper-bug claim follows. Static learned priors and online state memory should be distinguished experimentally.
- [OSSM, 2026-09](https://arxiv.org/abs/2609.33566) already separates correction from autonomous propagation. [Adaptive World Models](https://arxiv.org/abs/2411.01342) and [JEPA-TTT](https://jepa-ttt.github.io/) are close boundaries for an adaptation lead. These works are reviewed here, not executed as comparative reproductions.
- [DireSMC, 2026-10-06](https://arxiv.org/abs/2610.08652) already tackles rare-event querying of diffusion surrogates. A generic guided-sampling wrapper is not an unoccupied alternative to C44.

## Interpretation contract

This round is a cross-benchmark adaptation of existing memory mechanisms, not an original-paper headline reproduction and not a proposed new method. Two real processes do not validate a universal world model. Conditional mean forecasting, delay sensitivity, numerical convergence and actual history use can locate a concrete residual; low scores alone cannot establish one. No EVAL-driven retuning, invented new architecture, forced action/causal requirement or unprecedented-operator requirement is imposed.


The frozen source comparison is now complete. Deprioritize the current explicit-delay route as a proposed new framework. It provides no stable advantage over the competent history-aware controls. On Jena, GRU is better than learned delay in both seeds for open-loop and updated forecasts. On ETT, learned versus GRU depends on seed; the weak scalar NODE has the lowest mean open-loop MSE among these arms. This does not rule out useful memory generally or establish NODE as a universal world model.
