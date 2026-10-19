# Next gate: native probabilistic context-conditioned dynamics, not a weak forecaster

Written after the prefix-contract results and literature screening, before downloading or inspecting any trained Explore-the-Context checkpoint. This is an independent capability audit of an old strong method, not a new architecture and not a reproduction of its full leaderboard.

## Source and artifact choice

`EmbodiedVision/explorethecontext`, corrected model-selection branch, fixed revision `f0baf6262523d9c5abaf8c0381b4db97818f7c34`. Native code already holds one context draw fixed across time and candidate actions and supports active calibration. Do not claim that persistent system samples or generic information-seeking are missing from existing methods.

Use the author-linked corrected `experiments.tgz`, Content-Length2210754075 (2.21GB decimal), published SHA1 `f24b54d4c817ac8d32fb80d1f0c8b08ededa4933`. Available server D drive space65.2GB before download. Download on server only with normal TLS verification. Preserve immutable archive; inspect member names before any extraction. Select JSON configurations/metadata and weight checkpoints, not training signals or existing calibration rollout arrays, for the three published seeds. Never select a checkpoint by a new test metric. The archive selected models reflect the author's corrected validation selection, as documented by source. Record exact files and configuration. Restricted tensor loading only; no arbitrary pickle execution to bypass an unsupported checkpoint.

## Scientific question and boundaries

For a fixed context C, the native decoder and a persistent global latent draw define a joint predictive law over a probe observation and an independent query from the same system. Does adding a draw from that law to C and rerunning the amortized encoder implement conditioning on that same law? The diagnostic requires matching future controls, fixed latent decoder, physical initial-state information, and transition noise. It must not confuse changing physical state with changing system belief.

Test prior physical first/second moments of a fixed query against moments after averaging over model-generated probe observations. Monte Carlo error needs a convergence check and paired seeds. Changing a model's own latent entropy is not by itself proof of better or worse physical uncertainty: latent coordinates may be unidentifiable. Physical query distributions and actual held-out transitions are the relevant outcomes.

Competitors: unchanged native inference; no-update control (can be coherent but uninformative); ordinary likelihood reweighting of the same persistent latent particles (a known Bayes/SMC control, not a proposed innovation). Compare proper scores on genuinely new systems after real evidence; do not accept lower consistency error by merely ignoring context. Use the native noise and allowed context sizes; unsupported interventions, context-size extrapolation and measurement duplication cannot be presented as paper failures. Three native environment types and seeds are candidates; first inspect the configurations and state/API contract, then write an executable fixed workload before new measurements.

If ordinary inference already satisfies useful physical update contracts, abandon this lead. If it fails, quantify consequences and compare known martingale/NP methods and old likelihood updates before any new-method claim. A consistency regularizer is not novel merely because applied to time series: ICLR2023 MPNP and ICML2026 From Drift to Coherence already exist. The 2026 CNP conditioning-gap paper studies its factorized joint versus re-encoding; that quantity is not automatically the same as the tower/marginal-mixture diagnostic here.

No final research direction is accepted by this document. An unsupported archive or old dependency is an engineering issue, not a scientific gap or an overall-task blocker.
