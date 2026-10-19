# Framework development decision

The user defines the contribution level by WorldTS: a reusable state-learning and dynamics framework supported across tasks. This does not impose WorldTS's exact two-stage architecture, numerical-only inputs, multimodality, physical identifiability, action labels, one universal checkpoint or a purely new mathematical operator.

WorldTS explicitly learns encoded future states from history and available covariates, then fits a decoder. Source: https://arxiv.org/html/2609.31162v1 §§4.1–4.5. Its stated task is covariate-aware forecasting. It does not establish that every learned state is a minimal physical state or a calibrated Bayesian belief; those stronger properties must not be attributed to it or required as a definition of all world models.

## Working development route

**Observation-correctable latent dynamics.** Learn a reusable state interface for history encoding, future-state propagation, new-observation correction and decoding. The intended capability is that the same learned dynamics remain useful when an open-loop simulation is interrupted by new measurements. Candidate contribution type: model/training framework, if evidence justifies a new interaction between state learning and correction. Current novelty status: unresolved; current evidence does not establish a new mechanism.

The mechanism still to be decided is how state supervision should make an update retain the uncertainty and predictive information needed by subsequent transitions. Merely adding a GRU filter, a residual update, conditional flow, KL term or consistency loss is already represented by prior art. A proposed interaction must beat its separate parts and ordinary matched controls, not merely reduce an internal residual.

## Diverse alternatives considered

| Parent / operation | Candidate | Current weakness | Minimum useful development |
|---|---|---|---|
| C39 fixed-target evidence / refine | Correctable dynamics | AR/SSM, FlowNP, SFM and ForcingDAS already cover much of the interface | First test real evidence, common targets and quality/cost. Only then locate a state-learning interaction. |
| WorldTS heterogeneous conditions / combine | State dynamics conditioned on time-varying information availability | Broad missing-modality conditioning overlaps CAMOS and standard mask-aware models; C38 found ordinary frontend bugs | Would need a new interaction under genuine as-of updates, rather than masking performance alone. Keep as fallback, do not launch another dataset tuning run. |
| C39 internal-law finding / instrument | A framework-level simulation/assimilation diagnostic | An audit alone does not fulfill the preferred new-model direction | Useful falsification tooling for the main route; do not present as a substitute model contribution. |

## Evidence contract

1. A mechanism test with known latent dynamics and observation truth must distinguish state information loss, approximation error and conditioning failure.
2. Real datasets from multiple domains validate that the same architecture/training mechanism works; no dataset-specific modules or test tuning.
3. Open-loop rollout, interrupted rollout with actual new observations and changed observation availability are distinct tests. No causal/intervention claims from ordinary conditioning.
4. Ordinary probabilistic AR/recurrent state-space models, direct conditional generators and sequential flow updating are required comparisons. Single-site MAE gains cannot establish the contribution.

The immediate C40 code is only a source-qualified ordinary diagnostic. It uses two real carriers to test whether the motivating difficulty survives a strong recurrent latent-state baseline and native FlowNP. It does not yet train the final proposed framework. SFM/other ordinary controls remain outstanding, so even a positive residual cannot qualify a research gap.

## Falsification

If ordinary models already combine useful simulation, observation correction and reasonable cost, discard the broad motivating claim. If a new method reduces law discrepancy but not real held-out predictive loss, discard that objective as the main contribution. If gains need per-dataset architecture/threshold choices, the framework claim fails. If undertraining or capacity explains the gap, repair the baseline before drawing a mechanism conclusion.

No candidate is accepted as novel or submission-ready. The selected question is a work priority, not a positive result.
