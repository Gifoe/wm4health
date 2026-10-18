# Current research status after C61

Overall goal: ACTIVE / UNACHIEVED. Accepted final research directions: **0**.

The goal remains a distinct, useful and experimentally supported time-series world-model research problem, not a single-dataset optimization or a count of reproductions. No new framework is claimed.

## What the latest experiments changed

C60 completed twelve fixed-budget ordinary neural state-space/GRU fits on two measured systems. Native-component NSSM three-seed mean MSE / fit variance was 0.01017582 on N87 and 0.0097524285 on pick-place, below all tested fixed memory banks. This removes present evidence for a new mixed-memory necessity in these cases. N87 reused a historically viewed exploratory split; pick-place was one chronological device record. These are separately fitted controls, not shared-checkpoint transfer.

C61 executed existing state-age likelihood/filter/survival code and twelve fixed-budget unlabeled fits in two controlled finite-duration families. All selected models were sealed before test scoring; independent density/posterior/survival replay and score arithmetic were retained. Its prespecified outcome is **EXISTING_LEARNED_HSMM_SUFFICIENT_ON_BOTH_CONTROLLED_FAMILIES**.

- peaked: HSMM mean NLL above known-law reference 0.0073708777 nats/step; remaining-life probability MSE 0.00027206725; all-seed sufficiency pass True.
- bimodal: HSMM mean NLL above known-law reference 0.0067375658 nats/step; remaining-life probability MSE 0.00065689286; all-seed sufficiency pass True.

C61 is a prior-art exclusion, with zero real-data domains, known model order/support and fresh episode starts. It does not show that all regime, adaptation or stochastic modeling problems are solved. Known-law references are privileged numerical ceilings; fitted methods learn from observations only. The failed first support-mask adapter, corrected native extension and Python metadata/runtime boundary remain disclosed.

## What is not yet established

No unexplained cross-system capability deficit and distinct method improvement have survived the necessary existing controls. Generic stochastic latent, explicit duration, mixed memory, environment latent, assimilation-aware representation or conformal-calibration wrappers are insufficient novelty descriptions by themselves. Existing ingredients do not preclude a useful new combination; the missing evidence is an observable residual or efficiency advantage beyond competent nearest methods on independently specified data.

The next decision must follow failure attribution and primary-source comparison. Do not retune the opened C60/C61 test sets or promote a known-method repair to a new framework. Keep negative results and independent raw evidence.

## Evidence

- [C60 actual measured-system audit](cycle60/README.md)
- [C60 outcome-dependent interpretation](cycle60/INTERPRETATION60_V1.md)
- [C61 existing duration-control audit](cycle61/README.md)
- [C61 all-seed evidence](cycle61/outputs/EVALUATION61_V1.json)
- [C61 research disposition](cycle61/final_research_direction.md)
- [Additional primary-literature boundaries](cycle61/NEXT_RESEARCH_BOUNDARIES61_V1.md)

Raw data, checkpoints, latent audit truth and predictions remain on the authorized server G disk. Only code and compact evidence are published to Gifoe/wm4health; no local result-file synchronization.
