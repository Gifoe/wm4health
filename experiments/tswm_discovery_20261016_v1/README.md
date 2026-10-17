# Autonomous Time-Series World-Model Discovery

This directory records completed controlled experiments and source-grounded audits, including negative results. **No final novel research direction is accepted yet.** The research goal is a reusable time-series world-model framework, rather than optimization of an individual dataset.

- [C49 scientific decision](cycle49/SCIENTIFIC_DECISION.md): eight conditional stochastic-world fits, 18,800 formal updates, independent prediction/score replay and a pre-registered 60/120/180-second frontier. The observation-averaged score has no stable quality/cost advantage over complete-clock ELBO; deprioritized. Two controlled process families, zero independent real domains.
- [C49 main results](cycle49/outputs/analysis_v1/RESULTS.md) and [fixed-time results](cycle49/outputs/cost_frontier_v1/RESULTS.md).
- [C52 original-data audit](cycle52/README.md): both original third Greenhouse Challenge batches acquired without images; no future-target history leakage in 1,781 native accepted windows. A valid positive control still detects the constructed source defect. This is an ordinary input-qualification audit, not research novelty.
- [C52 verification](cycle52/outputs/verification_v1.json) and [acquisition provenance](cycle52/outputs/acquisition_v2/receipt.json).

## Reproduction and artifact scope

Each experiment retains its source, frozen protocol, source/data hashes, aggregate/paired tables and verification receipts. Older script versions are historical traces; use the active versions specified in the protocol and terminal reports. Do not rerun output-asserting scripts over this published evidence snapshot; create a fresh execution directory and copy its source/protocol there. C49 setup regenerates the controlled episodes; its selected-time recognition/optimizer state was not separately saved, as disclosed in the registry.

C52 uses the unchanged [TSWM official source](https://github.com/HaochenZhang717/TSWM-Official-Repo) at commit `2e5e48540ff0da48e8192163969a6c58df1d5a3e` and the original third Greenhouse Challenge [DOI 10.4121/21960932.v1](https://doi.org/10.4121/21960932.v1), CC BY 4.0. `source_metadata52.py`, `range_inventory52.py` and `acquire_timeseries52_v2.py` preserve the exact acquisition route and repair history. Version 2 resumes the already verified first archive from version 1. The acquisition is an evidence-preserving script sequence, not a one-command cross-platform installer. The native audit uses the server storage config recorded by `configure_storage52.py`; change paths in a fresh run as appropriate.

Raw XLSX, retained signal arrays, future draws, checkpoints, environments and caches remain on the experiment server. Future large artifacts use the user-designated large disk. No result files are synchronized back to the local workstation after the user's instruction. The GitHub manifest records the exact published byte hashes. This upload adds only this new experiment directory and does not alter the repository's existing VitalDB experiments.

## Current decision

The active research goal is incomplete. Neither the negative C49 recipe nor the inactive C52 preprocessing risk establishes a new world-model gap. The repository's existing VitalDB transition-geometry and multi-timescale evidence must be audited before planning a duplicate experiment or claiming novelty.
