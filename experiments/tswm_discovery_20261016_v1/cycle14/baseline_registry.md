# Baseline registry — cycle 14

## Native frozen models actually executed

- WorldTS source: `269426ea6217d7152a244da0e209d922acd46d4f`.
- Existing `NP` and `PJM`, H=24 and H=360, seed 2021; stage2_decoder_mix checkpoints, original resolved configs and original data SHA256 checked against the earlier sealed provenance registry.
- Native constructor, target/condition encoders and complete native forecast used. No fitting, adaptation, edited weights or new test-driven model selection.
- Inputs: 32 training-context origins per checkpoint; target OT first, six historical covariates, actual corresponding future covariates for native forecast timing. This follows the historical forecasting interface and is not a controlled causal experiment.
- NP state patch 3, latent 128, condition 64; PJM state patch 5, latent/condition 224. Raw target patches contain only 3 or 5 scalars: a generic latent transport is not automatically cheaper than re-encoding.
- Executed Python source snapshot: `outputs/native_worldts_executed_source.zip`, 67 hash-checked files. Runtime/config/checkpoint/data hashes are recorded in results.json. Large pre-existing inputs remain read-only in their original folders; no checkpoint duplication is required for the arithmetic verification.

## Controls actually executed

- Newly computed moving-window reference statistics.
- The original reference statistics applied to the shifted window; identical overlapping patches compared after excluding replicate-padding-dependent positions.
- Independent NumPy float64 arithmetic over the saved GPU float32 arrays.
- Complete native forecast vs separately timed stats plus history target/condition encoders, batch 1 and 8. These are descriptive separate timings, not an additive partition or realized caching speedup.

## Literature controls not executed this cycle

Prefix/cumulative causal normalization, Bayesian neural processes, hierarchical/GP state-space models, UWM-JEPA/BJEPA, MDN-VJEPA, OneWorld, and proximal/deconfounding methods. Do not mark them as reproduced. Existing cycle 1-13 experiments remain separate evidence with their original task definitions.

The official Bayesian Context Aggregation repository was pinned at `20cfb5230e7be2f980365a992dab138ddd027070` (36 files, AGPL-3.0) and archived without executing its code. Source was fetched locally because the server refused GitHub connections, then transferred and hash-verified on the server. The native aggregator already supports incremental Gaussian evidence updates; its sampled decoder broadcasts a latent draw across target queries. Neither feature is a proposed invention. This is source inspection, not a trained dynamics-model result.
