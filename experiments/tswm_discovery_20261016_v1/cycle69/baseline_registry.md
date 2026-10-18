# Baseline identity and execution registry

| Artifact | Actual execution | Information/provenance boundary |
|---|---|---|
| Meta-ID ckpt_430_fine-tuned_3100000_lower_noise | Complete5548418 parameters /204 state tensors, frozen | Synthetic pretraining; training records used as context only. Exact source and cached/uncached equivalence verified. Original meta training not reproduced. |
| Cached TSTransformerBenchmark | Same weights and source computation | Native cached class, not a proxy model. Source encoder output precomputed; clear cache per task/protocol. |
| dynoNet WH author artifact |323 parameters /10 state tensors, epoch8000 | Target-trained released checkpoint; selection/training protocol not independently reconstructed. |
| dynoNet Tanks author artifact |323 parameters /10 state tensors, epoch8000 | Separate target-trained checkpoint. |
| dynoNet CED author artifact |323 parameters /10 state tensors, epoch8000 | Same checkpoint for both CED trajectories; scaler uses both training records, unlike per-trajectory meta context. |
| Context mean | Executed from training values | Sanity baseline without known future inputs. |

No published result is copied into our experimental rows. Paper figures are only comparison references. Whole pretrained weights are strict-loaded, key/value-checked and SHA-verified. Only exact selected source AST definitions execute for the official loader and dynoNet adapter; downloader, TLS fallback, argparse launchers and cache deletion never execute. Sources/protocols are pinned in SOURCE_SELECTION69_V1.json, DATA_SOURCE69_V2.json and SOURCE_CONTEXT69_V1.json.

