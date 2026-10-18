# Executed model registry

| Model/control | Execution | Information and boundary |
|---|---|---|
| NormWear-2 full pretrained | 186714897 parameters, one frozen checkpoint | Physiological pretraining; this is controlled-system transfer. History-only predict is separated from known-plan modes. |
| Native empirical transitions | Indexed and same-view action controls;3 MC seeds | Fixed10 clusters,4 fused sampled trajectories; seeds are not neural fits. |
| PCA/ridge latent operator |9 VAL configurations; selected latent_ridge_k16_a0.1 on EVAL | Same exact encoder/decoder; context-only PCA and regression. Linear model/truncation are limitations. |
| Author meta-identification Transformer | 5548418 parameters, strict state load; one frozen checkpoint |8 disjoint400-point contexts,30 observed query initial pairs,100 known inputs; no future output embedding. Different pretraining. |
| Raw ARX and cubic ARX | Per-context fits;3 alpha settings each on VAL | Coefficients learned from history. Cubic dictionary includes the true functional form; structural solvability control. |
| Context mean / last output | Direct execution | Same observed history; no future inputs. |

Source revisions, bytes and SHA256 hashes are in MODEL_FILES68_V1.json and inherited C66 source receipt. Exact coefficients/bases and forecast arrays are server-only. Third-party repo has no LICENSE file in the pinned tree; source/weights are not redistributed.

