# Baseline and data registry

| Model/source | Pinned revision | Execution status and interpretation |
|---|---|---|
| COFE, polars8948/COFE | a53409aa6d97c65b5b36b7f6b27fc5fbdb5e3df1 | Full 638917-parameter native model/loss; three 50x100-update ETTh1 runs. CLOSE source/config reproduction plus two private implementation controls. Manual uniform-origin sampler replaces the GluonTS iterable; not paper-table MATCHED. |
| KUMA, SuperBearcafedo/Review, KUMA-Official | 8ed21081366309e313f821f3dc760f6ac945ac4b | Complete disclosed PEMS03 96-to-12 full configuration; unmodified engine/model/data/evaluation, seed2025, width256, max20, early stop14. Worker0 replaces launcher10; pinned single-thread runtime. Source-faithful released-configuration reproduction. |
| Meta-iLaD | 0bc016a51b7ddb5f23b143a3fc01d2e34a1f92cd | Public repository has README/papers, zero Python source; explicitly says code forthcoming. No reproduction claimed. |
| TIME | 912291c69045d369ac45fcd5e96ecf60398c092a | Code availability archived (34 Python files). Zero-shot forecasting benchmark; no new model execution here. |

PEMS03 comes from the pinned KUMA repository, 15,800,415 bytes, SHA256
34859517425eded66a0014db63de54b95d0428c38890741a1710ed4d3b043de8.
Native data shape (26208,358,1); first feature, chronological60/20/20,
per-sensor scaler fitted on training only. Windows train15617, val5135, test5135.

ETTh1 comes from zhouhaoyi/ETDataset revision
1d16c8f4f943005d613b5bc962e9eeb06058cf07, 2,589,657 bytes, SHA256
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066.
The source Git blob hashes and paths are in `outputs/dataset_registry.json`.
Both files were fetched locally after server raw-GitHub refusal, then uploaded
with SHA verification; no synthetic substitution or TLS bypass.

The KUMA README reference is MSE0.063/MAE0.165. Our result is close to those
rounded repository values. The PMLR landing page was verified, but its linked
paper PDF was not successfully archived; this is not independent verification
of every paper-table setting. [Primary landing page](https://proceedings.mlr.press/v306/xiong26j.html).

The continuous counterfactual pilot is newly designed with known kernels;
it is neither an existing benchmark score nor a learned-world-model comparison.
