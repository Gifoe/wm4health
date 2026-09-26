# wm4health

## VitalDB intervention-grounding diagnostic — seed 0

第一轮实验的代码与结果。研究问题：准确预测未来生理状态，是否仍可能没有学到药物暴露与给药时序信息？

**本轮结论：Outcome C — Gap not supported（未充分支持所提出的研究缺口）。** 该结论不等于证明模型具有因果或完整生理机制正确性。全队列时移影响较小；较大剂量变化的补充分析在初始诊断后加入，报告保留了这一限制及全部原始分析。

### Results

- [完整实验报告 / Final report](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/FINAL_REPORT.md)
- [主诊断表](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/main_diagnostic_table.csv)
- [原始事件结果](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/transition_diagnostic_table.csv) · [较大剂量变化的补充结果](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/large_transition_diagnostic_table.csv)
- [图表（8 张，PNG/PDF）](experiments/vitaldb_intervention_grounding_seed0_v1/plots)
- [数据摘要](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/data_summary.json) · [病例划分](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/split_caseids.json)
- [配置](experiments/vitaldb_intervention_grounding_seed0_v1/config.yaml) · [代码](experiments/vitaldb_intervention_grounding_seed0_v1/src) · [运行脚本](experiments/vitaldb_intervention_grounding_seed0_v1/scripts)
- [模型权重](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/checkpoints) · [完成检查](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/completion_checks.json) · [协议修订记录](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/protocol_amendments.md)

最终队列包含 **495 个病例、493 位患者**；train/validation/test 分别为 345/74/76 个病例，按患者隔离。仅下载 VitalDB 数值轨迹及元数据，未下载波形。CE 是设备计算的 TCI 参考状态，未进入预测模型的输入或训练损失。

| Model | BIS MAE, full 5-minute trajectory | PPF CE R² | RFTN CE R² | Matched wrong-action MAE increase |
| --- | ---: | ---: | ---: | ---: |
| State-only | 4.640 | — | — | — |
| Action Transformer | 4.375 | 0.675 | 0.743 | 14.98% |
| RSSM | 4.320 | 0.780 | 0.896 | 16.61% |

MAE above is averaged over the full future trajectory and weighted equally by patient. It is distinct from the 5-minute endpoint MAE in the main diagnostic table. Wrong-action degradation uses the matched subset and its corresponding unperturbed baseline.

![Forecasting results](experiments/vitaldb_intervention_grounding_seed0_v1/plots/figure1_forecasting.png)

### Reproduction and artifact scope

See the [experiment README](experiments/vitaldb_intervention_grounding_seed0_v1/README.md) and [recorded environment](experiments/vitaldb_intervention_grounding_seed0_v1/outputs/pip_freeze.txt). Run commands from the experiment directory:

```bash
cd experiments/vitaldb_intervention_grounding_seed0_v1
```

The repository contains the original experiment source, configuration, metadata snapshots, split IDs, preprocessing records, metrics, figures, logs, trained checkpoints and frozen linear-probe coefficients. Original source files are preserved byte-for-byte; their SHA-256 hashes are recorded in `outputs/final_source_sha256.json`.

Raw numeric track CSVs, aligned per-case arrays and full per-window prediction/latent arrays remain on the experiment server and are **not included in this repository**. `src/prepare.py` reacquires numeric data through the official VitalDB API and rebuilds the case arrays; the pipeline regenerates training and evaluation outputs. `scripts/finalize.py` requires those regenerated prediction arrays. The saved completion checks describe verification performed on the server before publication.

The experiment README's server paths describe the original run. To reproduce the recorded report classification after reviewing regenerated metrics, replace its `A_OR_B_OR_C` placeholder with `C` only if the evidence still supports that classification. Rerunning the pipeline writes into the experiment's output directories; use a separate checkout to retain this published snapshot.

### Data provenance

Data come from the [VitalDB Open Dataset](https://vitaldb.net/dataset/) via its [official API](https://vitaldb.net/docs/?documentId=API%2FWeb_API_OpenDataset.md). The original [VitalDB propofol/BIS notebook](https://github.com/vitaldb/examples/blob/master/ppf_bis.ipynb) is retained as a reference under `outputs/official_ppf_bis.ipynb`. Third-party data and the reference notebook retain their upstream terms and attribution.
