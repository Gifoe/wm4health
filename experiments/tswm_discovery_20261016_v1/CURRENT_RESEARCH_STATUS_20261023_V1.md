# 当前研究状态 — 2026-10-23

研究目标：ACTIVE / UNACHIEVED；已接受最终新颖研究方向：0。当前主线固定为WorldTS、LatentTSF、TSWM。

## 本轮已经执行的证据

- 重新核对20个WorldTS/LatentTSF旧checkpoint、40个TSWM子集checkpoint与125个TSWM源码文件；不计为60次新训练。
- 6个原生潜态重复、6个官方raw对照、12个固定100epoch归因对照完成；另有前12模型的逐小时推断审计。
- 独立重算原生/逐小时指标；seed42权重和预测逐字节复现。
- 更密集的逐小时查询没有导致预想的性能退化，该假设被否定。
- 固定100epoch裁决：`RAW_ADVANTAGE_SURVIVES_FIXED100_ON_BOTH_HORIZONS`。完整逐种子结果见下方报告。

当前残差不等于信息丢失或一般world-model缺陷；已知的decoded-loss、非线性预测器及AE来源仍待分别排除，单一Electricity分数不能确立可投稿的通用机制。

[本轮完整结果](three_baseline_failures_20261023_v1/outputs/matched_evidence_v1/experimental_results.md) · [初轮/逐小时结果](three_baseline_failures_20261023_v1/outputs/primary_evidence_v1/experimental_results.md) · [原始协议](three_baseline_failures_20261023_v1/PROTOCOL.md)

## 历史C71的执行闭合

三颗seed原预算训练已经完成。保留原评估guard失败后，以未改动的评估器在完整父进程环境重新启动。当前观察状态：`RUNNING`；实际进程/终态见[观察回执](three_baseline_failures_20261023_v1/outputs/c71_observed_state_v1.json)。其完整评估未完成前不据部分数值宣布成败；不因一次观察超时重启。该历史支线不增加第四篇主baseline。

## 数据和发布

只将紧凑代码、config、训练曲线、指标、错误回执和报告上传GitHub。数据、权重、预测数组、resume与完整日志留在服务器，未同步本地。原有归档和全部旧实验保留。

更早阶段的结论仍见[研究索引](../../docs/RESEARCH_ATTEMPTS_INDEX.md)，旧进度以各阶段终态和本次观察时间为准。
