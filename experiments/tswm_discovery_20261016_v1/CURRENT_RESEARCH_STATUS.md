# 当前研究状态 — 2026-10-23 V2

研究目标：ACTIVE / UNACHIEVED。已接受最终新颖方向：0。主线仍为WorldTS、LatentTSF、TSWM；TSWM为已说明范围的源码子集执行。

## 本次决定

六个官方decoded-loss DLinear控制完成，并独立复核。十八个已选权重的确定性完整Val测量完成，不重新选epoch。相对raw的Val均值，decoded在H336为+1.35%，H720为−1.03%，均进入预先规定的5%范围。按Val规则停止当前新架构动机；十八个iTransformer分支单元没有触发，实际训练数0。

**Test差距仍存在**：decoded相对raw在两个horizon分别高约7.21%和7.35%。这不是整个失败已被唯一解释的证据，也不能用于推翻Val规则、继续调权或扩模。

[最新完整报告](three_baseline_failures_20261023_v1/outputs/decoded_evidence_v1/experimental_results.md) · [机器决策](three_baseline_failures_20261023_v1/outputs/decoded_evidence_v1/verification.json) · [预先冻结规则](three_baseline_failures_20261023_v1/OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md)

## 已完成和已否定的证据

- 当前campaign累积30个新训练单元、24个冻结模型逐小时审计；本轮增加6个训练、12个逐小时审计，另测量18个冻结权重的Val。
- 固定100模型的逐小时审计现已闭合；12模型中10个改善，2个略差。原始权重的全改善结论没有替换新结果。
- 泛称“重构好但不适合动力学”已有直接相邻工作，不能作为当前新颖性依据。近期两篇仅作prior-art核查，没有增加主baseline或伪报复现。

[新颖性边界](three_baseline_failures_20261023_v1/NOVELTY_SCREEN_20261023_V2.md) · [上一轮固定预算结果](three_baseline_failures_20261023_v1/outputs/matched_evidence_v1/experimental_results.md) · [三篇失败台账](three_baseline_failures_20261023_v1/THREE_PAPER_FAILURE_LEDGER_V1.md)

## 既有C71评估

三颗原预算训练seed已完成。原评估guard失败已保留；当前未改动评估器的实际状态为`RUNNING`，观察时间2026-10-22T18:43:15.594973+00:00。以[进程/终态回执](three_baseline_failures_20261023_v1/outputs/c71_observed_state_v2.json)为准；完整评估仍未闭合时不依赖部分种子宣布成败。它不增加第四篇主baseline。

## 保存和后续

全部原实验与已发表归档保留。只发布紧凑源代码、配置、曲线、指标、错误记录和报告；数据、权重、预测、resume及长日志仍留服务器，没有同步本地。下一步先闭合既有评估；新方向需要一个未被普通控制或现有论文解释的独立能力失败，不能重命名当前损失修复。
