# 三篇主线的失败归因：本轮已闭合，研究方向仍未成立

当前结论：真实运行与失败归因已经形成证据链，但尚无足以立项的新颖世界模型问题。总体目标ACTIVE/UNACHIEVED，最终接受方向0。不能把已知loss控制、普通VAR、推断预算或表示指标平移敏感性重新命名为创新。

## 三篇与可比边界

| 论文 | 固定源码 | 已有真实运行 | 失败现象与目前判断 |
| --- | --- | --- | --- |
| WorldTS | 269426ea6217d7152a244da0e209d922acd46d4f | 原ledger四个NP/PJM单元，另有Loop、readout、Stage1、NESO归档 | Loop收益不稳定，gradient cosine与E1最优不足以证明状态丢失，普通低维校正已解释一部分updater收益。旧路线保持关闭。 |
| LatentTSF | 7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2 | 原十六单元及本campaign三十个新fit | Electricity/H336,H720配方差距真实存在；dense-origin并没有产生预想的普遍劣化。固定预算、官方decoded loss和冻结AE组合排除了多种草率归因。 |
| TSWM recipe | 2e5e48540ff0da48e8192163969a6c58df1d5a3e | C53四十个源码子集fit，C79八十次冻结模型/数据bank重放 | 没有发现预想的future-action leakage；准确率与方向一致性分离及方向监督已在原文，不能再作为新发现。 |

这是两项模型工作与一项benchmark/recipe，不能不加验证地统称三种领先SOTA模型。条件未来输入、历史单独输入、方向监督和原始数据资格各不相同，不制作虚假统一榜单。旧权重重新校验不是新训练；40个TSWM模型不是本轮重跑。详见[模型注册表](../../baseline_registry.md)和[历史台账](../../THREE_PAPER_FAILURE_LEDGER_V1.md)，历史台账中的待处理条目由本文件更新，不覆盖原文。

## LatentTSF的失败不支持什么

本campaign实际新训练30个：6 native latent、6 raw、12 fixed100和6官方decoded-loss。另有24个冻结模型逐小时审计。原18个条件iTransformer扩模分支按预注册Val规则未触发，实际fit数0。固定epoch/更新量不等于函数类、初始权重或FLOPs完全匹配。

官方decoded控制相对raw的Val均值在H336为+1.35%、H720为−1.03%，都进入预设5%范围；这要求停止当前新增架构动机。Test仍差约7.21%/7.35%，所以不是完整因果解释，也没有按Test推翻Val分支规则。冻结组合表明同一AE的历史重构Test/Val误差比约2.15/2.60，只重构历史即可使原raw预测器Test风险增约5.01%/3.35%。误差有符号交叉项保留，不能直接等同编码器信息丢失。

[decoded控制与冻结决定](../decoded_evidence_v1/experimental_results.md) · [冻结组合与归因边界](../transport_evidence_v1/INTERPRETATION_V1.md) · [直接先行工作](../../NOVELTY_SCREEN_20261023_V2.md)。通用“重构好但动力学不好”已有直接先行工作；本轮没有验证一个超越普通控制的新机制。

## 历史C71现已完成

三颗原预算训练seed的12个神经评估臂及3个经典控制全部完成；reserved EVAL未打开。H50平均MSE：100步native搜索0.004623231，1000步0.000600306，真实图特权诊断0.000312806，history-VAR2 0.000232233。VAR2低于每颗神经seed的平均误差，也低于每颗特权诊断；不是声称每条轨迹都优于它们。更大搜索预算有显著作用，真实图仍留下学习库/自由rollout残差。这个充分观测子集有强普通控制，不能据此证明需要新架构。

全部15个预测数组、训练/校准/选模链和scalar指标独立复核；两种VAR在全部256条开发历史上用不同数值解法重放，24组搜索接受trace前缀一致。旧RUNNING文件仅反映其当时观察时间，现在终态为完成。它是已有source-based子集，不是第四篇主SOTA，也不是完整论文或未见结构组合验证。

[最终C71报告](../../../cycle71/outputs/free_simulation_closure_20261023_v1/experimental_results.md) · [独立复核](../../../cycle71/outputs/free_simulation_closure_20261023_v1/VERIFICATION71.json)。

## 一个已确认但不新颖的指标问题

两份原始native DLinear/H96 checkpoint上，保持物理预测完全相同的潜态原点平移，可把预测潜态TTC从ETTh1的0.994526升至0.999986、Electricity的0.979415升至0.999900；最高偏移下打乱时间TTC仍约0.999785/0.999520。所有偏移由TRAIN固定，指标在Validation计算，没有选择有利的Test分数。

这只证明TTC绝对值不能独立认证时间行为。公式来自论文，但epsilon和采样由本协议明确指定，未复现论文geometry表；不指控论文算错，不涉及CKA/有效秩。平移敏感性与中心化属于已知数学/表示评测议题，不能作为新world-model方法。

[完整协议与结果](../../metric_validity_v1/outputs/experimental_results.md) · [1160个独立scalar核验](../../metric_validity_v1/outputs/VERIFICATION_V1.json)。

## 当前Research Question与下一道门槛

目前能回答的是：这些失败是否已经排除了普通配方、推断、数据和评测解释，足以要求一个新世界模型机制？本轮答案是否。我们保留未唯一归因的Test残差，却不据此强立研究方向。

- Evidence：原源码重放、三seed预算/目标控制、冻结组合及自由模拟完整评估；代码、config、SHA、报错和终态全部可定位。
- Existing Methods：官方decoded objective、普通历史辨识和更充分native推断已经推翻多种新架构动机；方向一致性和表示不变性也各有先行研究。
- Proposed Idea：无已接受的新方法。
- Falsification：若普通控制闭合失败、输入资格不符、失败仅是指标/执行问题、已有方法已解决，淘汰该解释。
- Feasibility：单RTX5090、已有三篇固定源码与原始checkpoint；不靠继续扩预算挽救这个案例。
- Next Experiment：本轮不再追加这个数据集的forecast训练或指标偏移。后续候选先必须从已有台账定位一个独立的可观测能力残差，并确认未被普通控制或直接先行工作覆盖；尚无合格残差时继续收敛检索，不谎报已确定方向。

本轮对三个主paper的聚焦执行承诺和C71旧评估已闭合；这不等于完整三篇论文复现，更不等于最初研究发现目标完成。仅向GitHub发布紧凑代码与证据，数据、权重、状态库、预测和长日志留服务器G数据盘。
