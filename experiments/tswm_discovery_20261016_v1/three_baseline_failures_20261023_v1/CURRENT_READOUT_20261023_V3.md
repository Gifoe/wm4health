# 当前研究状态 — 2026-10-23 V3

研究目标：ACTIVE / UNACHIEVED；接受最终新颖方向0。主线仍是WorldTS、LatentTSF、TSWM。

## 冻结组合归因已闭合

本轮0个optimizer update，18个既有已选预测器、同一AE，完成12个split×horizon×seed审计单元，每单元6个冻结组合。324个独立数值/误差恒等式检查通过。原有30个新训练和24个dense模型审计计数保持不变；不能把组合数加为新训练模型数。

只重构历史输入，会使原raw预测器的Test MSE在H336/H720分别增加约5.01%/3.35%；同一AE的历史重构Test/Val误差比约2.15/2.60。这说明普通AE往返和跨时期误差已经参与残差，尚未唯一归因到编码器信息丢失或潜态转移。输出重构的误差不是任何潜态模型的下界；有符号交叉项必须保留。

[最新归因判断](outputs/transport_evidence_v1/INTERPRETATION_V1.md) · [六路径完整结果](outputs/transport_evidence_v1/experimental_results.md) · [独立数值复核](outputs/transport_evidence_v1/verification.json)

## 不恢复已停止的方法路线

原Val规则继续有效：官方已有decoded loss已进入raw的预设Val范围，十八组扩模分支没有触发。Test残差仍在，没有按Test重选、调参、校准或挑选组合。宽泛的重构/动力学不兼容已有直接先行工作，当前不据此立项。

既有C79已经包含lagged impulse检查；原文已有causal decay/conv/attention选项，不重复把点态Gate的零滞后支持当作新缺口。

## 历史C71

同一未修改评估器的当前状态：`RUNNING`，观察时间2026-10-22T19:07:48.632788+00:00；以[实际进程/终态回执](outputs/c71_observed_state_v3.json)为准。全部训练早已完成，完整评估未终态前不据部分种子宣布成败。它不是第四篇主baseline。

## 保存与下一步

全部旧结果保留；只发布紧凑源代码、config、曲线、指标、错误记录和报告。原始数据、权重、预测和长日志留服务器，未同步本地。停止继续追逐本Electricity案例的分数；闭合C71既有评估后，下一项研究问题需来自独立的、未被原有控制或直接文献解释的能力失败。
