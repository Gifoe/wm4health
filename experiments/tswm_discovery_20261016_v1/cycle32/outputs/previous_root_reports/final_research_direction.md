<!-- cycle31-current-start -->
# 最新 cycle31：按用户纠正研究范围

UTC 2026-10-16T19:51:23.870858+00:00。通用时间序列 world-model 框架，以 WorldTS 为研究层次参照；不强制零样本共享 checkpoint、因果/物理解释或某种结构。NormWear-2 原生预测执行、三处条件接口诊断和普通修复完成，独立复核通过；不是新方法或论文性能复现。接受方向0，整体 ACTIVE_UNACHIEVED。

见[范围纠正](cycle31/SCOPE_CORRECTION_20261017.md)、[实际证据](cycle31/experimental_results.md)与[当前研究判断](cycle31/final_research_direction.md)。以下各轮历史保留，其“当前”仅指当时。
<!-- cycle31-current-end -->

<!-- cycle29-index-start -->
# 最新证据cycle29

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/final_research_direction.md](cycle29/final_research_direction.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/final_research_direction.md](cycle28/final_research_direction.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

# 当前研究结论：没有通过全部关口的研究方向

**NO_VALIDATED_NOVEL_DIRECTION. 当前原型淘汰；整体自主发现目标仍未完成。**

## Research Question

本轮深入检查的问题是：冻结连续动态模型能否在明确coupling假设下，同时控制有限样本模型误差与结构歧义，并提供仍有用的个体轨迹反事实区间？它是一个具体问题清单项，当前不能作为“已确认尚未解决的新Gap”推荐。

## Evidence

- Native WorldTS未来条件支路支持只有5/9步；但普通FIR、GRU、ARX已修复受控测试，动作记忆方法路线淘汰。
- 在学得的非线性Gaussian kernel上，保持kernel不变仍可改变36.5%–57.6% held-out效应符号；这是CoMP等已知不可识别性的复核。
- 7→256个angle仍欠覆盖；全局最优OLS控制也表明拟合kernel的结构range不等于统计置信区间。
- Joint confidence tube恢复限定家族下的覆盖；shared-parameter prototype在独立确认中sign判定率4.4%→13.7%，但width仅减少3.28%，未通过预设15%gate。
- 同一集合内的可行优化把任何sound区间的最大平均宽度改善限制在约16%，说明大部分宽度由集合本身强制要求。
- 随后实际执行通用CROWN：同一共享L2系数球、同一全局连续角度集合和factual-abduction计算图，无自定义relaxation。它已构成强竞争者，短/中时域区间比当前专用prototype窄；长时域完整原始结果见 `experimental_results.md` 的E8。

## Existing Methods

CoMP已给Gaussian rotation反例；CSM/APID已做连续静态部分识别；Haugh/Singal和Lally已做动态/离散bounds，2026-10-14的Lally工作已联合统计与因果不确定性。Flow*/CROWN和non-rectangular robust inference已有参数依赖、通用计算图外界和共享机制处理。原始链接、代码pin与边界见 `literature_review.md`。

## Proposed Idea

**没有一个新机制被本轮证据批准推荐。** 已试的一阶共享参数sensitivity加有界remainder是已知Taylor/reachability思想的特例；它没有通过自身gate，也没有证明超越通用CROWN。不能给它重新命名后宣称创新。

## Falsification

本轮执行了强反证，而不只列未来检验：普通动作记忆baseline解决问题；已知文献覆盖宽泛新颖性；更密angle网格没有消除拟合误差；独立方法gate失败；通用验证工具有竞争力。这些结果足以停止当前原型的新方法路线。它们并不证明所有连续非线性world model问题均已解决。

## Feasibility

在RTX5090服务器实际完成15个新小型神经训练job、4个旧native checkpoint审计、两组各192个独立OLS fits、另8个可行对手fits，以及CPU通用验证对照。关键数据/数学不变量测试已通过。全部代码、plan、日志、原始指标、权重和witness保留于独立目录；没有公开上传，没有修改旧实验。

## Next Experiment

调整研究范围，而不继续微调本线性prototype。优先审查尚未验证的非线性、异方差/部分观测机制中，现有world model的可检验失败是否在普通完整状态估计/递归模型及成熟推断工具下仍存在。必须先给出新的竞争假设和独立数据计划；不能把本轮已淘汰的原理直接当作下一轮新Idea。检索与证据不足时继续淘汰，不强行提交论文方向。

## 推荐

当前**零个**方向满足“明确新颖、问题重要、已有方法不足、最小实验证明可改进”的完整标准。可保留上述区间问题作检索线索，但不把它冒充最终推荐选题。下一轮应继续自主发现，整体目标保持未完成。
