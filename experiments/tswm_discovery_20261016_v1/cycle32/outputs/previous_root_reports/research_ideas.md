<!-- cycle31-current-start -->
# 最新 cycle31：按用户纠正研究范围

UTC 2026-10-16T19:51:23.870858+00:00。通用时间序列 world-model 框架，以 WorldTS 为研究层次参照；不强制零样本共享 checkpoint、因果/物理解释或某种结构。NormWear-2 原生预测执行、三处条件接口诊断和普通修复完成，独立复核通过；不是新方法或论文性能复现。接受方向0，整体 ACTIVE_UNACHIEVED。

见[范围纠正](cycle31/SCOPE_CORRECTION_20261017.md)、[实际证据](cycle31/experimental_results.md)与[当前研究判断](cycle31/final_research_direction.md)。以下各轮历史保留，其“当前”仅指当时。
<!-- cycle31-current-end -->

<!-- cycle29-index-start -->
# 最新证据cycle29

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/research_ideas.md](cycle29/research_ideas.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/research_ideas.md](cycle28/research_ideas.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

# 候选问题与淘汰

下面记录本轮曾保留、随后在E8进一步否定当前方法创新价值的问题。没有一个候选已经通过完整创新性筛选；不凑三个方向。

## 优先继续检查：冻结连续动态模型的可靠而有用的轨迹反事实区间

Research Question：在只用unpaired transition data、完整factual trajectory供retrospective abduction、以及明确声明的coupling敏感性类时，能否同时保留跨时共享机制和跨分支噪声约束，得到可靠且有非平凡效应判定能力的连续轨迹区间？

为何重要：同一准确conditional kernel支持可靠prospective planning，不代表单条已发生轨迹的反事实唯一。把任意seed耦合下的点估计写作确定个体结论，可能得出在compatible模型下反向的判断。纯结构range又会遗漏有限样本误差；简单补上统计置信集可能让区间几乎没有决策信息。本轮区分了这三件事。

最小机制：对冻结model保持kernel不变的coupling family，联合传播abduction与prediction对共享参数的query sensitivity；保留有界高阶remainder，按真实数据可支持的范围输出interval或abstain。first-order Taylor prototype只是已知工具的基线，不能作为独创名称。

竞争者：CoMP、CSM/APID、Haugh/Singal、Lally IMDP、Flow*/CORA、non-rectangular robust inference。潜在贡献必须发生在**连续非线性、冻结模型、abduction-conditioned query和严格覆盖/效用/成本权衡**，而不是首创不确定性或参数共享。

资源：现有脚本和全部数据生成器已在RTX5090服务器运行；核心线性确认可在CPU数十秒完成。下一轮可用小型frozen Gaussian SSM与公开DoTime连续模拟结构；无需先训练大型foundation model。RP-GSSM需单独处理JAX运行环境和控制接口，不能直接当成已支持action的baseline。

当前价值判断：**问题证据成立；创新性与通用可解决性仍不充分；尚不推荐写论文或大规模训练。** 当前prototype宽度gate失败。该题比继续Loop微调更有信息价值，但没有证据支持它已优于多数现有解决方案。

淘汰方向：条件动作记忆增强（普通基线已解决）；泛latent/readout alignment（历史NO-GO）；仅更换反事实shared seed（已知且不可识别）；首次联合两类不确定性（已有直接论文）。


补充反证（E7）：通用可行优化已迫使同一置信集合的区间宽度至少为0.93463，对应basic outer为1.11427。因此该线性特例中任何sound方法的最大平均宽度收益被限制在16.12%以内；这使15%筛选门槛接近可达上限。不能把不可省的不确定性宽度归咎于缺少新架构。该候选仍不足以作为已确认的顶会方向。


E8最终筛选：淘汰当前证书prototype的新方法路线。没有一个候选已经通过“新颖性+强基线+独立实验证据”的完整关口。下一轮应重新选取有残余缺口的机制，不继续在本线性系统上微调工具设置。
