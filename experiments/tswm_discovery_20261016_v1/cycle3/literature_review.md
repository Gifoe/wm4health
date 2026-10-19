# 第三轮文献与创新边界

检索截至2026-10-16；优先最近六个月。这里只列对决策有作用的来源，阅读深度明确区分，不把source检索计作训练。

| Primary source | 已核对部分 | 对当前判断的作用 |
|---|---|---|
| [EpiCF-Bench, KDD2026](https://arxiv.org/html/2606.05692v1) | 任务/变量/数据生成/结果/公开全套轻量基线 | 原文说明模拟state为covariate；新执行进一步确认目标场景状态整段进入测试。干预序列文字部分不一致，实际源码为逐日binary flip，不自行改成constant干预 |
| [数据v1.2](https://zenodo.org/records/20762823) | record API、字典、三个文件校验 | 相比README链接v1.1，v1.2只改字典说明，signal MD5相同；公开许可证CC-BY4.0 |
| [PRO-DYN, ICML2026](https://arxiv.org/html/2507.15774v2) | 定义、模型分类、Linear与time-delay关系、方法边界、真实仓库 | learnable point-map不保证state identification，作者亦承认；不能自动把不同定义作为新缺口 |
| [Deployable Causal Action Geometry](https://arxiv.org/html/2610.16644v1) | problem setup、anchored response target、limitations、公开数据实验描述 | 时间漂移下action ranking/response geometry已有新近方法；它明确限于one-step和persistence/overlap假设。未发现可运行官方代码，未复现其数值 |
| [Feature Selection in World Models, author analysis](https://huguesva.github.io/blog/2026/world-models/) | 线性Gaussian假设、objective排名及PushT stress test描述 | JEPA保留稳定nuisance、predictability不等于task relevance已经被作者实际实验；blog不冒充同行评审，未复现其结果 |
| [Controlled World Model Identifiability](https://arxiv.org/html/2607.22430v1) | excitation/identification条件及counterfactual均值定义 | 不能把action excitation不足或conditional mean与unit CF差异重命名 |
| [Gong et al. UAI2017](https://proceedings.mlr.press/r15/gong17a.html) | primary abstract/识别假设 | 时间聚合因果辨识已有基础，非Gaussian noise和known aggregation factor是关键条件；2026重发不是新论文 |

继续核查（尚未作完整技术评价/数值复现）：[Semigroup-JEPA](https://arxiv.org/html/2609.10464v1)、[One-Step Next-Latent Prediction Is Not a World Model](https://arxiv.org/abs/2609.36227)、[A Control Theory of Predictability in Latent World Models](https://arxiv.org/abs/2607.10362)。这些已有与均值闭合、latent memory、on/off-manifold planning有关的近邻，不能忽略。

额外screen：Dynamic OT的跨环境map monotonicity需要矩阵反例核对；其固定prior下Brenier唯一性与prior不可识别性必须区分。SOBI已有利用各源不同autocovariance辨识Gaussian混合的经典结果。当前二者都没有被立为新方向。


## 新近直接竞争者

[MotionJEPA, 2026-09-20](https://arxiv.org/html/2609.23881v1)已经对slow-feature suppression、time/flattened SIGReg、stop-gradient和inverse dynamics进行数值对比；[TC-LeWM, 2026-07-29](https://arxiv.org/html/2607.26924v1)已通过temporally centered residual避免跨任务cluster收缩。它们的已有机制不能重命名。MotionJEPA是视频/control，不等同于所有time-series任务；TC-LeWM的主要下游是冻结encoder的behavior cloning，不冒充latent-space planning证据。

[ChronoJEPA作者仓库](https://github.com/MrRobotop/ChronoJEPA)已有time-axis collapse、position/order和下游收益的分离实验。只核查公开说明，未执行其数值；无证据将其当作顶会同行评审。它仍构成必须考虑的直接公开近邻。

[LeNEPA官方实现](https://github.com/langotime/lenepa-milets-2026)已固定并执行原生组件反证；其默认input-layer target和位置配置使简单的任意位置表反例不适用。不是只读标题就声称解决或失败。
