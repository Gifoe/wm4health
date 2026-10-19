# C34 文献与创新边界

本轮三个独立视角围绕数值时间序列 world-model 的实际任务、失败归因和可构造的有限预算方法检索。范围以 WorldTS 为框架层次参照，允许按数据集训练，不附加强制因果、动作、物理或零样本门槛。完整来源、版本、原文位置与限制在三份 framework_*.md 中。

可执行性与新颖性是两件事。[Sundial](https://github.com/thuml/Sundial) 的公开生成器可以提供固定历史下从720维高斯源到未来路径的映射，本轮已执行并独立重放；[PathFM](https://arxiv.org/abs/2610.22358) 的作者材料目前称将发布代码与权重，未确认可运行发布，不能声称复现。Sundial是早于主要六个月检索窗口的已发布底座。

稀有事件 proposal 线索的更近先例包括 [Exogenous Matching](https://arxiv.org/html/2410.13914v2) 的输出条件 source proposals、[MadNIS](https://arxiv.org/html/2212.06172v2) 的已知 channel 神经密度学习、[Murthy/Juneja/Blanchet](https://arxiv.org/pdf/1206.3390) 的首次穿越时间区块 IS。把 first-hit bin 当观察标签时，候选目标恰好是 categorical CE 加逐 bin forward KL；不是只因组件古老而否决，而是未找到目标函数或有限预算机制的实质区别。没有声称某篇先例逐字实现完整 Sundial 组合。

[Pythia](https://arxiv.org/html/2610.05240v1) 的数值分支与上下文校正提供另一个可测问题：上下文是否补充新信息，还是补偿共同训练削弱的数值分支？原表格采用不同指标聚合，不能仅凭 aggregate reversal 确认机制。REST-TS、普通输出残差、额外数值损失和预测组合是必要对照。此线索尚未运行、未接受为新框架。
