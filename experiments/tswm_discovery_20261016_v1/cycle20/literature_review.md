# 文献与创新边界

本轮从“不可控但与奖励相关的特征可能被丢弃”缩小到：**当前已观测的随机驱动会改变下一步物理状态，紧凑的自预测表示是否仍会丢掉它？** 本轮已运行受控实验；新颖性尚未成立。

[2606.30068v1](https://arxiv.org/html/2606.30068v1) 的特征决定奖励，提供了直接的近邻问题。把 uncontrollable-but-relevant 重新命名不构成创新。[CDL,ICML2022](https://proceedings.mlr.press/v162/wang22ae.html) 更早已区分可控变量、不可控但影响动作结果的变量和无关变量，因此“保留不可控的因果父变量”也不是新概念。

[Sensorimotor World Models](https://arxiv.org/html/2606.20104v1) 的联合正逆预测、[SLIM](https://arxiv.org/html/2610.03137v1) 的逆动作辅助、[Hamiltonian JEPA](https://arxiv.org/html/2609.33497v1) 的双空间/结构化输入端口均是必要边界。[MoP-JEPA](https://arxiv.org/html/2607.05238v1) 采用EMA、单位球、路由和硬分配，不能把本轮简化的 joint-target two-head arm 当作其原生实现。[Var-JEPA](https://arxiv.org/html/2603.20111v1) 已给出变分生成视角；本轮对角Gaussian控制不是该论文复现。[CausalDreamer](https://arxiv.org/html/2610.12016v1) 已做奖励/可控性分组，不能再把一般因子化包装为创新。

更精确的待核查方向：在非因子化观测中，从动作响应学得输出目标，再用它要求输入状态保留必要因果信息。动作响应输出空间与完整输入状态不必相同。[主动子空间](https://arxiv.org/abs/1304.2070) 和 [导数辅助输入/输出降维](https://www.jmlr.org/beta/papers/v27/25-1861.html) 已是成熟机制。现已实际运行敏感性降维：全部输出目标失败，而同一压缩器使用动作响应输出目标成功。动作输出选择是两个成功方法共享的待核查成分；不能把旧压缩器改名为创新，也不能据此声称旧论文已经提供了当前自动目标选择。

完整事实卡和范围区分见 LITERATURE_BOUNDARY.md。八份固定版本原文/PDF、提取文本和SHA256登记在 outputs/literature。原文阅读与源码执行分开：cycle20 是数学目标的受控任务迁移，没有复现上述论文的原生视觉排行榜。
