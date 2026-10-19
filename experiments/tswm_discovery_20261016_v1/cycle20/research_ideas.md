# 一条待发展的候选；最终批准方向仍为零

Parent: reward-only exogenous-feature failure + existing causal state abstraction. Operation: refine to physically coupled incoming innovations, and instrument input information separately from action recovery and target predictability.

**候选问题：** 能否在无需奖励、无需手工指定物理状态通道的条件下，学得紧凑且对动作条件演化充分的世界状态，同时保留不可控但影响结果的当前驱动？

**本轮支持：** 三类系统中接近完美的逆动作预测仍伴随随机gain丢失和响应幅度错误；强探针未恢复。更大状态修复，说明有明确可解决性，也给方法提出强容量/预算对照。

**拟检验机制：** 在完整观测teacher中用支持内动作查询估计响应输出空间；把该空间作为预测目标，而不是把“输入状态必须记住什么”和“未来目标必须可预测什么”强行设成同一对象。已有因果抽象、控制子空间、导数降维可能实现同样原则；当前不把这段思路称为已证实创新。

**具体救援与反证：** cycle21同时比较完整future-bank蒸馏、单纯action-contrast、投影future-bank、同rank随机投影、额外预算重建/预测、更大状态和规范化EMA路由混合。成熟active-subspace已在同pilot上配合动作响应目标成功，故当前不提出新的神经压缩器。它与候选共享目标选择成分，后续应检验目标选择是否确有独立价值、是否超出已知因果降维，以及跨系统和非线性观测能否成立。若只在线性观测toy成立，不能提交一般world-model框架结论。

它是当前最值得继续验证的候选，不是最终已推荐选题。真实/native任务、长期闭环模拟与精确新颖性仍缺失。
