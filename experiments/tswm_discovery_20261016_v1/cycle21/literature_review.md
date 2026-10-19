# World-model scope and novelty boundary

目标是通用时间序列世界状态与条件演化框架，数据集是证据载体。最新范围与具体源码依据见 RESEARCH_SCOPE_UPDATE.md 和 LITERATURE_INTERFACE_BOUNDARY.md。

WorldTS已分别编码目标序列与条件，不能把“分开条件和预测目标”本身当成创新。CDL与Denoised MDPs明确保留影响动作结果的不可控因子；主动子空间、导数辅助输入/输出降维、非线性平衡降阶是成熟机制。H-JEPA的双空间、MoP-JEPA的EMA归一化/路由与Var-JEPA随机建模也已存在。本轮不宣称这些原生模型被复现或被推翻。

候选剩余问题仅为：在没有手工目标/协变量角色划分的观测中，能否从条件响应学到合适的预测目标，并形成紧凑且充分的输入状态？当前尚无已核实的新颖性结论。新文献原文、固定版本、代码字节和哈希见 outputs/literature。JMLR导数降维及CoBRAS/平衡降阶的筛查不等于完整复现。
