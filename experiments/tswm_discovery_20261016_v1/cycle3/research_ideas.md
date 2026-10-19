# 候选研究问题：尚无接受项

本轮保留一项有证据的评测材料：EpiCF发布条件预测分数不能单独证明行动模拟能力。它可作为未来完整研究的sanity control，但不是已经通过创新核查的方法Idea。

不推荐把以下主题直接立项：删除未来状态、增加行动输入、对DMS加循环、只加semigroup consistency、JEPA加入普通重建、一般性“预测不等于规划”。这些都需要与明确已有工作进一步区分，且多项中心命题已有直接文献。

下一项检索优先级：读完整Semigroup-JEPA及最新one-step诊断的模型/信息假设，检查存在的概率、多尺度、部分观测处理；再决定是否有一个可区分的高信息量实验。当前没有承诺使用Loop Transformer或大型foundation model。


## LeNEPA位置假设的源码/组件反证

固定官方源码6574eb9e123305334e09e180de67682844f2e61f；下载92个代码/配置/论文源文件883,823 bytes。默认nepa_target_layer=0、target_uses_pos_embed=false、objective覆盖num_registers=0，并在[0,8]两层施加时间正则。未修改的原生Block/SIGReg组件在三颗种子下检查：八层RoPE block对恒定时序输入的最大时间差仅约2.4e-6至2.9e-6。任意Gaussian位置表可骗过抽象time-axis statistic，但不能直接替代此实现的layer0目标；该简单构造不构成native失败证据。没有执行完整LeNEPA训练，不声称证明该方法保留所有动力学信息。详见`outputs/lenepa_position_null_audit.json`。
