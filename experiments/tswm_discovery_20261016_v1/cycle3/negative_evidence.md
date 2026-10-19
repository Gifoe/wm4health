# 第三轮负面证据

1. 上一轮RPM校准路线已被普通生成式SSM和既有RPM似然训练修复，继续关闭，不追加调参。
2. EpiCF输入信息问题是发布协议的具体证据，不等于发现新模型机制。普通causal g-computation、CRN/CT/TE-CDE和历史条件模拟已经存在；禁止把删除oracle covariates重新命名为创新。
3. “目标病例可由目标S差分精确重建”未通过。保存全部数值和时间错位/跨县负对照；不把近似源码恒等式当作数据事实。具体聚合/导出流程尚未在公开材料中完整恢复。
4. 5个完全重复县轨迹降低了独立样本数；没有证据表明它单独解释神经模型现象，不能夸大为论文结果全部失效。
5. PRO-DYN明确定义DYN为历史到未来的learnable mapping，并承认Linear是relaxed time-delay dynamics。它没有承诺完整物理系统识别；不能用“并非真实动力学”作为稻草人反驳。
6. 新检索发现Semigroup-JEPA、One-Step Next-Latent Prediction Is Not a World Model和A Control Theory of Predictability in Latent World Models。只提出“多步一致性/均值复合不闭合/低预测误差不保证规划”已经有直接近邻。完整机制核查继续；不据摘要宣称其全部已解决。
7. 时间聚合导致因果频率恢复问题至少已有Gong等UAI2017的非高斯state-space辨识结果；不能把coarse logging ambiguity本身作首创。


## LeNEPA位置假设的源码/组件反证

固定官方源码6574eb9e123305334e09e180de67682844f2e61f；下载92个代码/配置/论文源文件883,823 bytes。默认nepa_target_layer=0、target_uses_pos_embed=false、objective覆盖num_registers=0，并在[0,8]两层施加时间正则。未修改的原生Block/SIGReg组件在三颗种子下检查：八层RoPE block对恒定时序输入的最大时间差仅约2.4e-6至2.9e-6。任意Gaussian位置表可骗过抽象time-axis statistic，但不能直接替代此实现的layer0目标；该简单构造不构成native失败证据。没有执行完整LeNEPA训练，不声称证明该方法保留所有动力学信息。详见`outputs/lenepa_position_null_audit.json`。
