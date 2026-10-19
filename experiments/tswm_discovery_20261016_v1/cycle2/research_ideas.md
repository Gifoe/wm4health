# 候选问题与可行性

更新时间：2026-10-15T21:07:41.248013+00:00。这是第二轮工作记录；上一轮目录及结论未改写。研究总目标仍未完成。所有数值只从已保存结果读取。

仅保留一个需要进一步核验的候选，避免凑足三个方向。

**条件性不确定性与概率latent消息的收缩限制。** 一个传感事件可能提示状态更极端、更不确定；有效Gaussian消息的正精度却只会压缩固定prior。研究问题是：平均marginal匹配与较好均值预测，是否掩盖了罕见但可观察条件下的系统性风险低估？如何在不依赖高维观测重建的条件下保留这类信息？

已有证据包括固定概率模型的可计算反例、Gaussian家族/辅助界分离、原生组件与新任务训练初步结果。尚缺充分证据包括：长训练归因、额外潜维度与合适观测模型对照、独立数据、真实任务价值，以及与最新RP-GPDS等近邻的创新边界。

最小修复候选是允许相对prior的负精度证据并保留proper likelihood/尾部可积性。但这与DKF/EP/Gaussian VI直接相邻，**目前不能作为新机制或新方法命名**。只有发现标准方法留下的具体效率、可辨识性或校准矛盾，并通过对照排除，才进入方法设计。

计算可行：一维受控问题的原生训练在CPU执行，普通数值SSM训练用单张5090；当前无需大checkpoint或新大数据集。是否能扩展到高维decoder-free动态，是待验证问题。


## 最新强对照结论

普通生成式SSM的三颗初始化×两种告警条件已全部完成。原先登记的prefix16/horizon1/告警且无anchor条件下，其相对10000步native模型的Gaussian NLL差为-0.15202，95%重抽样区间[-0.22053,-0.09123]；尾部Brier差为-0.02260，95%重抽样区间[-0.03577,-0.01044]。测试病例配对重抽样；分别训练的方法独立重抽样初始化，同一冻结checkpoint的修正才配对初始化。这不是独立数据重复或多重比较校正。

因此否定“现有world models普遍不能处理此现象”的宽泛缺口。冻结ratio校正只产生小幅改善，不能据此排除训练期间的变分偏置。下一项强对照是已有RPM模型的数值likelihood重新训练，分别保留正site类与允许full Gaussian recognition；不是新方法。


## 已有decoder-free RPM对照已闭合

数值likelihood重新训练的8个cell均已完成：state-dependent条件中正site/full Gaussian各3颗种子，independent告警中各seed41。它们沿用已有RPM概率模型/目标，训练不使用潜状态真值，也没有观测decoder。原native positive-site类被重新数值训练后，主条件coverage也已恢复；full Gaussian版的尾部概率/适当评分接近普通生成式SSM。**正site proper RPM概率模型的表达能力，不能与native Kalman surrogate后验容量混为一谈。**

|条件|方法|已完成初始化种子|MSE|Gaussian NLL|95%覆盖率|预测尾部概率|真实/噪声条件期望尾部概率|
|---|---|---:|---:|---:|---:|---:|---:|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|positive_site_exact_ML|3|1.83206|1.72458|0.92738|0.12232|0.15357|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|full_gaussian_exact_ML|3|1.84091|1.72439|0.95833|0.15557|0.15357|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|positive_site_exact_ML|1|1.00182|1.42126|0.93645|0.03745|0.06020|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|full_gaussian_exact_ML|1|1.01443|1.42612|0.93645|0.04659|0.06020|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|positive_site_exact_ML|3|0.57158|1.00566|0.95884|0.12206|0.14636|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|full_gaussian_exact_ML|3|0.53145|0.96975|0.96316|0.14934|0.14636|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|positive_site_exact_ML|3|0.79048|1.28253|0.95953|0.09021|0.09033|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|full_gaussian_exact_ML|3|0.77204|1.26782|0.95156|0.09157|0.09033|

因此，本受控诊断不再支持“需要新decoder-free方法才能修复”的命题。冻结后验校正幅度小而数值likelihood重新训练改善大，支持训练过程中近似/目标偏置这一解释；但PyTorch重实现与Flax native初始化等差异仍在，不能把全部改变量严格归于单一代码行。通用Gaussian VI/数值RPM等已有路线应被承认。

这是一项有证据的机制审计，不是已成立的新颖研究方向。暂不为其追加新架构、更多同类调参或大规模训练。所有单次告警、负对照、长训练和不利结果完整保留。
