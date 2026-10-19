# 第二轮反证与停线路径

更新时间：2026-10-15T21:07:41.248013+00:00。这是第二轮工作记录；上一轮目录及结论未改写。研究总目标仍未完成。所有数值只从已保存结果读取。

1. 上一轮专用反事实区间原型已被未修改CROWN在相同不确定集合上超过；该新方法路线停止，不能借更换术语继续推进。
2. DoTime中反向响应“高方向分数”依赖单向干预。平衡正负剂量后消失，否定了不加条件的普遍gaming论点。普通双臂contrast/适当oracle已能修复解释，未发现必须新架构的需求。
3. 原作者早在2023年就给出辅助界的额外KL，2026最终版明确列为限制。不能把“exact surrogate不等于true posterior”本身宣称新颖。
4. 固定正site反例中连无信息观察都可能让surrogate收缩。这是刻意挑选的参数，不代表训练必然会选择它；所以追加了原生学习与独立告警对照。
5. 初始学习结果：在prefix=1告警且无anchor时，保留F的数值推断几乎没有修复低覆盖率。否定“主因只是冻结模型推断丢掉F”的强解释；表示/目标或site容量仍是竞争解释。
6. 1000步验证损失仍下降，不能据此排除训练不足。因此在查看测试结果前，固定两种条件、所有种子统一10000步。延长预算结果必须完整保留。
7. 广义Gaussian辅助界的小TV反例需要某些full recognition方差大于prior，超出当前native正site类。不能把它直接当作native模型内反例。另建了所有full recognition方差<=prior的独立反例，后者不声称小TV。
8. Gaussian VI在固定反例上已显著减轻误差；它是已有方案。普通生成式SSM也必须作强对照。若后者解决问题，“现有world models无法解决”的宽泛缺口立即淘汰。
9. 只有一套新受控数据生成种子，没有真实观测数据验证；不能声称跨数据集/领域泛化。训练初始化重复不等于独立数据重复。
10. 潜坐标、额外维度、非线性物理读出可能绕过固定坐标下的协方差上界。当前不提出所有world model的普遍不可能性定理。


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
