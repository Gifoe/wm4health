# 第二轮竞争假设

更新时间：2026-10-15T21:07:41.248013+00:00。这是第二轮工作记录；上一轮目录及结论未改写。研究总目标仍未完成。所有数值只从已保存结果读取。

|解释|可证伪实验|目前判断|
|---|---|---|
|H1：普通Gaussian家族本身无法表达rare posterior|固定严格log-concave单峰posterior，对比最优Gaussian VI与最优Gaussian辅助界|V=3稀有极限下普通VI的KL趋零而辅助界gap不消失，H1不足以解释固定构造|
|H2：全局混合F近似prior即可保证所有条件的推断准确|F=(1-eps)N1+eps*NV，eps下降，完整计算条件误差|总变差<=eps而rare条件辅助gap持续存在；只反驳无尾部/稀有度条件的uniform保证，不反驳作者的全部定理|
|H3：冻结训练模型的主要失败来自忽略F|同一checkpoint/坐标/训练reference，原生对比数值ratio posterior|prefix1的低coverage几乎未修复；初始证据不支持主要归因于该项|
|H4：学习不足|所有组统一10倍更新预算，只按验证目标选择|长训练已启动/按实际文件更新；不能预判结果|
|H5：观察的正site参数化无法表达条件方差扩张|固定正确prior的原生组件、原生类内反例；再检查训练读出与oracle条件方差|数学容量约束已确认；物理坐标/额外维度/非线性读出的限定必须保留|
|H6：常规方法已经足够|不提供latent标签的普通生成式SSM，未知告警MLP，匹配更新预算|强对照正在执行；若修复则淘汰宽泛“新方法必需”命题|
|H7：稀有条件其实不重要/统计不稳|独立数据重复，低风险/高风险决策成本、适当tail proper score及CI|尚未做完，不能用单个覆盖率替代任务价值|

下一轮判据：先读取全部10000步和普通SSM结果，做按样本配对及初始化重抽样区间，再决定是否值得增加数据种子/维度。禁止看到哪一项测试改善就改primary。当前primary预先定义为prefix16,horizon1,alarm且无anchor；prefix1是预先登记的诊断条件，不能换名为原primary。


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
