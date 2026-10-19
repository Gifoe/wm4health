# 第二轮实验事实

更新时间：2026-10-15T21:07:41.248013+00:00。这是第二轮工作记录；上一轮目录及结论未改写。研究总目标仍未完成。所有数值只从已保存结果读取。

已保存：1000步原生训练 6/6，评测 6/6；10000步训练 6/6，评测 6/6；普通生成式SSM训练 6/6，评测 6/6。

## 任务与比较边界

这些是新的受控机制实验，不是原论文排行榜复现。状态为已知正确的 AR(1)，rho=.95，平稳方差1；训练4096条、验证512条、测试4096条，各16步。20%的时刻有带噪线性状态测量，其余测量缺失；另有依赖状态平方的稀有二元告警。独立告警对照与状态无关。三颗种子仅为训练初始化重复，使用相同的数据集，不能当作三套独立数据。

RP-GSSM使用原生MLP、目标、训练器和后验组件；正确prior以零学习率固定。潜状态真值未进入训练目标或checkpoint选择。训练集前1024条只在模型冻结后拟合线性物理状态读出。所有模型同一划分。

## 初始1000步结果（均值）

|条件|方法|已完成初始化种子|MSE|Gaussian NLL|95%覆盖率|预测尾部概率|真实/噪声条件期望尾部概率|
|---|---|---:|---:|---:|---:|---:|---:|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|native|3|1.84582|1.86996|0.82976|0.03948|0.15357|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|exact_same_readout|3|1.84712|1.86148|0.83214|0.04196|0.15357|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|known_observer_oracle|3|1.83494|1.72391|0.95714|0.16044|0.15357|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|native|3|1.01895|1.43178|0.92642|0.03611|0.06020|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|exact_same_readout|3|1.02076|1.43215|0.92642|0.03633|0.06020|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|known_observer_oracle|3|1.01076|1.42432|0.93645|0.04337|0.06020|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|native|3|0.64797|1.13922|0.91901|0.07523|0.14636|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|exact_same_readout|3|0.63125|1.11852|0.92752|0.08470|0.14636|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|known_observer_oracle|3|0.52566|0.96780|0.96546|0.15911|0.14636|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|native|3|0.82923|1.32084|0.93245|0.05621|0.09033|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|exact_same_readout|3|0.82137|1.31445|0.93518|0.06160|0.09033|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|known_observer_oracle|3|0.76867|1.26679|0.95414|0.09660|0.09033|

## 统一10000步预算对照（完成多少写多少）

|条件|方法|已完成初始化种子|MSE|Gaussian NLL|95%覆盖率|预测尾部概率|真实/噪声条件期望尾部概率|
|---|---|---:|---:|---:|---:|---:|---:|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|native|3|1.88740|1.88408|0.82024|0.04288|0.15357|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|exact_same_readout|3|1.89016|1.87860|0.82024|0.04459|0.15357|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|known_observer_oracle|3|1.83494|1.72391|0.95714|0.16044|0.15357|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|native|3|1.00832|1.42369|0.93534|0.04129|0.06020|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|exact_same_readout|3|1.01064|1.42469|0.93534|0.04164|0.06020|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|known_observer_oracle|3|1.01076|1.42432|0.93645|0.04337|0.06020|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|native|3|0.64413|1.11787|0.91431|0.07627|0.14636|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|exact_same_readout|3|0.64674|1.10498|0.91803|0.08200|0.14636|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|known_observer_oracle|3|0.52566|0.96780|0.96546|0.15911|0.14636|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|native|3|0.82844|1.31790|0.93119|0.05728|0.09033|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|exact_same_readout|3|0.82997|1.31543|0.93247|0.06071|0.09033|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|known_observer_oracle|3|0.76867|1.26679|0.95414|0.09660|0.09033|

## 普通非线性观测SSM（已有方法强基线）

|条件|方法|已完成初始化种子|MSE|Gaussian NLL|95%覆盖率|预测尾部概率|真实/噪声条件期望尾部概率|
|---|---|---:|---:|---:|---:|---:|---:|
|state_dependent_alarm; L=1, H=0, alarm_no_anchor|generative_ssm|3|1.83532|1.72383|0.95952|0.16847|0.15357|
|independent_alarm_control; L=1, H=0, alarm_no_anchor|generative_ssm|3|1.00933|1.42435|0.93311|0.04067|0.06020|
|state_dependent_alarm; L=16, H=1, alarm_no_anchor|generative_ssm|3|0.52389|0.96585|0.96308|0.15520|0.14636|
|state_dependent_alarm; L=16, H=8, alarm_no_anchor|generative_ssm|3|0.76794|1.26652|0.95099|0.09191|0.09033|

生成式基线学习线性Gaussian传感器、全局观测噪声和普通MLP Bernoulli告警似然，使用常规数值边际似然训练。没有提供真实告警函数或潜状态标签。它有结构化传感器似然假设，不能宣称与原生RP具有完全相同的归纳偏置；匹配的是数据、更新数、batch大小与大致网络规模，未匹配CPU/GPU壁钟时间。

## 固定概率模型审计

通用Gaussian辅助界：16个固定参数cell，精确积分与160/320阶Gauss-Hermite Gaussian VI。原生正精度site兼容反例：9个cell。原始JSON包含全部正/负结果。V=1.5的低于容量上界对照接近零gap；当common和rare recognition完全相同时，真实posterior等于prior，而固定的收缩site仍会给出过窄posterior。这说明固定反例不能替代学习后的证据。

## 指标和数值限制

NLL与95%区间对数值后验和oracle使用其矩匹配Gaussian；不是精确分位数区间。尾部概率对数值后验保留完整grid分布。未来指标在给定真实当前状态后，对新的转移噪声解析平均，没有把未知未来噪声交给模型。

原生数值smoother会在方差上额外加1e-5；组件容量检查扣除了它。401点[-8,8]对照801点[-10,10]，原生冻结模型的固定64个样本均检查均值/方差误差小于1e-4。生成式基线另按5e-4检查。通用概率模型/辅助界5项独立检查与生成式似然/梯度2项检查在服务器通过。

## 经验reference的限定

原生冻结模型的F只由训练集组成。对未见连续观测，称为“训练混合reference下的plug-in ratio posterior”；原子经验p0对未见观测本身赋零质量，不能未经说明就称为一个连续观测生成模型的精确posterior。给定ratio时数值归一化是精确的，reference估计与模型错设仍可能有误差。较大batch/reference、容量、其他数据种子尚待控制。


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
