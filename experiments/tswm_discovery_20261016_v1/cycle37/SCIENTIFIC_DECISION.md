# C37 root研究裁决：关闭当前新机制动机，不做站点优化

UTC 2026-10-16T23:06:41.197043+00:00。仅消费完整verified结果，不由partial日志选择故事。12个neural fits/48000updates、2个HGB、56个paths arms；原训练1416.234秒。独立12selected模型/2树点模型/504score cells通过，score最大差1.21156307209e-10，native prediction重放最大差0。TEST没有评分。

当前结果不支持由此提出新的时间序列world-model框架。下面是实际区分出的普通解释，不是数据集排行榜推荐。

1. **日轨迹依赖确实影响评测，但经典rank coupling已能改变它。** 同一MLP每个lead的128个值完全不变，CAL-PIT历史秩相对IID的daily-mean CRPS：wind +9.94%，absolute paired CI [0.009582, 0.011849]；solar +7.76%，absolute paired CI [0.001144, 0.001749]。hourly CRPS改善均为0（exact marginal multiset恒等性另核）。这支持指标对path coupling敏感，不支持“需要新state机制”；Schaake/ECC是已知解释。

2. **L1源码不能直接归罪，标准L2没有修复本轮泛化差距。** raw L2相对L1的hourly CRPS：wind -2.36%，absolute paired CI [-0.021497, 0.014139]；solar -2.62%，absolute paired CI [-0.002192, 0.000693]。两项CI跨0，点估计均不利L2；不能声称显著优劣，更不能说原L1必然导致所见失败。10→50steps也没有稳定跨任务主指标收益；全对比保留原表。

3. **校准有用，但它是共同普通控制。** raw→同CAL校准各KITE主边际都有改善；校准后的L2仍在wind明显落后MLP rank+CAL/HGB，solar差距较小且CI跨0。由这些有限拟合不能定位唯一state缺陷。HGB在wind两主CRPS点值更低，但模型/预算/FLOPs不严格匹配，不能把它推广为所有world model无用。

4. **joint并非任何路径指标都必胜。** solar HGB analogue joint相对同marginal IID的daily-mean CRPS -3.34%，absolute paired CI [-0.000980, -0.000021]，而ramp CRPS +28.26%，absolute paired CI [0.007842, 0.020049]。这是真实已存同样本的评分差异；不据一个路径指标挑最有利copula。其CI仍是未校正多重比较的探索区间。

5. **不是训练没动，也未证明最优收敛。** 所有12fits都较step0改善，选中step0为0，末1000updates>1%改善flag为0；wind6fits全选500，随后VAL变差。它更像有限数据/正则/归纳偏置问题，不能简单靠追加steps挽救；也不能声称无限训练已被穷尽。2seed、wind VAL66/105完整cases、solar118、仅各zone1，均限制泛化。

6. **C36 forcing压缩前提仍未建立。** 只有NWP与目标功率，没有已识别true future weather标签；CAL target residual相关/低effective-rank不是weather forecast error。128条paths只是分布评分MC，不是外部forcing积分K压力。不能把本轮差异改名为q(U|F)新缺口。

**行动裁决：** 不对这些zone继续调loss、rank、seed或训练长度，不评分TEST来追一个更好数字。保留原权重、路径、来源和全部固定CI；本轮只为一般框架研究排除了错误归因。接受最终方向仍0，整体ACTIVE_UNACHIEVED。C38异质条件补查同样没有给出明确新交互，不能把普通frontend修复包装成交付。

下一步研究优先级是明确一种可复用状态/转移学习作用，再与最强普通完整方法做可失败比较；不是在上述两任务上挑一个冠军作为idea。latent/readout效用失配与稀少有价值context的学习保留仍是未隔离线索，C35具体protected residual配置保持关闭。

详细数字/所有7个loss及2seeds：[完整结果](experimental_results.md)、[742预列配对行](outputs/analysis_v1/paired_comparisons.csv)、[机器分析](outputs/analysis_v1/analysis.json)。图已由root目视检查标题/轴/seed SD说明/完整曲线，PNG/PDF均保存，无TEST。
