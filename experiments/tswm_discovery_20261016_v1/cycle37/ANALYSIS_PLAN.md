# C37 分析冻结补充

2026-10-17，在C37全部结果及独立验证完成前保存。当前原worker仍在wind最后一个neural fit，不基于partial联合评分选择对照。本文件不改变训练/分割/选模/评分协议，不打开TEST。

使用2000次paired **7个日历日**block bootstrap，按origin而非小时/path抽样。完整case的时间缺口按原始日历分组，不能把不相邻保留行当成连续一周。固定seed2026370；percent change同时给绝对差，coverage/width不当作越大/越小越好的loss。

预列比较：
- 同quantile fit：shuffle/IID、analogue/IID，raw和同CAL版本。
- 同HGB mean：unconditional joint/IID、analogue joint/IID。
- 配对KITE初始化/schedule：L2/L1，raw和CAL版本。
- 每个family的CAL/raw（quantile的三个coupling版本，KITE两个目标）。
- 同selected权重的50steps/10steps，KITE两个目标。
- KITE L2 raw及CAL分别对quantile analogue CAL，以及HGB analogue joint；这些跨家族比较不是严格参数/计算匹配。

主分数marginal CRPS与daily-mean CRPS；ramp CRPS/energy/variogram/MAE/MSE为辅助。所有对比都报告，不只保留正结果；CI为探索性描述，未作多重检验校正。两seed的原始结果分列；seed平均指score平均，不是forecast mixture或新增ensemble。

附带只用CAL计算HGB目标残差的跨lead covariance/effective rank；它不是weather forecast error，也不是新机制证据。预列图：marginal/daily CRPS两任务比较、所有neural训练/验证曲线。必须先通过独立raw/window/score/权重重放，再生成用于科学判断的分析。

无自动方法接受阈值。已知校准/rank/标准objective/数值修正解释的收益不构成新world-model；即便generative control更好也仅是已有方法证据。对不足训练/缺失窗口选择/VAL复用/小样本与不同容量保留明确限制。
