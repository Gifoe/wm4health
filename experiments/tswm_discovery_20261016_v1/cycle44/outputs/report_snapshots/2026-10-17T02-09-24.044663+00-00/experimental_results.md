# C44 Actual evidence and unfinished comparisons

UTC 2026-10-17T01:52:25.004265+00:00。C44本轮是机制筛查PROGRESS；整体ACTIVE_UNACHIEVED，最终接受方向0。正式训练16/16完成，训练状态completed；已完成模型的正式updates共55500。VAL分配阶段running。独立EVAL质量阶段pending/running；EVAL truth reference pending。同拟合law high-sample reference、16-replicate estimator比较与实际allocated-call timing尚未完成，不能宣称新方法收益或完成研究目标。

资格检查：energy独立误差2.28198918e-08；条件梯度普通修复前向差0.0，启用后梯度max 0.000294584956；mixture logprob独立差2.35318346e-07；Kalman likelihood独立差0。30步资格训练全部丢弃，不用于效果。

VAL truth reference状态REFERENCE_READY：Gaussian exact Kalman与两particle counts/两rep核查；nonlinear count convergence核查只说明这些VAL histories数值可用，不能保证EVAL。强普通analytic/conditional-noise readout的代数资格状态PASS。普通prefix/coarse路径资格PASS。这些不是新方法效果。

本轮人工known-law EVAL与历史真实数据TEST不同；不要把其执行状态混同。详见各outputs/status.json原始时间戳。
