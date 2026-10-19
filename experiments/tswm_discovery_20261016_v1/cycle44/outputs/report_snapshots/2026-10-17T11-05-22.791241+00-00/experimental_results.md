# C44 Actual evidence and unfinished comparisons

UTC 2026-10-17T02:09:24.044663+00:00。C44本轮是机制筛查PROGRESS；整体ACTIVE_UNACHIEVED，最终接受方向0。正式训练16/16完成，训练状态completed；已完成模型的正式updates共55500。VAL分配阶段completed。独立EVAL质量阶段scored, independently verified；EVAL truth reference REFERENCE_READY。同拟合law high-sample reference/16-replicate查询阶段running；独立CPU查询分析pending。尚未据此确认任何新框架，不能宣称完成研究目标。

资格检查：energy独立误差2.28198918e-08；条件梯度普通修复前向差0.0，启用后梯度max 0.000294584956；mixture logprob独立差2.35318346e-07；Kalman likelihood独立差0。30步资格训练全部丢弃，不用于效果。

VAL truth reference状态REFERENCE_READY：Gaussian exact Kalman与两particle counts/两rep核查；nonlinear count convergence核查只说明这些VAL histories数值可用，不能保证EVAL。强普通analytic/conditional-noise readout的代数资格状态PASS。普通prefix/coarse路径资格PASS。这些不是新方法效果。

独立held-out full-law scores（不是查询收益或框架创新证明）：

|model|joint energy ↓|mean MSE ↓|marginal90% coverage|
|---|---:|---:|---:|
|gaussian_ar_mixture_2026441|0.6867024|0.9605021|0.86774|
|gaussian_ar_mixture_2026442|0.6874094|0.9624191|0.86815|
|gaussian_lgssm_2026441|0.6853878|0.9571471|0.87344|
|gaussian_lgssm_2026442|0.6852043|0.9562825|0.86990|
|gaussian_spf_condition_trainable_2026441|0.6989573|0.9883678|0.83974|
|gaussian_spf_condition_trainable_2026442|0.7045032|1.0103286|0.89091|
|gaussian_spf_native_2026441|0.6995348|0.9905006|0.84122|
|gaussian_spf_native_2026442|0.7024512|1.0038967|0.86978|
|nonlinear_ar_mixture_2026441|0.3372209|0.2313709|0.86720|
|nonlinear_ar_mixture_2026442|0.3372108|0.2314198|0.86986|
|nonlinear_lgssm_2026441|0.3464305|0.2450122|0.88362|
|nonlinear_lgssm_2026442|0.3458733|0.2442610|0.88362|
|nonlinear_spf_condition_trainable_2026441|0.3461070|0.2411484|0.84247|
|nonlinear_spf_condition_trainable_2026442|0.3470025|0.2412409|0.80807|
|nonlinear_spf_native_2026441|0.3467905|0.2400801|0.82379|
|nonlinear_spf_native_2026442|0.3464106|0.2415759|0.82076|

每行64独立histories/64 iid futures，原始paths/targets保留。所有family在各自预算下训练，不能单凭排名归因架构。

本轮人工known-law EVAL与历史真实数据TEST不同；不要把其执行状态混同。详见各outputs/status.json原始时间戳。
