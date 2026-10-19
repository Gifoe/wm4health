UTC 2026-10-16T19:21:54.232219+00:00。CPU诊断完成；原归因假设被否证。接受方向0，总体目标ACTIVE_UNACHIEVED。

# 竞争解释

| 假设 | 诊断 | 判断 |
|---|---|---|
| 三角causal速度约束足以保持joint | 同边际独立U重放 | 否；24/24跨链case的covariance错误 |
| 上述投影错配只由反馈造成 | 无反馈及独立重放控制 | 否；原归因被混杂，不能立闭环特有Gap |
| 结果只是条件回归公式或积分bug | MC500k系数、Schur、DOP853、步数加倍 | 核验通过，错误期望来自科学假设 |
| 正确causal概率模型也无法解决 | 已知三角正控制/同日志Gaussian MLE | 否；普通正确模型可表达/拟合 |
| 因果生成必然需要H步串行teacher rollout | IAF/PDD/逆一致性方程 | 不成立为普遍动机；已有并行学生强基线 |

原始计划与修订分别固定在TRANSPORT_DIAGNOSTIC_PLAN.md、DIAGNOSTIC_AMENDMENT_V2.md；数值实现检查不认证创新性。
