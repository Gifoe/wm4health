UTC 2026-10-16T19:21:54.232219+00:00。CPU诊断完成；原归因假设被否证。接受方向0，总体目标ACTIVE_UNACHIEVED。

# 完整诊断结果

| 协议（k≠0,a≠0） | 速度目标 | cases | gain最大绝对差的均值 | covariance相对Fro差的均值 | do误差>1e-3 |
|---|---|---:|---:|---:|---:|
| feedback | full_conditional_fm | 24 | 0.482222 | 0.612013 | 24 |
| feedback | causal_mask_fm | 24 | 0.296486 | 0.475773 | 24 |
| matched_independent_replay | full_conditional_fm | 24 | 3.52603e-15 | 2.02701e-15 | 0 |
| matched_independent_replay | causal_mask_fm | 24 | 1.77636e-15 | 0.274541 | 24 |

全部148个flow结果及222个MLE拟合保留。有限MLE的gain误差均值0.00695211，covariance相对Fro误差均值0.0167785。仅3种子和同一机制，不是泛化框架证据。

Full FM匹配Schur最大相对差2.22e-14；两分辨率差<3.3e-14；独立DOP853差<4e-12；MC速度系数差<.0067；a=0结构控制通过。V1因错误控制预期exit1保留，V2未降低积分/公式阈值，只修正假设并记录所有负面控制。

独立输入下causal-mask的mean gain正确而covariance错误；只报告平均轨迹分数会漏掉这种错误。它是本naive投影目标的诊断，不是所有causal模型缺陷或未解决新Gap。
