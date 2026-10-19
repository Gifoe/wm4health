UTC 2026-10-16T20:12:23.034221+00:00。通用时间序列 world-model 框架发现任务 ACTIVE_UNACHIEVED；接受最终方向0。

# 实际执行：固定编码的任务兼容性

Y∈{−1,0,1}；等概率 contexts A/B，条件概率分别(.45,.40,.15)/(.60,.10,.30)。每context20行精确计数，合40行。固定编码 phi(−1)=(100,100)、phi(0)=(103,100)、phi(1)=(100,103)。所有 future target 的逐样本 Y 可用 `(phi1+2phi2−300)/3−1` 精确恢复。

| 项 | 结果 |
|---|---|
| 样本线性重构误差 | 0 |
| Target 两坐标 std | [1.299038105676658, 1.2527469816367551]，均>1 |
| Target covariance eigenvalues | [1.147439827712113, 2.192944787672502]，均>0 |
| 原生 loss 固定编码全局最优摘要 | contexts A/B 都是(100,100)，reg0与.1均有严格全局margin证书 |
| 原观测 MAE 最优预测 | A=0、B=−1 |
| 共用摘要最佳 decoder MAE | 0.70 |
| 区分contexts的 Bayes MAE | 0.65 |
| 不可避免 excess | 0.05=1/20 |
| 普通 protected raw anchor | 读取0/−1，MAE恢复.65；没有声称未保护cos/reg耦合anchor的保证 |
| 普通平方 target +重拟合 MAE读出 | Ephi区分两概率分布，正确读出0/−1，MAE恢复.65 |

作用域是固定编码、H1、给predictor的两个context、decoder额外side information相同。WorldTS实际decoder还接收历史状态与RevIN统计；这些若包含context，可以修复collision。因此本例不是全模型失败。更不说明自由联合训练会选择此编码。

源组件与独立NumPy公式128次比较，最大relative error 1.67e-15；reg0/.1各2304个预定随机扰动均满足正risk gap。大扰动quartic covariance造成最大absolute差约7.63e−6，实际relative差约1.67e−15，使用执行前的abs+relative容差。全局最优由解析margin证明，扰动/数值优化不是证明替代。

独立 verifier 用 Fraction 有理数重算1/20、独立解线性重构系数、枚举4001个raw决策、检查2个全局证书、128个原组件比较以及普通读出；全部通过。科学执行 1.074s，verifier 0.240s。见 [结果](outputs/compatibility_witness_v1/result.json)、[验证](outputs/compatibility_witness_v1/verification.json)。

这一步证明的是一个范围受限的数学现象，没有真实benchmark分数或新方法优势。
