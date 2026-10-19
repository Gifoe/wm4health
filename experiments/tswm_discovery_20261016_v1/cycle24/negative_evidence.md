# 负面证据和被否定的扩大解释

1. “只是未来归一化/坐标错误”：本轮使用独立原生因果核，连续前缀、固定query初值与全范围来源校验；新failure仍出现。与cycle23那份WH生成器问题分开。
2. “只是模型latent坐标变化”：真实物理query联合NLL和覆盖率确实恶化；不是只看latent entropy或latent mean。
3. “已证实整个时间序列world-model领域都不能更新证据”：没有。只测一个公开方法、两个actuation变体、三个checkpoint seed，不能外推全领域。
4. “通过不更新就解决了问题”：不更新对informative probe明显差，需要同时检查信息利用和无信息保护。
5. “一致性更高就必然更好”：普通完整Bayes并不在所有真实条件上优于原生更新；decoder/likelihood有模型误差，摊销更新可能补偿它。一致性与truthfulness需分开验证。
6. “必须另造新架构”：普通defensive importance校正+Gaussian矩投影已经修复大部分冷启动损害。generic Bayesian correction不是创新；本轮没有证明必要的新结构。
7. “EKF/MAP的不足说明老推断都无效”：不成立。EKF是局部线性近似；MAP-GGN还存在局部优化/单模态限制。更完整粒子对照已给出相反证据。
8. “坏NLL就是科学失败”：原GH积分、EKF尾部及少数全量尾界问题均先排查并保留原尝试，修复数值后才生成表。独立verifier的float32 log(scale)导致ESS差异也只修复verifier dtype，没有放宽阈值或改producer。

未证明的部分：高维非线性系统、其他模型、长时联合模拟、部分可观测条件下同一失败是否重要，及已有高质量推断是否仍不足。不得用扩大模型/数据规模维持一个已被普通方法解决的论点。
