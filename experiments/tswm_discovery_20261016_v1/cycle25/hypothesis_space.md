# 竞争解释与实测裁决

| 假设 | 区分实验 | 目前裁决 |
| --- | --- | --- |
| 无信息probe含隐含物理参数信息 | 源核独立递推、跨gain零控制、reset身份检查 | 排除；max差0，独立核4480检查通过 |
| 更新失真只来自1D线性toy | 原生16D latent非线性、四局部机制 | 物理预测律仍会改变，但质量损害不普遍；不能泛推新架构 |
| 受约束encoder稳定优于无约束 | 两套发布变体、三个seed，同cohort | 不支持；并有rescaling混杂 |
| 全部偏差是摊销encoder导致 | 同decoder SLR/IS与原生重编码 | 部分更新差异由encoder造成；旧方法可避免大量变化，decoder/prior也有误差 |
| 丰富后验是必须的新机制 | 粒子与相同矩diag/full Gaussian | 没有成立；简单近似足以构成强反证 |
| 现有方法完全不利用新信息 | probed quadrant与其他query分开 | 排除；所有6个原生checkpoint的probed CRPS显著改善 |
| 所有Bayes/particle结果可信 | 固定预算3MC、ESS、独立权重与SLR求解 | 总体数值通过；极少长历史低ESS仍限制局部结论 |
| 通用更新一致性是新题 | BA/structured NP/FlowNP/CNP gap/OSSM核查 | 否定该宽泛创新主张；更窄问题仍须新证据 |

不通过增加相同toy规模来维持原路线。任何继续的候选都应先给出当前成熟方法尚未解决的具体量，并在新的held-out系统/现代模型上检验。当前没有被实验证明需要的新架构。
