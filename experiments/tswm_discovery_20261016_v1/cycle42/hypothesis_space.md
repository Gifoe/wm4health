> 完成后的科学裁决：ordinary双侧factorization有人工heldout收益；新框架贡献尚未建立，seen代价与优化限制须同时保留。见[SCIENTIFIC_DECISION](SCIENTIFIC_DECISION.md)。

# C42 Competing hypotheses

服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对131072条未来序列score和524288个next-probability cells，max NLL差4.25e-07。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。

- H1：观测/动态分离只是ordinary结构先验。预测11可学习，甚至最好，但没有独立新机制；本轮11概率模型与候选相同，结构层面已成立。
- H2：两侧共享有额外有限数据交互。以预注册I及其全部world/init范围检查；正均值不足总体证明，见实际表。
- H3：差异由optimization/参数量解释。需要强ordinary joint SSM足够优化和active-capacity/compute controls，不能从本轮完全排除；保留cap/curve计数。
- H4：任务信息不足。C41给出了对称性反例；本轮用asymmetric transitions及同信息anchors，并用known-law oracle和可学习ordinary模型区分。未知sensor任意泛化不作为目标。

下一步不得以latent坐标对齐替代observable-law指标，或者将弱conditioner差距说成整个领域失败。
