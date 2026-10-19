# C35 竞争假设的实际状态

| 假设 | 判别证据 | 可解释范围 |
|---|---|---|
| 保护分支仍被context梯度更新P | 自动微分0梯度、V2四arm逐step P/hash与numerical分数相同 | 反驳该实现错误；V1停止并保留 |
| context补偿数值退化 | JZ/JZ_N numerical相对N同step；fixed-batch梯度余弦 | 观察性训练诊断，不由单一latent norm推出 |
| 额外数值loss已解释保护收益 | JZ_N vs PZ，同forward/loss但不同梯度通路 | 预算内clean gradient-treatment比较 |
| latent结构优于普通output residual | PZ vs PY、PZ_NC；预列双任务≥2% screen | False；还捆绑辅助loss，不是纯latent因果归因 |
| added capacity解释context收益 | 同容量PZ_NC | 只相对所学target representation；不是全raw历史条件独立证明 |

- solar：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 2.366% / -1.627% / 9.549%；预列screen passes=False。
- jena：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 0.000% / 0.000% / 0.000%；预列screen passes=False。

- solar：在JZ_N自己的所选step，与N同step比较的数值支路MAE变化为 -0.933%（正值更差）；固定TRAIN batch的五个梯度余弦为 0.4610, 0.3804, 0.0508, 0.1691, 0.4645。不是总体梯度分布或因果证明。
- solar：统一step2000的numerical MAE为N 0.30649836、JZ 0.30151334、JZ_N 0.30162983；JZ相对N变化-1.626%，JZ_N相对N变化-1.588%。所选step0不会掩去这些完整训练轨迹，但它们也不能取代预列VAL选模。
- jena：在JZ_N自己的所选step，与N同step比较的数值支路MAE变化为 +0.000%（正值更差）；固定TRAIN batch的五个梯度余弦为 0.6546, 0.0643, 0.7542, 0.3420, -0.1256。不是总体梯度分布或因果证明。
- jena：统一step2000的numerical MAE为N 0.14888102、JZ 0.21013599、JZ_N 0.14850502；JZ相对N变化+41.144%，JZ_N相对N变化-0.253%。所选step0不会掩去这些完整训练轨迹，但它们也不能取代预列VAL选模。

TEST未评分；不得用未来TEST选择新变体。若继续候选，必须另冻结独立确认、强native/exogenous/output baselines和多seed，且检查teacher/arm预算不足、readout与辅助loss混杂。
