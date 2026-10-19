# C34 竞争假设

| 假设 | 判别 | 当前状态 |
|---|---|---|
| 暴露source noise改变原生生成法则 | same-noise API/cache/source独立重放 | 所测H96中反驳：最大差0 |
| 显式noise route不保留原生梯度 | 原始sample自动微分与中心差分 | 所测fixture通过，误差约0.001255 |
| first-hit监督提供不同于既有conditional channel拟合的新目标 | 函数展开与先例逐项映射 | 反驳当前唯一新意；未接受 |
| 稀有query的proposal训练可摊薄总成本 | 同批量MC/CE/MIS/RQMC、完整训练成本与独立context | 未测；廉价MC提高对照门槛 |
| 上下文分支补偿数值分支共同训练损失 | JZ、JZ+N、保护latent、保护输出残差，共同E/初始化/预算 | 未运行；不是已证明缺口 |

若推进最后一项，应先用TRAIN/VAL pilot判别，包含额外数值objective、no-context同容量、普通输出残差和模型组合，不根据TEST重新设计结构。潜态误差只能作诊断，不能代替预测效用。
