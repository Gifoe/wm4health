> 完成后的科学裁决：ordinary双侧factorization有人工heldout收益；新框架贡献尚未建立，seen代价与优化限制须同时保留。见[SCIENTIFIC_DECISION](SCIENTIFIC_DECISION.md)。

# C42 Negative evidence and rejected claims

服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对131072条未来序列score和524288个next-probability cells，max NLL差4.25e-07。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。

1. **拒绝初始架构新颖性主张**：11等于ordinary structured HMM；不能把同一模型另命名后报告超越ordinary baseline。
2. 00/10/01若有失败，只能归因到这些具体conditional parameterizations与训练预算。其输入有同样信息，未见cell由shared MLP生成而非未训练lookup，但这不代表所有单体world model。
3. 人工worlds按factorization生成，本轮只能筛选可学习性及known-component inductive bias，不能说明真实系统普遍存在同样可组合机制。
4. 只2world、2initializations；名义参数一致，有效参数有差；15/32选cap，18/32末段仍改善。未完成计算/容量/优化充分性及跨现实域强基线闭环。
5. 强known-parameter oracle与learned方法信息不同，只作truth reference；不以其差距直接命名representation failure。

没有通过改变noise、训练cell或seed挽救正面故事。原C41的ordinary正确dynamics/充分anchor控制结论继续有效。
