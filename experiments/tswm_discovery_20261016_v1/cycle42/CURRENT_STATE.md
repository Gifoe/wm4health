> 完成后的科学裁决：ordinary双侧factorization有人工heldout收益；新框架贡献尚未建立，seen代价与优化限制须同时保留。见[SCIENTIFIC_DECISION](SCIENTIFIC_DECISION.md)。

# C42 — bounded screen completed, overall goal active

服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对131072条未来序列score和524288个next-probability cells，max NLL差4.25e-07。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。

未知参数观测/动态2×2 ordinary HMM已真实学习；选模只用seen VAL，人工heldout评分后未调参。所有raw scores/selected checkpoints独立重建。11本身就是ordinary structured HMM，当前不能把它当新框架贡献。结果、交互、优化限制见[experimental_results](experimental_results.md)。当前接受最终方向0；后续须发现区别于普通模型的具体机制，或换到更有证据的研究问题。
