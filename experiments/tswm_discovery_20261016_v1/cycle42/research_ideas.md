> 完成后的科学裁决：ordinary双侧factorization有人工heldout收益；新框架贡献尚未建立，seen代价与优化限制须同时保留。见[SCIENTIFIC_DECISION](SCIENTIFIC_DECISION.md)。

# C42 Candidate status

服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对131072条未来序列score和524288个next-probability cells，max NLL差4.25e-07。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。

候选A的最初离散形式已经明确是ordinary因子化模型。该组合复用**研究目标**可以保留，但尚缺少超出已有结构建模的具体必要交互；不批准为最终新idea。若后续仅扩宽MLP/增加latent loss/换数据集，不足产生新颖性。

候选B（共享coarse law×MLMC）保持备选，无本轮效果；mr-Diff/WaveletDiff/generative MLMC近邻已存在，不能直接声称新。最多保留这两个调查方向，当前没有被实验证明的最终推荐。

WorldTS是贡献范围参考：同一框架可复用的预测状态/动态能力与多任务证据。单个HMM人工集合的优化不作为成果终点。
