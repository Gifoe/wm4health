# C41 选择后续开发路线

优先候选A：**可更换的分布性观测接口 × 可重组动态模块**。目标是同一world model在有合理同步锚的sensor–system新组合上复用已学机制，测有限数据下的预测质量和适应成本。每个领域可分别训练同一框架；不强制万能checkpoint。当前证据只支持这是一个合法、可证伪的问题，尚未支持新架构。

备选B：共享粗路径marginal × 多层路径函数估计。已有mr-Diff/WaveletDiff/MLMC是完整近邻，须真正胜过cheap GRU/现成MLMC的真实误差—成本。暂不启动。

主流提出的law distillation×有限state预算保留为诊断控制，不优先命名新框架：ordinary teacher distillation、PSIM等重叠重，且不能重开此前被关闭的target/readout故事。三份独立notes保留开发细节与相互竞争解释；不得只挑建设性建议忽略反证。
