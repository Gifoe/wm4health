# C34 负面证据与停止决定

1. 撤回以 first-hit labels 监督 proposal mixture 为唯一差异的新方法：目标函数是普通 observed-channel conditional MLE，有直接先例映射。未运行长prototype。
2. Sundial资格通过只排除adapter错误，不证明原模型动力学不足或我们的框架更好。
3. 1024paths约0.084秒的批量普通采样使加速主张必须包含完整训练/推断成本；未据此断言所有IS都无用。
4. PathFM的未发布实现不能作为可执行底座，也不构成Research Gap。
5. Pythia latent/forecast聚合不同，不能由表格反转直接推断上下文补偿；尚需同目标空间的配对机制实验。
6. 先前C33普通decoder结果关闭NP/PJM target/readout路线；不以更多同两任务调参重新包装该路线。

已知组件组合仍可能在有限数据/计算下形成贡献，本轮撤回基于具体等价与缺失证据；不要求新方法必须击败无限预算的全部经典算法。
