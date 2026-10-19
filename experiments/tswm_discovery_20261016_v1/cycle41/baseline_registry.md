# C41 原生与控制注册

本轮研究单位仍是可复用时间序列 world-model 框架；数据不是优化终点。选择一个后续开发候选，不代表已接受新颖机制或完成总目标。

- LIFTS固定commit e42430980e51bebf2e1b54d1beddeda7a31dee51：26个原始source文件SHA/Blob核验，C41只读组装后import/训练/采样。模型1330809参数，作者minimal case1设置，5loops/15minibatch更新，80train/20interface check。执行lambda_enc=1，与论文主要实验0不同；只称作者示例资格，不称论文性能复现。
- 原生fit约8.4389s，进程peak 1.4897GiB。运行前其他GPU负载为 3203 MiB, 17 %；耗时不作隔离性能benchmark。
- Imputation/forecast RNG重放、masked-value污染、forecast过去半段的future污染、observed硬替换，误差均0；评测过程权重不变。保存config/scaler/checkpoint/data/latent truth/samples/rawlogs。
- 精确MCAR Kalman与block Gaussian，有限100步burn prior，原生Cholesky有效covariance；raw float32 state arithmetic用声明的float64 LG model近似。坐标变换只比较observable law，不比较任意state坐标。
- 四状态补偿/ordinary transition修复，以及全部24 observation permutations的Bayesian mixture。真实模型参数作为privileged control给出，不冒充未知参数公平模型。
- NCDSSM/modular NRI/PSIM等本轮为文献完整对照，未执行。近期候选新方法尚未训练。
