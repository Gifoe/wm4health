# C30 精确Gaussian诊断：原归因被否证

V1原计划预期独立输入控制通过，但在该控制exit1。V1保留；V2只验证计算并记录该负面事实，不降低原阈值。

| protocol (k非0,a非0) | method | cases | mean max gain error | covariance relative Fro error | do error>1e-3 |
|---|---|---:|---:|---:|---:|
| feedback | full_conditional_fm | 24 | 0.482222 | 0.612013 | 24 |
| feedback | causal_mask_fm | 24 | 0.296486 | 0.475773 | 24 |
| matched_independent_replay | full_conditional_fm | 24 | 3.52603e-15 | 2.02701e-15 | 0 |
| matched_independent_replay | causal_mask_fm | 24 | 1.77636e-15 | 0.274541 | 24 |

1024/2048 RK4、Full FM的Schur目标、独立DOP853积分、500000样本速度条件回归、a=0结构正控制和3seed正常方程检查全部通过。74协议case及222个有限样本MLE拟合全部公开在本地/服务器实验目录；它们来自同一两步机制，不是74个独立机制或公开模型成绩。

结论：直接投影到causal速度依赖并不保证保持joint，这一错配在外生输入下也存在，不能作为闭环特有新Gap。Full conditional FM的反馈偏差仍属于已知普通条件/因果条件区别；正确causal Gaussian likelihood及三角生成器是强普通修复。本例没有证明现有causal flows不足，更没有得到新方法。

后续只有核查已有causal normalizing flows、P-CFM、MAF/IAF、Parallel WaveNet密度蒸馏后仍有具体有限预算问题才可立项。当前接受方向0，不启动神经长训练。
