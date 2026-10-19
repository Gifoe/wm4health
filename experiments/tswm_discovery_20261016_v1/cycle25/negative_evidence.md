# 反证与被否定的推广

1. 否定“cycle24六个scalar checkpoint退化，所以通用非线性世界模型的新观测更新必然退化”。本轮unconstrained三个checkpoint的冷启动无信息probe CRPS点估计均改善；其他variant也非全部显著退化。旧结果保留且仍成立，不把不同proper score/plant硬拼为相同实验。
2. 否定“物理上无信息的观测改变预测，所以真实预测分数一定更差”。原生近似prior有误差，无信息更新有时补偿误差。忠实性与平均分数必须分别检验，不能把低CRPS当作物理更新正确。
3. 不支持“16D非线性后验使成熟推断普遍失效”。SLR和defensive IS总体能工作，冷启动ESS充分；full/diag Gaussian与粒子均已运行。少数长历史低ESS须保留，不得夸大为普遍高维不可解。
4. 否定“positive版本比较是正权约束唯一变量”。它同时改变rescale_raw_stddev，不能做该因果归因。
5. 不把作者prior sampler的RNG共享现象当成新Research Gap。保留实际训练分布是协议要求；改成名义范围再评判会制造OOD。
6. 不将Bayes聚合、完整cov、粒子似然校正、观测校正/传播分离或flow条件采样重命名为新框架。相应近邻论文已经明确实现。
7. 保留两次远程公开源码下载失败和第一版复核的batch数值误差。没有因失败重启主比较、换checkpoint、改cohort或按测试选方法。

本轮最终接受方向0；当前工作线索不能据此声称顶会级新颖性或新方法必要性。
