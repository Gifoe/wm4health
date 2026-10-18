# C80 文献补查：上下文支持、动力学后验与新增信息

本补查延续C79的三个研究问题，主流、批评、相邻辨识三路独立检索，再核验关键原文和公开实现。重点2026-04-18至10-18，必要时回溯。目的不是把每篇限制都变成Idea，而是判断C80实测新初值现象是否独有。以下均为作者结果/范围，除FIM外没有在本轮执行这些方法。

## 1. 保持上下文拟合、构造不同的动力学

[Giampiccolo、Iacca、Marchetti，2026-05-27，MOD](https://pmc.ncbi.nlm.nih.gov/articles/PMC13494032/) 已直接研究普通NODE/UDE种子ensemble在轨迹外过度自信。其方法在保持训练轨迹损失的条件下，增大选定区域的向量场分歧，并检查新初值模拟；[Julia代码](https://github.com/cosbi-research/UQNU) 已发布。区域、阈值和区间近似会影响结果，不能称为任意OOD的校准后验。它直接阻止把“多个同样拟合的场在别处不同”作为新发现。

相比这个约束分歧构造，[Hegde等，UAI2022，GP-ODE](https://proceedings.mlr.press/v180/hegde22a.html) 从含噪轨迹学习场后验，并以连贯函数样本积分；[实现](https://github.com/hegdepashupati/gaussian-process-odes) 是更直接的普通贝叶斯基线。[Magnani等，ICML2025，LUNO](https://proceedings.mlr.press/v267/magnani25a.html) 则在线性化的已训神经算子上构造函数值GP，实验中需要250个有标签验证对作校准，不能当作零标签控制。这三者不同于每步独立重抽点态方差，后者会改变固定未知场的生成假设。

## 2. 上下文系统分布与主动获取信息

[Qing等，SANODEP](https://arxiv.org/html/2406.02352v2) 使用多条轨迹推断系统潜在分布，并以任务目标选择初值/测量计划；[JAX实现](https://github.com/TsingQAQ/SANODEP) 及外部权重链接已找到，权重本身未核验。它解决的是任务BO，不能升级为所有新初值的可信度证书。其更早的 [Neural ODE Processes](https://arxiv.org/html/2103.12413) 前身也已有上下文随机动力学和观测选择，不能只换成foundation一词重提。

[Larrañaga、Fasel、Brunton，2026-06-10，Active-SINDy](https://arxiv.org/html/2606.12182v1) 通过稀疏模型ensemble的导数分歧选择ODE初值，获得新轨迹后更新；[真实Python实现](https://github.com/alarran/active-sindy) 的停止规则基于稀疏模式和系数稳定性，而非未来评估系数。它依赖状态可见、可重置初值和合适函数库。PDE部分使用候选的观测导数信息，不能一并称为无候选标签的获取。我们执行的弱积分回归不是该论文算法复现。

若初值不能自由重置，[OPAX，NeurIPS2023](https://arxiv.org/html/2306.12371v2) 已沿可达行动策略规划信息，不能把只在任意点可查询的实验与它公平并列。其不确定性校准和函数类假设需保留，[JAX实现](https://github.com/lasgroup/opax) 在5090上的兼容性未核验。本轮未执行主动数据获取。

## 3. 不用未来标签选择适配

[Huang、Michailidis，2026-09-21，PBML-LTI](https://arxiv.org/html/2609.24117v1) 已用可用上下文内的验证后缀选择适配prefix，再重拟合；[实现](https://github.com/chenfeng-huang/PBML-LTI-TMLR-2026) 可查。其证书依赖query/support Gram关系，不是任意查询的support-only保证；某个正则界仍有未知真动力学项。它证明普通上下文验证已有明确方法，却没有证明任意非线性陌生世界可被这样认证。

FIM-ODE原文已承认路径外欠定，也已披露训练损失选择和测试窗口选择的差别。辅助不确定性头主要作向量场损失加权；本轮没有测它的校准或场后验，不能写成已经发现它“没有不确定性”。只测试与接口契约相符的能力，才可能分辨实际残差。

## 4. 对研究问题的判断

RQ1：公开底座既有点估计场，也有概率场、连贯采样和主动试验，不是统一的“只会续写”。RQ2：C80支持数值执行真实、部分新初值有困难；普通回归呈异质结果，成熟的支持/先验/后验解释仍未排除。RQ3：当前没有一个达到接受标准的新机制方向。已有论文的假设边界不等于其外部问题已解决；同样，单个已知摆失败也不足以宣称一般性研究缺口。

接下来的证据若仍研究支持，应在固定数据预算下区分重复采样和几何/激励多样性，排除先验家族、积分、噪声和上下文验证因素，并对比GP-ODE、MOD/ensemble、普通辨识或主动采集。只有独立系统残差仍存在且最小方法优于匹配强基线，才接纳方向。不能因为现有方法有条件，就假定我们的版本更有创新。
