# 第二轮文献与创新边界

更新时间：2026-10-15T21:07:41.248013+00:00。这是第二轮工作记录；上一轮目录及结论未改写。研究总目标仍未完成。所有数值只从已保存结果读取。

重点核查了与当前机制直接相邻的原始论文，而非用“没有搜索到”证明首创。

|原始来源|核查的机制|对创新主张的约束|
|---|---|---|
|[Walker et al., AISTATS2023](https://proceedings.mlr.press/v206/walker23a.html)|Eq.8–9给出内部变分界及其额外KL；还比较了普通重参数化VI与二阶近似|近似gap、Gaussian VI和这些修复路径均已有，不能重新命名为创新|
|[RP-LDS](https://discovery.ucl.ac.uk/id/eprint/10198967/1/RP_LDS_camera_ready.pdf)|早期Gaussian recognition动态模型|广义Gaussian recognition与当前正site实现应分开|
|[RP-GSSM, ICML2026最终版](https://proceedings.mlr.press/v306/hromadka26a.html)|Sec3.1/C：full recognition = prior×Gaussian site，g=q/prior；Sec3.4明确承认近似可能偏置解；F.1用矩/MMD和矩匹配Fisher距离检查平均F接近prior|不能声称作者未讨论近似；潜在新点只能是具体条件性失败及其后果，不能是“Gaussian posterior会有误差”|
|[RAMP, ProbML2026](https://proceedings.mlr.press/v327/fox26a.html)|A.4同样使用内部界和prior替代；B.4用Cholesky precision生成有效Gaussian消息|更一般的非线性message passing已经存在；Gaussian有效site对不确定性扩张的约束仍须单独检验|
|[Discriminative Kalman Filter](https://burkh4rt.github.io/pubs/Burkhart-et-al-2020-Neural-Comput.pdf)|判别式后验与prior密度比更新；需要相应监督/分布条件|负相对精度、Gaussian density-ratio校正本身不是新方法|
|[Probabilistic contrastive learning, ICML2023](https://arxiv.org/abs/2302.02865)|已有条件下的aleatoric posterior恢复理论|概率embedding/恢复不确定性不是新研究目标|
|[UWM-JEPA](https://arxiv.org/abs/2605.25313)、[MDN-VJEPA](https://arxiv.org/abs/2610.13956)|已有belief/多模态latent未来|不能把一般“保留不确定性”作为创新|

新近邻 **Identifying Latent Neural Dynamics with Recognition-Parameterized Gaussian Process Dynamical Systems** 已出现在[作者2026-09-29论文列表](https://www.gatsby.ucl.ac.uk/~maneesh/researchli1.html)，列为NeurIPS2026。已确认题名与作者，正文当前尚未取得；不能声称已核查或已排除此近邻。OpenReview浏览出现challenge，没有绕过。当前不提出“首次非线性RP dynamics”之类主张。

完整4篇PDF、提取文本、SHA256与下载来源在outputs/literature/。最终版RP-GSSM与2025 arXiv的标题、理论与页数不同，当前判断依赖2026最终版。所有方法的理论、实际代码、我们的新实验分别记录。

DoTime的新审计直接调用固定版本方向分数、PFN batching、VAR基线。前者有prospective mask，VAR使用更长factual输入；二者信息集不能混为同一部署协议。方向分数受未知future factual噪声与共同校准偏差影响；但反向响应高分在正负剂量平衡后消失。这个协议现象不能单独构成新方法论文。


补充近邻：[Biased Dreams, 2026-04](https://arxiv.org/abs/2604.25416)分析latent RSSM ensemble的epistemic uncertainty与rollout attractor；当前受控现象使用正确的ID动力学，针对观测后aleatoric条件分布，不能把二者说成同一个实验或声称首次发现latent uncertainty误导。[VRKN/TMLR2022](https://arxiv.org/abs/2210.09256)与[真实代码](https://github.com/pbecker93/vrkn)已经研究RSSM推断引起的aleatoric偏差、missing observations与sensor fusion。本文及代码已检索，尚未执行其完整任务，不能列为已复现。通用不确定性失准和Kalman式修复不新。
