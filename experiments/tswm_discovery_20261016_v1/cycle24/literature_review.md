# 文献与创新边界

本轮新增阅读[Variational meta-learning inference for low dimensional neural system identification](https://arxiv.org/abs/2607.18965)，2026-07-21。其方法已有潜在系统参数流形、摊销初始化、MAP与Laplace近似。本文只核查相关方法边界，未复现该JAX实现，也未独立认可其全部数学/校准主张。原文已保存outputs/literature/，SHA256可查。

旧边界继续成立：[Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html)已有上下文条件动力学与主动校准；[MPNP](https://arxiv.org/abs/2304.09431)、[From Drift to Coherence](https://proceedings.mlr.press/v306/kim26ah.html)、[CNP conditioning gap](https://arxiv.org/abs/2604.19312)覆盖概率一致性与近似推断的重要部分。CNP factorized-joint条件化与本轮共享latent联合律的tower检验不是同一量，不能混为一谈；但泛化的“更新一致性”并非无人研究。

EKF、MAP/Laplace、importance sampling、defensive mixture proposal、Gaussian moment projection/assumed-density filtering均为已知方法。本轮把它们用于原生检查点的失败归因，不是提出新算法。不能把“编码器提议+似然校正”改名为新的世界模型框架。

当前有实测的可靠性问题，但没有足以证明现有方法仍不足的证据。更广的高维、非线性、部分可观测场景仍需检验；也不能仅通过让已有方法进入不合理的预算或支持范围制造失败。
