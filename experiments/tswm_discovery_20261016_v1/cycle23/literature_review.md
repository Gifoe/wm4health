# 文献与创新边界：系统身份与信念更新

本轮针对通用时间序列世界模型的跨系统推理能力筛选，不以单数据集预测提升作为研究目标。以下主要是源文献/代码接口核查，不声称全部完成论文复现。

1. [Probabilistic in-context system identification](https://arxiv.org/abs/2610.13411)，2026-10-09。已有跨系统共享模型、概率读出、长上下文和合成数据用途；不能将这些能力重新命名为创新。其源码已在cycle16/17缓存。本轮只新增生成器前缀配对审计，未重做原有协方差实验，也未重训该检查点。
2. [Martingale Posterior Neural Processes](https://arxiv.org/abs/2304.09431)，ICLR2023。利用预测生成数据表达函数不确定性；通用martingale posterior并非新想法。本轮阅读原文相关机制，尚未执行其代码。
3. [From Drift to Coherence](https://proceedings.mlr.press/v306/kim26ah.html)，ICML2026。已有预测重采样和一致性微调。仅把同类正则用到时间序列，不足以成立创新。
4. [On the Conditioning Consistency Gap in Conditional Neural Processes](https://arxiv.org/abs/2604.19312)，2026-04-21；arXiv及PDF标注TMLR2026，OpenReview页面本轮被浏览器验证挡住，录用状态未独立复核。文中比较factorized joint的条件化与重新编码，给出特定均值聚合条件下的界。这个量不能直接等同于对未来证据积分的tower误差，也不覆盖所有神经世界模型。
5. [Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html)，AISTATS2021，[公开实现](https://github.com/EmbodiedVision/explorethecontext)。已有共享上下文条件动力学、潜在系统分布和主动校准。源码保持同一latent draw跨时间、跨候选控制，故“现有模型都无法共享同一系统样本”已不成立。采用修正模型选择错误的分支f0baf6262523d9c5abaf8c0381b4db97818f7c34；34个源码/许可/配置接口文件Git blob校验。24个发布检查点完成受限加载及前向接口验证，尚无新校准质量或belief-update实验结果。

原文7份已在服务器保存，正常TLS校验及SHA256见outputs/literature/manifest_requests_attempt2.json。首轮urllib证书错误原样保留；Requests CA路径恢复后全部取得，未关闭证书验证。未采用博客、AI摘要或MDPI作为论据。旧方法足够强并不自动否定所有新的任务需求，但本轮没有发现足以推荐的新机制。
