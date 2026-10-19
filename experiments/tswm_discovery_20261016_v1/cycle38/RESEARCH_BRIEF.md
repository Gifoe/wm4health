# C38：通用状态学习框架，补查多模态与异质条件这一盲区

2026-10-17，检索前冻结。C37原始学习作业仍在执行，禁止另起训练、修改其源码或提前从partial指标裁决。整体ACTIVE_UNACHIEVED，接受最终方向0。本轮先进行独立原文研究；root同时完成C37预先约定分析。

用户要求的层次是可复用时间序列world-model框架，以WorldTS为参照；数据集是证据载体，允许每数据拟合，不强制跨域零样本单checkpoint、因果/物理状态、动作、Loop、LLM或online adaptation。此前检索和实训高度偏numeric，这次补查images/text/heterogeneous side information，但不把必须用多模态设为新成功标准。

## 冻结三个问题

1. 最近六个月的WorldTS及可执行相近框架，实际如何把异质数值/文本/图像条件用于状态形成和转移学习？有什么具体学习目标、表示接口或可复用能力差异，已经被真实训练和强控制支持？哪些所谓不足仅是接口/预处理/原文没有承诺的性质？
2. 在有限样本/计算下，是否有一个重要的**状态/转移学习**残差，未被最近完整方法和普通direct/fusion/conditional state-space解释？优先从作者真实实验的失败和限制、而非宏大原则或经典模型理论表达能力出发。已知组件组合允许成为贡献，但必须有可区分的交互与实际可证伪收益。
3. 哪个具体最小机制与单5090实验可以推进一个候选？必须给出最近完整先例、差异在哪里、最强普通对照、合法公开输入/任务/代码，以及什么结果会关闭它。若没有足够证据可以无候选；但不能只罗列论文或因为Kalman/通用RNN理论可表达就自动否定所有有限预算方法。

## 三个独立clean-context视角

- mainstream：真实框架、目标/接口、强基线与论文证据；优先公开源码与真实可复用训练底座，不只预测分数表。
- critic/methodology：查新、任务/信息公平、implementation/undertraining/ordinary repair、哪些能力与用户要求相符，哪些是人为加强门槛。
- constructive/adjacent：提出至多一个有新机制区别的有限预算候选；承认已有组件先例，不要求纯新原理，不把旧机制改名。给一个小而决定性的可失败实训，若无可执行证据就明确未建立。

主窗口2026-04-17..10-17，必要早期基础工作单列。仅依赖原论文全文、作者代码/card、官方proceedings和数据提供方；身份与具体主张分别核实，未核实来源不使用。先宽后窄，交付来源位置和比较综合而非摘要堆积。

## 既有证据边界，禁止重复

WorldTS公共repo decisionintelligence/WorldTS pin269426ea6217d7152a244da0e209d922acd46d4f早已实际执行；不是“没找到code”。引用WorldTS时不能称所有future covariates是未来truth：NP/PJM/BE/FR/DE部分是已发行预报。原文多模态的任务合同须逐项核实，不能把所有image/text都称未来可见或都称泄漏。

已关闭旧NP/PJM readout/epoch1/Loop调参、NESO persistent updater。C33普通global readout解释部分native误差；C35 common E/D、条件latent保护方案两真实任务未过screen，不再调那一具体配置。C34first-hit proposal与conditional-channel MLE目标等价；不要改名复活。

更早普通Kalman/joint covariance/Bayes/IS/NP、系统辨识/forced-input离散化、semigroup toy、missingness policy toy与response-only目标都已有具体已知控制或负证据。不能把这些宽泛主题再包装为首次。它们不证明所有有限预算组合已解决；若确有新的能力交互可提出，但要具体。

C36 observer-only外部预报校准+frozen generator+integration与普通两阶段目标等价，关闭该提案。KITE/Chronos2/Cross-Unet/DSSMF/MMFE/UKF-L/PBC为已知先例。C37正跑wind/solar正式12fits，quantile+rank/CAL/HGB与native KITE L1/L2/10vs50steps，TEST未评分；本轮不据其partial结果发明机制。不强制围绕forecast uncertainty继续。

每个视角仅写自己命名note到cycle38，不修改其他文件，不启动服务器作业、不批量下载权重/数据、不触碰cycle6 secret/W&B launcher、不发布内容。root合并独立复核后才定新实验。
