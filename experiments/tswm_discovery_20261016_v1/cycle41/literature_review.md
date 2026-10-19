# C41 文献综合：状态的贡献必须落实到可复用能力

## 摘要与问题

本轮问：现有框架究竟证明了什么状态收益；怎样区分状态信息、转移近似和推断误差；哪种具体交互值得单卡实测。当前最值得开发的候选是有同步观测锚时，观测推断接口与动态模块的交叉复用。其创新尚未证实。

## 方法与证据分层

deep-research 的三个干净上下文分别检索主流、方法学反证、相邻建设性机制；只共享冻结 brief，不读 sibling notes，不访问服务器。主进程做源码/实验并在最后串行综合。代理原文和可核验直接来源保留在 mainstream_state41.md、critic_state41.md、constructive_state41.md。主进程另核对 PMLR/NeurIPS 原始页面和 MDN-VJEPA §8.6。论文事实、实验事实、候选假设分别陈述，未检索到完全相同组合不作为新颖证明。

## 1. 学状态、学未来 law 与学推断器是三件事

[WorldTS](https://arxiv.org/html/2609.31162v1) 与 [LatentTSF](https://proceedings.mlr.press/v306/yang26y.html) 支持 latent training 的可迁移预测配方，但 encoder 形状或预测改善本身不证明新增了关于隐藏过程的信息；[PSIM](https://proceedings.mlr.press/v48/sun16.html) 则已把未来统计量和递归过滤更新组成完整方法。因此“用未来监督一个可更新状态”不是新机制。状态坐标距离也不能跨任意 latent gauge 比较，必须以相同 observable future law 或充分性诊断作证据。

[MDN-VJEPA](https://arxiv.org/html/2610.13956v1) 明确把未来分布/轨迹/组合机制隔离于 encoder 学习，而 [LIFTS](https://arxiv.org/html/2610.03916v1) 实际联合学习生成与推断，覆盖了更完整的链条。前者 §8.6 的 learned-encoder 实证缺口不能被当作后者也没有推断能力；后者 synthetic calibration 无系统增益也不证明状态学习无用。它们共同要求把信息、模型近似和推断误差分开归因。

## 2. 复用两个接口的交互仍需要强完整对照

[NCDSSM](https://proceedings.mlr.press/v202/ansari23a.html) 已用 auxiliary observation variables 分离 recognition 与 dynamics，[modular NRI](https://papers.neurips.cc/paper_files/paper/2019/hash/b294504229c668e750dfcc4ea9617f0a-Abstract.html) 已用可重组动态模块做系统关系推断。两者不是互相覆盖：前者着重不规则/部分观测推断，后者着重动态图组成。候选需要验证有限数据下同时复用两个接口，是否在未见 view–mechanism 组合产生独立交互收益。普通 joint hierarchical SSM 可表达同一 factorization，必须给它同样锚、噪声模型和参数估计机会。若它吸收全部收益，不能把相同模型重新命名。

## 3. 界限与另一条备选

连接的训练组合图不保证目标 observable law 可确定；本轮精确置换反例和普通 Bayesian mixture 直接给出边界及正控。这个信息问题不是神经网络训练失败，也不要求恢复唯一物理状态。备选的粗路径—多层查询方向受 [mr-Diff](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d64740dd69bcc90ba225a182984b81ba-Abstract-Conference.html) 与 [generative MLMC](https://arxiv.org/html/2409.15511v2) 约束，现有粗到细生成和多层估计本身都不能冒充新算子；暂不与主线同时扩训。

## 结论：逐项回答研究问题

RQ1：已有完整方法支持预测归纳偏置、概率未来 law 和多任务推断；这些收益不能相互替代。RQ2：统一信息接口/精确 posterior/转移控制和 gauge-free observable metrics 可以分开归因，本轮已建立可执行量尺。RQ3：有锚的观测接口×动态模块复用有一个可证伪的有限数据交互问题；尚未有新方法效果，下一步必须让 ordinary joint model 先参与，而非先发明正则。
