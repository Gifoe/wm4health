# 异质条件如何进入时间序列世界模型：C38证据综合

## 摘要

UTC 2026-10-16T23:03:29.601710+00:00。C38为三个独立clean-context视角的定向文献补查及root代码/原文复核；新训练0、新方法0、最终接受方向0，整体ACTIVE_UNACHIEVED。C37原作业另行处理，未使用其partial指标构思本轮故事。

本任务以WorldTS为贡献层次参照，寻找可复用时间序列world-model框架。数据集用于检验机制，不以某站点或榜单涨点作为终点；允许每数据拟合，不额外要求action、causal identification、物理真状态、共享zero-shot checkpoint、多模态、Loop或LLM。已知组件的新交互可以有贡献，不要求从未出现的数学算子。

本次补查此前numeric偏重的盲区。完整先例已覆盖异质输入、latent future prediction、残差监督、内部routing及条件化recurrence。仍有作者实测的latent/readout效用失配与上下文负迁移，但未隔离为一个超过强普通修复的新状态学习机制。不能把“文献有相似模块”直接等同“所有组合都无价值”，也不能把负迁移直接升级为新框架必要性。

## 引言：冻结的三个问题

RQ1：真实框架怎样用异质条件形成和预测状态，有哪些已训练的具体能力？RQ2：哪些有限数据/计算残差尚未被普通direct/fusion/conditional SSM解释？RQ3：有没有一个与最近完整先例不同、单5090可证伪的最小机制？原始合同见[RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)。

## 方法与核验

主窗口2026-04-17..10-17，必要更早先例标明。mainstream、critic/methodology、constructive/adjacent三视角各自先宽后窄检索，不读sibling notes；root只补充既有C35排重事实。分别见三个独立note。身份与主张分开核查，采用原论文、官方proceedings、作者代码/数据；无法访问全文的FACTS/CFA不提供未核分数。root归档14个原文HTML和3个WorldTS固定源码，共11680761 bytes，正常TLS，17个SHA验证；不是模型/数据下载或论文复现。来源manifest为[SOURCE_RECEIPTS.json](SOURCE_RECEIPTS.json)。

WorldTS固定commit `269426ea6217d7152a244da0e209d922acd46d4f`。root重新读取文本embedding函数和图像输入函数，另通过arXiv原文核Pythia具体负例与CAMOS定理范围。public源码不执行，仅静态审计；论文作者实测不混称本项目重跑。

## 分类：条件改变了哪一段

| 位置 | 完整先例及交叉比较 | 本轮仍需区分的普通解释 |
|---|---|---|
| 信息进入接口与内部路径 | UniCA统一异质输入；Aurora/ AIR让模态改变attention或routing；WorldTS把历史文本/图像送入predictor | encoder质量、时间对齐、已有cross-attention/FiLM，而非先假设新transition缺失 |
| 预测状态与学习责任 | LatentTSF固定AE目标，WorldTS学习条件未来latent；REST-TS给文本专属残差目标，Pythia给context参考引导再单独读出 | direct+额外监督、兼容decoder、numerical fallback，不以单独latent距离裁决 |
| 时间传播、缺失与联合状态 | BeliefGraph-JEPA的driver持续状态/rollout；CAMOS availability控制耦合transition；NOAH共享随机latent连接多头 | timestamp-aware GRU/Transformer/selective SSM及既有条件生成器；临床任务不直接当普通forecast leaderboard |

接口支路见[UniCA](https://github.com/hanlu-nju/UniCA)、[Aurora](https://arxiv.org/html/2509.22295v3)、[AIR](https://arxiv.org/html/2512.10229v1)。学习支路见[LatentTSF](https://arxiv.org/html/2602.00297v2)、[WorldTS](https://arxiv.org/html/2609.31162v1)、[REST-TS](https://arxiv.org/html/2606.19413v1)。时间支路见[BeliefGraph-JEPA](https://arxiv.org/html/2610.05409v1)、[CAMOS](https://arxiv.org/html/2609.39484v1)、[NOAH](https://arxiv.org/html/2609.09140v1)。这些具体差异应比较，不能用“都是JEPA”压平贡献；空白接口也不自动等于research gap。

## 分支一：WorldTS的真实合同

固定文本loader的get_all_embeddings直接lookup token embeddings，再avg/max/min pooling，没有contextual Transformer forward。因此在截断后token多重集相同的条件下，句内顺序信息丢失；这是代码推论，未声称真实语料错误率已经实测。直接换contextual encoder属于普通修复，不是本项目新世界模型。[固定loader](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/data_provider/data_loader.py#L129)。

文本runner不传future_text/future_marks/x_future/prior；图像model要求satellite与past_target同长度，未来只允许feat_dynamic_real中的数值段。原论文部分数值任务本就用已发行预报。故不能把所有条件一概叫future truth/leakage；具体数据的issue-time仍需按合同核查。[图像输入](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/MMSP-bench/models/adapter/worldts_image/model.py#L126)。

原文App.A.4已让one-stage使用N1+N2 epochs；不能笼统指控其direct少训半程。w/oCovariates删除条件集合，不是单独图像消融；单seed及继承的baseline分数限制更强归因，但不抹去其框架贡献。[WorldTS原文](https://arxiv.org/html/2609.31162v1)。

## 分支二：具体残差，尚未定位病因

Pythia的固定Small结果中，无reference版本latent误差0.02236低于0.02291，MASE却0.7086差于0.6981；context改善29组平均latent，而相对numerical-only只在17组改善MASE、18组改善WSQL、15组两者都改善。这支持有限方案下latent距离与forecast utility失配，未证明唯一病因是geometry。其数值reference、context residual和独立readout已是完整近邻，代码/模型仍标待发布；不把重建小版称原作者复现。[Pythia §3、§4.4、F.4](https://arxiv.org/html/2610.05240v1)。

REST-TS已对文本提供数值主干无法代劳的残差监督，Pythia又分离context/reference控制。因此新的辅助loss必须击败普通direct+auxiliary及同信息残差分支；不能仅因文本梯度变大就声称学到语义状态。C35已关闭的具体protected residual配置保持关闭。

## 分支三：条件传播不能仅靠换名称立项

CAMOS已经让availability乘积进入transition；其Theorem5只覆盖availability-independent单线性层，明确不涵盖带mask的selective Mamba，也不排除非线性readout/fusion。因此它不能用来证明所有普通fusion不可能；却足以排除“首次让条件改变transition”的宽泛主张。[CAMOS §4.3](https://arxiv.org/html/2609.39484v1)。BeliefGraph-JEPA已有持续driver状态和rollout；其logged actions与NOAH临床联合状态均不成为用户必须采用动作/临床任务的理由。

## 跨分支综合：矛盾取决于条件

[Semantics or Structure](https://arxiv.org/html/2608.22321v1)在其Time-MMD设置观察语义替换弱效应；[MUSE](https://arxiv.org/html/2609.15087v1)则在更多任务发现错语义/错时间退化。两者不支持“文本总无用”或“正确文本总有用”；任务、时间配对、编码与numeric side channel不同。前者审计的部分numeric prior不由WorldTS runner读取，不能直接迁移归因。

仍未排除的两项问题是：A，在固定预算下，怎样使条件更新保留兼容readout可用的预测信息；B，当numerical已很强时，稀少语义信息怎样获得稳定学习。A优先，但普通观测loss/decoder适配可能已够；B须先用cross-fitted residual probe对dates/entity/missingness/placebo确认文本有独立信息。它们目前是研究线索，不是两项已批准idea。

## 可证伪下一步与是否执行

独立notes给出了编码接口×学习范式的2×2小诊断，加入direct+ordinary numerical auxiliary、conditional GRU/SSM、numeric-only与错时文本，并给对称有限1×/2×预算。若contextual前端或普通监督抹去差距，就关闭新状态目标的动机；若只见敏感性或latent变化而无预测收益，也关闭。即便原WorldTS胜出，仍只是已知框架证据，不自动产生本项目新方法。

root本轮**不启动这批额外fits**：当前它只会排查已知前端限制，尚无可区分的新交互。先完整解释C37已冻结普通控制，并避免再循环C35；确有新机制或独立信息证据后再冻结新学习实验。这个停止低信息实验的决定不等于任务完成。

## 结论：逐项回答

RQ1：已有完整、真实的异质条件预测状态框架；WorldTS可执行，其他路线需区分公开资源。RQ2：存在效用错位/负迁移线索，但尚未排除普通解释；不能据此强行立项。RQ3：本轮没有已成立的新机制候选；保留A/B及普通修复falsifier，目标继续未完成。已知组件组合仍可成为框架贡献，决定标准是明确交互和证据，不是纯新算子或无限预算不可能性。

## 参考与证据账本

每个原文与精确章节、公开代码pin、访问失败边界见三个独立notes。主文只使用其已核部分；SOURCE_RECEIPTS给本轮17个固定正文/源码SHA。报告未把作者分数当自跑结果，未调用外部语言生成API，未下载信号数据/权重，未对外发布。
