# C38 主流框架复核：异质条件已进入预测状态，尚未建立可立项的新学习残差

检索日期：2026-10-17。独立视角：mainstream。研究问题以同目录 RESEARCH_BRIEF.md 为准。只编辑本文件；未读取 sibling notes；未训练、启动服务或下载数据/权重。下文“实测”均明确指论文作者报告，不能替代本项目运行作者实现的结果。

## 摘要

本次补查数值之外的文本、图像和异质条件，重点核对潜状态目标、条件进入的位置、训练拆分及公开实现。WorldTS、Pythia 与较早的 UniCA/Aurora 已构成明确的连续先例：异质输入可作为统一数值表示、注意力/路由控制、未来潜状态修正或生成条件。文本梯度被数值主干压制也已有 REST-TS；驱动的延迟、持续与结构化 rollout 已有 BeliefGraph-JEPA。可定位的未解决现象是：潜空间匹配更好不必然对应读出更好，语义利用随训练方案和任务而变化，完整模型有真实负迁移。这些现象值得研究，但目前不能从它们推导出一个避开普通融合、辅助监督、读出适配和既有残差监督的最小新机制。本视角接受候选数为 0；给出一项可失败的低预算诊断合同，供 root 决定是否值得继续准备，而非要求本轮启动训练。

## 1. 问题、检索与范围

RQ1：最近完整框架如何把异质条件用于状态形成/预测，哪些能力由训练和对照支持？RQ2：现有作者实测中的状态/转移学习残差是什么，能否排除普通解释？RQ3：能否提出与最近完整先例有区别、可在单 5090 验证的最小机制？

主窗口为 2026-04-17 至 2026-10-17。先用 `WorldTS / time series world model / multimodal covariates / latent dynamics / JEPA` 搜索，再补 `negative transfer / asynchronous / text collapse / semantics / decoder`。使用 arXiv 全文、ICLR proceedings、论文链接的作者 GitHub 及公开 API；二手解读只作为检索线索，不承担结论。论文身份以标题、作者和 arXiv/官方 proceedings 双向核对，主张以具体章节/表/代码行核对。独立视角分工由 root 已安排，本视角没有再派生代理。

时间须分层：WorldTS（9 月）、Pythia（10 月）、REST-TS（6 月）、Semantics or Structure（8 月）、MUSE（9 月）、BeliefGraph-JEPA（10 月）在主窗口内；UniCA、Aurora、AIR、LatentTSF 的最初公开均早于主窗口，作为必要旧先例，不能按“2026 会议论文”冒充最近六个月首次提出。LatentTSF v2 在 5 月更新，仍明确保留其早期优先权。[1–10]

本报告不把跨域零样本、单 checkpoint、动作控制、物理真状态或 online adaptation 加成用户必须满足的新门槛；每数据拟合的可复用训练框架仍在范围内。也不以通用 RNN/Kalman 的理论表达能力直接否决有限预算机制。

## 2. 分类轴：条件改变了哪一段学习

下表按主要学习位置分类，而非把“多模态”当成统一机制。跨列的组件组合已有先例，空格不等于研究缺口。

| 主要位置 | 可复用能力与已知例子 | 尚不能从接口名称推断的性质 |
|---|---|---|
| 数值化/统一输入和融合 | UniCA 将异质条件变为统一表示，接冻结 TSFM；WorldTS 使用其图像前端 | “统一表示”不证明学到可识别真实系统状态 |
| 条件控制内部信息路径 | Aurora 的模态引导注意力；AIR 的文本控制路由 | 文本有通道不证明预报真的使用语义 |
| 未来潜状态监督及冻结读出 | LatentTSF、WorldTS、Pythia | 潜误差下降不保证任何读出/任务均提升 |
| 给弱模态独占学习信号 | REST-TS 的数值残差结构监督；Pythia 的 reference-guided correction | 提高模态响应不自动保证泛化收益 |
| 显式驱动效应的状态演化 | BeliefGraph-JEPA 的持续状态、future rollout 和结构路由 | logged future actions 的回顾性预测不等于因果干预识别 |

WorldTS 相对 UniCA 的核心差别是训练目标和预测状态的角色，而不是首次允许文本/图像作为输入；Pythia 相对 WorldTS 又加入稳定的预训练数值空间与条件修正目标。AIR/Aurora 则已经排除了“以前文本只能加到输入、不能改变内部计算”的宽泛新颖性说法。[WorldTS §4](https://arxiv.org/html/2609.31162v1#S4)、[UniCA 官方说明](https://github.com/hanlu-nju/UniCA)、[Pythia §3](https://arxiv.org/html/2610.05240v1#S3)、[AIR §3](https://arxiv.org/html/2512.10229v1#S3)、[Aurora §3](https://arxiv.org/html/2509.22295v3#S3)。

## 3. WorldTS：真实合同与可支持的结论

### 3.1 学习目标和状态接口

WorldTS 用共享 causal-patch 编码器编码历史和训练时未来观测；数值条件独立编码，文本/图像经前端和 adapter 进入 predictor。Stage 1 学习条件化未来状态，并使用匹配、相似度及方差/协方差正则；Stage 2 冻结此前模块，训练共享 decoder 的预测与重建。其可复用对象是条件潜状态预测框架，不要求逐步物理模拟。[WorldTS §4、Appendix A.3–A.4](https://arxiv.org/html/2609.31162v1#S4)。

相比只把观测送进回归头，作者 Table 4 的 one-stage MAE、pointwise encoder/decoder 和 no-covariate 对照支持这个组合有用；但该表并未独立控制每个预算、初始化和前端选择，不能把组合收益全部归因于某个“世界状态”概念。[WorldTS Table 4](https://arxiv.org/html/2609.31162v1#S5.T4)。相较 LatentTSF 的先重建并冻结 E/D，WorldTS 在潜动力学习后才训练读出，这才是训练合同的差异。[LatentTSF §3.3–3.5](https://arxiv.org/html/2602.00297v2#S3.SS3)。

### 3.2 固定提交源码核验，不能混淆三个分支

固定提交：`269426ea6217d7152a244da0e209d922acd46d4f`。以下均为这次直接读取公开原文件的证据，不是把 repo 名称当作实现证明。

| 分支 | 本次核对的具体输入合同 | 原代码定位 |
|---|---|---|
| Time-MMD 文本 | `get_text_embeddings` 只切 `s_end-seq_len:s_end` 历史文本。runner 不传未来文本、未来 marks、未来 x；预测路径不混 `prior_history_avg`。训练 future_target 仅作监督 | [data_loader.py L223–233](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/data_provider/data_loader.py#L223)、[exp_worldts_forecasting.py L126–149, L193–200](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/exp/exp_worldts_forecasting.py#L126) |
| 文本前端 | frozen GPT-2 **input embedding lookup** 后做 masked avg/max/min pooling，默认脚本 avg，截断 256 tokens；不是让 GPT-2 Transformer 对全文做上下文化推理 | [data_loader.py L130–184](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/data_provider/data_loader.py#L130) |
| MMSP 图像 | `satellite_data` 的前两维必须和历史 target 对齐；历史图像通过 CNN/homogenizer 变成 image tokens。未来数值条件来自另一条 `feat_dynamic_real` 切片 | [worldts_image/model.py L126–170](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/MMSP-bench/models/adapter/worldts_image/model.py#L126)、[image_homogenizer.py](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/MMSP-bench/models/adapter/worldts_image/image_homogenizer.py) |
| MMSP 数据构造 | 数值动态条件加载 `nwp/nwp.csv`；卫星与电站序列按时刻对齐。该代码事实不能单独证明所有 NWP 值的发行 vintage；也不能反向断言未来图像被输入 | [ts3m.py L195–251](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/MMSP-bench/data/multi_modal/ts3m.py#L195) |
| 两阶段训练 | decoder 同时处理 `h_pred` 和 `h_target`；第二阶段 representation 经 no_grad/detach，损失是预测误差加权重建误差 | [exp_worldts_forecasting.py L165–190](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/exp/exp_worldts_forecasting.py#L165) |
| 潜正则实现 | alignment 可选，脚本用 MAE；cosine 与 variance/covariance regularization。模块叫 `SIGRegLoss`，实现是方差阈值与 off-diagonal covariance，不应仅凭类名制造科学缺陷 | [latent_state_loss.py](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/models/worldts/losses/latent_state_loss.py)、[sigreg_loss.py](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/models/worldts/losses/sigreg_loss.py) |

由代码直接可推得：平均词嵌入对保留相同 token multiset 的词序变化不敏感。这是这一前端的信息瓶颈；换 contextual embedding 是必须纳入的普通修复对照，而不是新状态学习贡献。类似地，统一数据适配、补 missing masks 或 timestamps，除非改变了可检验学习能力，否则只是接口工作。UniCA 已提供冻结 TSFM 加异质 adapter 的普通强对照。[UniCA module.py L70–72、L227–264、L422–459](https://github.com/hanlu-nju/UniCA/blob/727d3670e9b7993472b19f47e42ac0d75142b127/models/adapter/unica/module.py#L70)。

论文说明 NP/PJM/BE/FR/DE 的若干外生量本来就是负载/发电预报，因此未来协变量不是统一的未来真值；MMSP 也须把历史卫星与 NWP 分开。各分支需审计发行时间，不能以一个总称宣判泄漏。[WorldTS Appendix A.1](https://arxiv.org/html/2609.31162v1#A1.SS1)。

### 3.3 证据能走到哪里

WorldTS 作者报告 21 datasets 和三个代表性消融数据；主结果许多 baseline 数值继承 DAG/KITE、VoT、UniCA 原协议，而非所有基线在当前实现中统一重训。公开脚本说明可复用训练路线存在，不能据此写成“已经独立复现所有 headline”。该论文也没有承诺 irregular sampling、识别物理状态或跨域单模型，缺这些性质不能成为对它失约的指控。[WorldTS §5.1–5.4、Appendix C](https://arxiv.org/html/2609.31162v1#S5)。

## 4. 最近完整先例已经挡住哪些宽泛提案

### 4.1 Pythia：稳定数值参照、上下文修正和独立读出已有整体实现描述

Pythia 用冻结 Chronos-2 数值空间，entity/event/covariate tokens 修正数值未来状态，Stage 1 是 direction、scale 和 detached reference 引导，随后冻结表示适配 quantile readout。matched direct/joint/numeric-only 与 source removal 已做；代码、模型仍写 To be released，不能说执行过作者源码。其 8 devices 和非公开 Industrial 训练部分也不等于单 5090 可完整复现。[Pythia §3、§4.3、Appendix A–B](https://arxiv.org/html/2610.05240v1#S3)。

这一完整路线已覆盖“保留数值主干、让上下文只补残差、冻结世界状态再读出”；不能因为 WorldTS 没有该保护就重提。与 REST-TS 的观测残差独占监督相比，它把弱分支监督问题移到未来潜空间；但 reference objective 只是平滑梯度加权，原文明确不保证 corrected error 不大于 numerical error。[Pythia Eq.4](https://arxiv.org/html/2610.05240v1#S3.E4)、[REST-TS §3](https://arxiv.org/html/2606.19413v1#S3)。冻结 brief 与 root 补充已确认 C35 的具体 protected-latent 配置筛选失败，本报告不把它复活。

### 4.2 语义梯度和驱动状态也有近邻

REST-TS 不只讲 text collapse：其 EMA 数值预报定义 residual，再将 trend/event 的专属目标交给文本 heads。因此“数值主干抢走梯度，应给文本不可绕过的目标”已是完整训练先例；作者还明确列出 frozen GPT-2、残差缩小时监督变弱、单 description 与频段分解假设的限制。[REST-TS §3、Appendix D](https://arxiv.org/html/2606.19413v1#A4)。AIR 的文本调节内部路径又说明 controller 而非简单附加输入也不是新概念。[AIR §3–5](https://arxiv.org/html/2512.10229v1#S3)。

BeliefGraph-JEPA 将不同 driver 的延迟/持续效应分成状态，未来驱动 rollout 后经图路由，direct tokens 仅走受限 residual；作者比较 flat、shared-state 和同类 residual 控制，承认图路由收益依数据而异。它堵住“静态 side information 应驱动持续状态，所以首次做显式条件转移”的宽泛主张。其 future actions 为 logged actions，不能外推因果识别；全文未找到作者代码链接，本次不称其为已执行底座。[BeliefGraph-JEPA §3–5.3](https://arxiv.org/html/2610.05409v1#S5)。

## 5. 作者实测留下的残差，及普通解释尚未被排除之处

### 5.1 潜匹配与观测效用的错位是真实线索，尚不是新机制

Pythia Appendix F.4 的 no-reference 版本 corrected latent error 更低（0.02236 vs 0.02291），但 forecasting 更差；这不是“作者没测”的假缺点。它说明跨训练方案比较时，潜匹配改善不能单独充当读出效用证明。更关键的是，该结果同时伴随 numerical estimate 质量变化；不能单凭它锁定“latent geometry 错了”，也不能忽略兼容读出、训练优化和预训练先验等解释。[Pythia Appendix F.4](https://arxiv.org/html/2610.05240v1#A6.SS4)。

这个现象对 WorldTS 的启示是测完整 state-to-forecast 链条，而非继续堆 latent losses。直接使用 decoder-aware latent distance 若等价于 decoded forecasting loss，就只是把普通观测目标换写法；若采用 Jacobian/低秩近似，有限预算收益仍可能成立，但必须证明它相对同信息 direct loss、同预算 auxiliary loss 和普通冻结 readout 有可区分作用。本次没有检索到足够证据将其中一个具体版本升级为候选。LatentTSF 已比较 observation-space perceptual loss 和 latent objectives，不能默认这些组合无人测过。[LatentTSF §5.3–5.5](https://arxiv.org/html/2602.00297v2#S5.SS3)。

### 5.2 语义利用不是永远不存在，也不是永远可靠

Semantics or Structure 在 Time-MMD 对 Aurora/TaTS/MM-TSFlib 做固定数值输入的文本替换，语义扰动对平均误差影响很小，并将后两者部分 lift 归因到同路的 numeric prior。它不能直接否定不读该 prior 的 WorldTS。相反，MUSE 对 VoT/GPT4MTS/TESS/TaTS 发现错语义或错时间会降低表现，提示数据、预处理与模型合同会改变结论；两文并不支持“文本对所有时序任务无用”。[语义审计 §3–5](https://arxiv.org/html/2608.22321v1#S3)、[MUSE §4.2.2](https://arxiv.org/html/2609.15087v1#S4.SS2.SSS2)。

这组冲突也有可查代码边界：审计 repo 的 README 与 arXiv 标题存在版本措辞差异，但论文直接链接该作者 repo；其 runner 指向官方 DecisionIntelligence/Aurora，README 下载示例却写 AutonLab/Aurora。本文使用作者论文的报告作为条件性证据，不把 README 的错误示例当作已核对全部权重 provenance；更不能由它推翻全部实验。[作者审计仓库](https://github.com/birla-ai-labs/semantics-or-structure)、[runner L69–74](https://github.com/birla-ai-labs/semantics-or-structure/blob/e1abb4a67339c39d4cf138dd72f739d9c5c55daa/code/runners/aurora_runner.py#L69)。

Pythia 又表明语义敏感性随容量和训练 recipe 改变；matched numerical-only 比较仍有多组负迁移。**可研究的是“什么训练导致语义对未来状态的可迁移贡献”，不是“让预测对文本有反应”本身。** 强迫响应也可能放大噪声，门控、late fusion 和 numeric-only fallback 必须先排除。[Pythia §4.4、Appendix G](https://arxiv.org/html/2610.05240v1#S4.SS4)。

### 5.3 为什么本轮不把异步和长效记忆直接立项

WorldTS Appendix C 主动限定 regularly sampled、per-dataset training；REST-TS 的多源 streaming 也是作者列出的限制。它们能形成任务动机，却没有证明既有 ordinary conditional state-space 在同输入、同预算下失败。异步 adapter 或新数据合同先解决的是可接入性；只有新的异质证据交互在 finite-data/compute 下稳定胜过 timestamp-aware GRU/Transformer、conditional SSM、direct/fusion，才成为状态学习贡献。BeliefGraph-JEPA 已覆盖持续效应和 future rollout，又提高了这条路线的最近比较门槛。[WorldTS Appendix C](https://arxiv.org/html/2609.31162v1#A3)、[REST-TS Appendix D](https://arxiv.org/html/2606.19413v1#A4)、[BeliefGraph-JEPA §5](https://arxiv.org/html/2610.05409v1#S5)。

## 6. RQ3：无已成立候选；可准备的单卡诊断合同

**本视角没有一个已经获得足够证据、可以命名并接受的新最小机制。** 原因不是经典网络在理论上可以表达，而是最近完整先例已经覆盖当前可具体写出的 residual、controller、latent supervision 与 driver rollout，作者负例尚不能排除普通训练/读出/前端解释。也不据 C37 partial 输出另造故事。

### 6.1 最强的两项尚未排除残差，以及如何被普通对照关闭

零候选本身不算本轮研究成功。以下两项是应保留的实证线索，优先级 A 高于 B；它们尚未满足“现有方法没有解决”的因果归因条件，但比“多模态没利用好”具体。

| 残差 | 作者已经观察到什么 | 最小普通对照与可失败判据 | 仍可能值得研究的交互，而非当前已获新颖性 |
|---|---|---|---|
| A：条件潜状态改进与兼容读出效用不一致 | Pythia F.4 的较低 corrected latent error 未带来更好 forecast；F.2 用 numerical predictions 替换 corrected states 后，MASE 和 WSQL 的方向也不一致。这是测过的接口学习效用残差，不是臆测模型缺少真状态 | 固定同一个 E、数值 predictor 与已训练 readout，分别输入 numerical/corrected states；再给两者各训同预算、同初始化 readout。加入只训练 readout、direct MAE/quantile loss、普通 latent+observation auxiliary。若差距由读出重训或普通目标消除，关闭新状态机制解释 | 在稳定数值空间中，条件增量的潜目标如何与之后允许的读出适配共同组织；要求同一机制在 latent 和 output 两层都有可检验交互，不能只加 decoder-aware loss 改名 |
| B：数值可预测性很强时，稀少语义条件的独立状态信息未被稳定保留 | REST-TS 作者明确指出数值残差收缩会削弱文本独占目标；Pythia 的 masked-only 和部分容量方案语义扰动响应很弱。这里假说是训练分配使有用的弱条件被压制；当前还未证明这些具体任务确有未用信息 | 先用 cross-fitted numeric-only 残差做文本预测 probe，并与同域错时间文本、日期/实体/缺失标记的 placebo 比较；随后同前端比较普通 text-residual MLP/REST 式目标、direct cross-attention、numeric extra loss 与 latent route。若 probe 不超 placebo，关闭“有用条件未学”；若普通残差头追平，关闭新 latent 学习必要性 | rare-event 学习窗口和状态监督/冻结时机的交互在有限预算下可能重要；必须先有真实条件信息，再证明共享训练信号不能替代。把 C35 protected-latent 配置换名不算新交互 |

上述 A 由 [Pythia Appendix F.2/F.4](https://arxiv.org/html/2610.05240v1) 定位，B 由 [REST-TS Appendix D](https://arxiv.org/html/2606.19413v1) 和 [Pythia Appendix G](https://arxiv.org/html/2610.05240v1) 定位。反向解释同样保留：A 可能只是 decoder adaptation；B 可能是数据根本没有条件增益或 masked objective 已够好。两者都不能因“经典模型可表达”自动判死，也不能因论文存在负例自动判活。

### 6.2 单 5090 的准备合同

如果 root 决定继续准备，优先做“state-supervision 是否提供超过普通融合的交互收益”的小诊断，而不是先增加新模块。以下是预注册式建议，**本轮未执行，也不是 accepted candidate**：

1. 任务与合法输入：选 WorldTS 原 Time-MMD 的 Health 与 Climate，保持官方 H=12 和历史 fact 文本合同、时间切分、同 target。不使用未来文本、未来 prior、oracle language。作者公开 [Time-MMD](https://github.com/AdityaLab/Time-MMD) 提供原材料，但 API 未列全库 LICENSE；因此只承诺使用公开研究材料，不宣称已核实全库开放许可，也不重新分发。若现有准备数据无法追溯报告发行时间，先停止此任务合同。
2. 公共训练底座：用上述 WorldTS 固定 pin，保留其明确的 no-prior runner。缓存文本表示之后，小模型训练在单 5090 的内存需求预期可行；这是工程估计，没有实测速率/显存，不能报预计训练小时。
3. 最强普通对照：原 WorldTS；相同 E/P/D 和前端、只用 observation MAE 的 direct/fusion；direct/fusion 加同预算 ordinary numerical auxiliary loss；numeric-only；以及把 GPT-2 lookup pooling 替换为同一个 contextual encoder 后，对 WorldTS 和 direct/fusion **同时**替换。此替换是普通修复，不归入新机制。
4. 控制预算：先冻结结构和小网格，用 validation 选择训练点，所有 arms 包括 step 0；报告实际 GPU time、更新次数、trainable parameters、early stopping。相同 update 数不等于相同 compute，必要时同时给 time-matched 与 exposure-matched 两列。筛选两 seed，未过筛不扩 seed；通过后才增加第三 seed 与另一任务，不拿单 seed 作为 accepted evidence。
5. 必须测的交互：分别保持历史数值、文本长度/缺失标记、任何 numeric side channel 不变，仅换成同域错时间文本、固定文本和原文；在训练时使用共享数据合同做数值/文本移除，而不是仅在测试时切掉一个分支。读取未来潜匹配误差、同一预测 latent 经兼容冻结 decoder 的误差、最终 forecast MAE/MSE。语义响应、latent improvement 和 forecast improvement 三者不能互相代替。
6. 关闭条件：若 contextual frontend 抹去原优势，问题归入前端；若普通 auxiliary/direct/fusion 在相同预算追平，缺口未建立；若只见 embedding/rank/语义敏感性变化而最终误差无改善，缺口未建立；若只对错误上下文更敏感而正确上下文没有受益，关闭“学会利用文本”的解释；若两任务都选择 step 0，关闭该具体训练配置。不得在失败后换汇总指标、任务或重新命名。

该诊断的价值是把“组合值得继续试”的有限预算主张变成可以被关闭的观察；它不要求纯新原理，但也不会把 ordinary repair 自动升级为论文贡献。

## 7. 对三个问题的明确回答

- **RQ1**：异质条件已经进入 latent predictor、状态修正、内部路由和生成条件。WorldTS 的两阶段训练及公开三分支是真实可复用底座；Pythia 更近、更完整，但发布状态与预算使它目前只能作纸面强边界，不能称已执行作者源码。多模态输入需逐分支核对，不能统一视为未来泄漏，也不能统一视为语义推理。
- **RQ2**：最具体的实证线索是 latent/readout 错位及 context 的 task/recipe-dependent utility；但目前没有一项被证实超过普通 direct/fusion、auxiliary loss、frontend replacement 与 conditional state-space 的解释力。长期效应、保护数值主干、残差监督本身已不是空白。
- **RQ3**：接受候选 0。暂不建议启动新训练来验证一个尚未定义的新方法；如需继续，只准备上节的失败可关闭诊断，并由 root 独立复核后决定。与 WorldTS 同贡献层级的框架仍可以由已知组件的新交互构成，但本次证据尚未给出那一交互。

## 参考文献与复核位置

[1] Yuhan Zhu, Xiangfei Qiu, Hanyin Cheng, et al. “WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting.” arXiv:2609.31162, 2026. [原文](https://arxiv.org/html/2609.31162v1)，核心定位 §4、Table 4、Appendix A/C。作者代码固定 pin 见 §3.2。

[2] Xilin Dai, Hongzhou Chen, Yifan Hu, et al. “Pythia: Toward Foundation World Models for Multimodal Time Series.” arXiv:2610.05240, 2026. [原文](https://arxiv.org/html/2610.05240v1)，核心定位 §3、§4.3–4.6、Appendix A/B/F/G；首页 Code/Model 为 To be released。

[3] Huu Hiep Nguyen, Minh Hoang Nguyen, Dung Nguyen, Hung Le. “Does Text Actually Help? Uncovering and Resolving Text Collapse in Multimodal Time Series Forecasting.” arXiv:2606.19413, 2026. [原文](https://arxiv.org/html/2606.19413v1)，核心定位 §3、Appendix D。未核实独立官方代码仓库，不写为可直接执行作者底座。

[4] Karthik Sridhar, Atharva Gupta, Nishant Pradhan, et al. “Semantics or Structure? Auditing Text Sensitivity in Multimodal Time-Series Forecasting.” arXiv:2608.22321, 2026. [原文](https://arxiv.org/html/2608.22321v1)，核心定位 §3–5、Appendix I；[作者仓库](https://github.com/birla-ai-labs/semantics-or-structure)，本次固定 `e1abb4a67339c39d4cf138dd72f739d9c5c55daa`。

[5] Peng Chen, Zhihao Zhuang, Hongzhou Chen, et al. “Beyond Numerical Time Series: A Unified Benchmark for Multimodal Forecasting with Heterogeneous Context.” arXiv:2609.15087, 2026. [原文](https://arxiv.org/html/2609.15087v1)，核心定位 Table 4/5、Appendix C。未把搜索中同名的 memory/unlearning MUSE 仓库误当该 benchmark。

[6] Yue Li, Kangqi Ni, Zhen Tan, Tianlong Chen. “BeliefGraph-JEPA: Structured Latent World Models for Action-Conditioned Time Series.” arXiv:2610.05409, 2026. [原文](https://arxiv.org/html/2610.05409v1)，核心定位 Eq.2/4/5、Table 2、§5.1；未核实作者代码，不称已独立复现。

[7] Lu Han, Yu Liu, Lan Li, et al. “UniCA: Unified Covariate Adaptation for Time Series Foundation Model.” ICLR, 2026；首次 arXiv:2506.22039 为 2025 年。身份由 [官方 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b5eb45a22ff33956c043dd271f244ea-Abstract-Conference.html) 核对；本次实现细节直接读取 [作者代码](https://github.com/hanlu-nju/UniCA)，固定 `727d3670e9b7993472b19f47e42ac0d75142b127`。OpenReview 直接 PDF 在本次触发 browser challenge，因此不以未读 PDF 的数字作结论。

[8] Xingjian Wu, Jianxin Jin, Wanghui Qiu, et al. “Aurora: Towards Universal Generative Multimodal Time Series Forecasting.” ICLR, 2026；首次 arXiv:2509.22295 为 2025 年。[官方 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a8e18adf6489fd3a27417d128660eaa9-Abstract-Conference.html)、[原文 v3](https://arxiv.org/html/2509.22295v3)、[作者代码](https://github.com/decisionintelligence/Aurora)。特别定位 §3.1.1：其 image branch 包含由原数值序列渲染的 endogenous image，不能与 WorldTS MMSP 的独立卫星观测混为一种信息源。代码本次固定 `a247760abbc9d17a861bc365c032368d317815f2`。

[9] Jie Yang, Yifan Hu, Yuante Li, et al. “From Observations to States: Latent Time Series Forecasting.” arXiv:2602.00297, 2026（首次 1 月；v2 5 月）。[原文 v2](https://arxiv.org/html/2602.00297v2)、[作者代码](https://github.com/Muyiiiii/LatentTSF)。定位 §3.3–3.5、§5.3–5.5。

[10] Jun Seo, Hyeokjun Choe, Seohui Bae, et al. “Adaptive Information Routing for Multimodal Time Series Forecasting.” arXiv:2512.10229, 2025。[原文](https://arxiv.org/html/2512.10229v1)，定位 §3、§5.4。作为旧先例证明 text-conditioned internal routing 已有具体方法，不援引其数字宣称普适优势。
