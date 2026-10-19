# C38 constructive/adjacent：异质条件下的状态学习，候选数 0

2026-10-17。独立分工；检索前已读本轮 RESEARCH_BRIEF.md，未读 sibling notes。按 deep-research 的原文核验与反向检索流程执行。本文件是研究判断，不是论文草稿，也没有启动训练。

## 摘要

本轮没有建立一个足以进入新实训的状态/转移学习机制。这个结论不是“多模态已经解决”，也不是“通用 RNN 理论上能表达，所以不能做”。有作者实测的有限预算残差：**较好的多模态 latent matching 不稳定地转化为较好的数值预测**。但把它修成 protected numerical residual、额外 numerical loss 或 decoder-aware task loss，会回到已知方法和 C35 已关闭配置；把 missingness/语义条件塞进转移算子，又缺少超越已有条件状态空间方法的具体交互。当前最有价值的动作是保留这个残差及强普通对照，停止为一个尚不存在的区别编造新名字。

## 1. 冻结问题与证据方法

沿用本轮三个 RQ：异质条件实际怎样参与状态形成与转移；有没有未被最近完整方法和普通条件模型解释的有限预算残差；有没有一个可用公开输入、单 5090 决定去留的最小机制。

检索主窗为 2026-04-17 至 2026-10-17，另追溯直接近邻。三组串行检索分别覆盖：latent/world-model 多模态预测；modality collapse、缺失和异步；条件/availability 改变转移。阅读使用 arXiv 全文 HTML、官方代码和数据仓库，机制与结果按小节/公式/表格核对；不采用二手综述的性能数字。Pythia、REST-TS 已是前轮先例，不算本轮新发现。root 补充的 C35 solar/Jena 已关闭事实仅用于去重，不将其改写成新实验建议。

核验限制：FACTS 的 OpenReview PDF 与 API 本轮返回 403/验证页，匿名代码也未成功访问；搜索缓存虽能显示原文片段，仍不把它作为本报告的决定性完整机制证据。Aurora 仅核到官方 proceedings 摘要，故不借其摘要展开未经核验的损失细节。没有使用未核实来源补齐结论。

## 2. 按条件参与学习的位置划分近邻

| 位置 | 经核验的完整先例 | 对本轮构思的实际约束 |
| --- | --- | --- |
| 条件修正未来 latent，再单独读出 | WorldTS；Pythia | “多模态进入状态预测、两阶段、冻结生成器、reference residual”都已有完整路线。应比较目标和有限预算交互，不能靠换名称立项。 |
| 将辅助分支的任务分离 | REST-TS | “数值流太强，文字学不到东西；给文字独占残差监督”已有直接方法。 |
| 异质编码、缺失条件推断与融合 | UniCA；CoRA；TRACE | 统一输入、adapter、routing、条件补全本身不是未解决的状态学习机制。 |
| 条件直接改变 recurrence | CAMOS；NOAH | “条件必须改变 transition 而非只拼接输入”不能当作空白；必须写出超越同信息条件模型的具体约束与收益。 |

WorldTS 的 §4.3–4.5、Eq.14–21 明确给出 modality adapter、历史 CFA、未来 numerical residual 与两阶段学习；不是只有数值接口。[WorldTS 全文](https://arxiv.org/html/2609.31162v1#S4)。Pythia 的 §3.2–3.4、Eq.1–5 已给 context residual、detached numerical-reference guidance 与独立 probabilistic readout。[Pythia 全文](https://arxiv.org/html/2610.05240v1#S3)。这两条是最接近用户所求框架层次的完整学习先例。

REST-TS 的 §3.2–3.6、Eq.4–13 以 EMA 数值预测形成 residual，并把其 trend/event 部分交给文本分支独占监督；Appendix D 同时承认单窗口单文本、固定分解和残差变小时信号减弱的边界。[REST-TS 全文](https://arxiv.org/html/2606.19413v1#S3)。因此，给辅助分支“不可被数值流代劳”的责任，已不是一个新的操作。

TRACE 的 §3.3–3.4 是 cross-modal MoE context 下的 conditional diffusion，再做下游 fusion；§5 明确主要补 within-modality partial observation，而非保证整种模态完全不存在时也能恢复。[TRACE 全文](https://arxiv.org/html/2606.06285v1#S3)。这个未覆盖范围只是范围边界，不能自动推出“全模态生成 + WorldTS”就是有新机制的框架。

CoRA 的 §3.1 Eq.3 对文字/图像做时间平均，§3.2 Eq.6–7 做权重与零初始化 scale/shift 注入；Appendix D 承认时间平均丢失细粒度动态。[CoRA 全文](https://arxiv.org/html/2510.12681v1#S3)。但这不是所有近邻的共同缺点：WorldTS 保留历史时间轴，Pythia 保留 event-time token。故不能只修 CoRA 的 pooling 就声称解决通用状态学习。

## 3. 一个真实残差，以及它没有证明的东西

Pythia §4.4 与 Appendix F.4/Table 14 给出罕见的直接证据：context correction 的平均 latent error 在全部 29 组降低，但 matched numerical-only 对比只在 17 组改善 MASE、18 组改善 WSQL；无 reference 版本最终 latent error 更低（0.02236 对 0.02291），预测反而更差（MASE 0.7086 对 0.6981）。这说明该有限训练方案的 latent 距离不是可靠的最终效用排序。[Pythia §4.4、F.4](https://arxiv.org/html/2610.05240v1#S4.SS4)。

独立综合：问题应写成“怎样在固定表示宽度和训练预算下，使异质条件更新保留可被同一读出利用的预测信息”，不能写成“现有方法没有多模态状态”。同样，以上数字没有定位唯一病因：target geometry、encoder 保真度、decoder 兼容、语义处理、联合优化和样本量都可能解释它。单凭这个失配，无法推导一个新正则项的必要性。

普通修复具有直接竞争资格：同信息的 joint forecasting，加数值辅助目标；更完整的 frozen semantic encoder；保留时间信息的标准 cross-attention/FiLM；或标准 conditional GRU/selective SSM。必须让这些方法充分训练，并以相同输入、参数与计算预算比较。不是在低配 early fusion 上赢一次，就获得了状态学习证据。

这里明确停止以下构思：不把 reference penalty 换权重说成新机制；不把 future encoded target 换成 decoder loss 说成新状态；不把 modality dropout、补全和 conditional transition 串联后，仅凭模块更多声称交互创新。已知组件可以形成贡献，但目前缺的是**一个明确交互为何在真实数据中解决哪个剩余错误**，而不是新原则。

## 4. 相邻路线核查：让可见条件进入转移，仍未形成新候选

CAMOS §4.2–4.4 已让 pairwise availability product 进入耦合 oscillator transition。它的 §4.3/Definition 4/Theorem 5 明确只针对 availability-independent 的单个线性层，明确排除附加 mask 的 selective Mamba，并不排除 nonlinear readout 或 MedFuse/DrFuse 的非线性融合。[CAMOS 全文](https://arxiv.org/html/2609.39484v1#S4.SS3)。因此这里不能用其定理制造“普通融合不可能”的缺口；其临床数据资格要求和未定位到可访问匿名仓库，也使它不是马上可复用的小公开训练底座。

NOAH 的 Methods “Multimodal token representation”与“Variational autoregression with a context-conditional prior”已经把多模态临床事件序列放入 history-conditioned latent prior/posterior，并用一个共享随机 latent 联结不同输出头。[NOAH 全文](https://arxiv.org/html/2609.09140v1)。它不是 WorldTS 的同任务得分对手，但足以表明“共享状态承载模态联合创新”需要进一步精确定义。

最接近可构思的组合会是“语义条件与可见性共同改变 transition，同时保存跨模态信息”。但本轮没有找到证明这个组合重要、且普通 conditional transition 未修复的公开实例；MMSP 的标准输入是规则对齐序列，人造大量缺失后取得优势也不足以证明原任务的剩余问题。当前没有新增学习约束，也没有能区分新交互与一般条件参数化的反事实预测。所以它没有通过候选门槛，**不建议给它命名或训练**。这不是对所有未来组合的否定。

## 5. 公开输入合同与可执行比较底座

最近的算法比较必须承认完整 Pythia；但本轮未在其论文或 arXiv 页面找到作者发布的训练仓库/权重链接，不能把自己在小数据上重建的缩小版叫作复现整个共享预训练系统。可立即审计的完整底座是已知 WorldTS pin `269426ea6217d7152a244da0e209d922acd46d4f`，以及作者的 UniCA 仓库；前者有分开的文本/图像训练入口，后者支持冻结 TSFM 的异质 covariate adaptation。[WorldTS README](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/README.md#L36-L67)，[UniCA 作者代码](https://github.com/hanlu-nju/UniCA)。

### 文字：Time-MMD

数据提供方公开的 schema 为 numerical `start_date,end_date,OT,...`，text `start_date,end_date,fact,pred`，附原始来源指针；这些区间字段本身不等于独立审计过的 publication/availability timestamp。[Time-MMD 数据仓库 README](https://github.com/AdityaLab/Time-MMD#dataset-overview)。原 benchmark 可以按其现有任务合同比较，但不能据此无条件承诺真实部署 release-time 合法性。

WorldTS 代码的具体合同：`data_loader.py:67` 选 `fact`，`:81–100` 做 chronological split 和 train-only numerical scaling，`:223–234` 只取历史窗口文本。更关键的是 `:141–181` 直接调用 `get_input_embeddings()`，再做 avg/max/min pooling，没有 contextual Transformer forward。由代码可直接推出：相同 token 多重集合的句内重排会得到相同文本向量。这是可定位的表示限制，先要排除普通 encoder 改善，不能直接归罪 latent dynamics。[固定版本文本 loader](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/data_provider/data_loader.py#L129-L184)。

### 图像：MMSP

提供方 FusionSF 的 §4 和 Appendix A 明确区分 historical satellite 与 future NWP；其公开仓库提供 `solar_power.csv`、`satellite.npy`/时间/坐标和 `nwp.csv` 的合同及数据链接。[FusionSF 原文](https://arxiv.org/html/2402.05823v1#S4)，[作者仓库](https://github.com/MAZiqing/FusionSF)。这是已存在的真实异质输入，不能称未来 NWP 必然是未来真值。

WorldTS MMSP `model.py:126–170` 强制 satellite 的前两维与 historical target 对齐；future numerical covariates 仅在 `with_future` 下切入。新机制若使用此载体，应保持同样的历史图像/未来 NWP 合同，并审计 NWP issue time；不能给新方法未来卫星图、同时让对照只见历史图。[图像模型输入检查](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/MMSP-bench/models/adapter/worldts_image/model.py#L126-L170)。本轮未下载数据，也没有声称数据包的所有发行时间已通过审计。

## 6. 单 5090 的最小决定性实验：仅普通修复诊断，不是新机制实训

**当前推荐的新候选训练数为 0。** 若 root 后续决定继续排除文字实现因素，可先做下列小诊断；它不复活 C35，也不能产出“新 world-model 方法”的结论。

固定 Time-MMD Energy 与 Health 两个公开任务，沿已有合同使用 `fact`、历史窗口、H=12、chronological 70/10/20；不选最有利数据片段。四个 arms 构成一个 2×2：

| | 原生 GPT-2 input-embedding pooling | 同一冻结 GPT-2 的 contextual hidden-state pooling |
| --- | --- | --- |
| 原生两阶段 WorldTS | W-lex | W-ctx |
| 同输入/相近容量的 joint/direct forecasting，含普通 numerical auxiliary | J-lex | J-ctx |

数值 encoder、条件 adapter、readout 输入、最大 token 数和可训练参数范围相同；唯一交叉因素是文本表示与学习范式。GPT-2 一次性离线编码，全部模型只读同维缓存，因此 forecast-training FLOPs 的比较不被语言编码成本混淆；另单独报告一次性编码成本。若本地没有已有权重/数据，不能把下载或数据审计假装已完成。

最小 screen 为每任务两种固定 seeds，共 16 个小 fits；全任务共用固定 optimizer/上限，不对每 arm 搜参。预算用总 optimizer updates 和实际 GPU seconds 双报；先做 100 steps 显存/耗时估计，必要时四组统一减 batch 并等效累积，目标峰值不超过 16 GiB。最终训练上限及 early-stop 规则在 test 前冻结。这里没有给未经实测的“几小时完成”承诺。

主指标 MAE，辅 MSE；按连续时间块做配对区间，避免把重叠窗口当独立样本。附正常文本、跨样本错配文本、constant text 的同 checkpoint 敏感性；这些只检测内容利用，不是因果证据。评价不得以 latent MSE 或 effective rank 单独裁决。

关闭/重开规则：

1. 若 J-ctx 已消除 W-lex 的主要缺陷，当前问题优先归为 encoder/普通 supervised 学习问题，不立新状态机制。
2. 若 W-ctx 与 J-ctx 相当，或文本干预无稳定效应，也不能宣称 latent world states 获得了额外能力。
3. 即便 W-ctx 稳定领先 J-ctx，也只是已知两阶段框架在较好文字输入上的证据，仍不构成新机制；需要另有明确新交互后才可重开研发。
4. 若以后有候选，至少必须在 J-ctx、W-ctx、同信息 conditional GRU/selective SSM 之外胜出，并经“去掉新交互、保留参数与训练信号”的消融区分；只有异常子集上的提升不够。

该诊断的作用是低成本关闭错误归因，不是继续刷文本榜单。既然本轮没有候选，完全可以不运行它。

## 7. 对三个 RQ 的回答

**RQ1：** 异质条件进入 latent future prediction、残差指导、表示补全及条件 recurrence 都已有真实完整方法，WorldTS 本身也有公开多模态训练底座。这里的缺口不能写成“尚无框架或尚无接口”。

**RQ2：** 有具体有限预算残差：latent alignment 与下游效用失配、辅助语义可能被弱编码/数值捷径掩盖；但本轮尚未把它与普通修复、已知 residual supervision、conditional state-space 分开。这个范围内没有可靠的新状态机制证据。

**RQ3：** 候选数 0。不存在可诚实填写的“新机制精确差异”；不能为了安排单 5090 工作而发明一个。保留原文残差、完整先例和上述可失败的普通修复诊断，等待真正能改变该判断的机制或数据证据。

## 核心来源定位

- WorldTS，2026-09-25：[全文](https://arxiv.org/html/2609.31162v1)，§4.3–4.5/Eq.14–21；Appendix A.1–A.3。代码位置已按固定 commit 单列。
- Pythia，2026-10-04：[全文](https://arxiv.org/html/2610.05240v1)，§3.2–3.4/Eq.1–5；§4.4；Appendix F.4/Table 14。来源用于机制范围和失配证据，不重复计作新线索。
- REST-TS，2026-06-17：[全文](https://arxiv.org/html/2606.19413v1)，§3.2–3.6；Appendix D。用于排除独占残差监督的重命名。
- TRACE，2026-06-04：[全文](https://arxiv.org/html/2606.06285v1)，§3.3–3.4；§4.3；§5 “Scope of Missingness Modeling”。
- CAMOS，2026-09-30：[全文](https://arxiv.org/html/2609.39484v1)，§4.2–4.4，特别是 §4.3 Definition 4/Theorem 5 及其范围说明；Reproducibility statement。
- NOAH，2026-09-08：[全文](https://arxiv.org/html/2609.09140v1)，Methods “Multimodal token representation”“Variational autoregression with a context-conditional prior”。仅作相邻机制边界，不把临床结果外推到一般预测。
- CoRA，主窗前先例，2025-10-14：[全文](https://arxiv.org/html/2510.12681v1)，§3.1 Eq.3，§3.2 Eq.6–7；Appendix D。
- FusionSF，主窗前先例，2024-02-08：[全文](https://arxiv.org/html/2402.05823v1)，§4 与 Appendix A；[作者数据/代码仓库](https://github.com/MAZiqing/FusionSF)。
- [Time-MMD 数据提供方](https://github.com/AdityaLab/Time-MMD)，README “Dataset Overview”“Numerical Data”“Textual Data”；[UniCA 作者代码](https://github.com/hanlu-nju/UniCA)，README “Overview”“Datasets”“Quick Start”。

执行边界：只写本文件；未运行 server/训练作业，未批量下载数据/权重，未触碰 cycle6 或 W&B launcher，未改 C37 源码或用 partial 指标裁决。
