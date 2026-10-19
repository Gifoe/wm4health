# C38 独立批评核查：先分离信息接口失败与状态学习失败

## 摘要

本报告按冻结 brief 核查可复用时间序列状态／转移学习框架，主检索窗口为 2026-04-17 至 2026-10-17。多模态是补查视角，不是附加成功条件。原论文、作者代码和近期评测呈现的主要张力是：上下文有时确实提高预测，但编码损失、上下文日期配对、普通融合、额外监督和读出适配都可能解释收益。WorldTS 已有真实两阶段训练和同结构 observation-space 控制；Pythia、REST-TS 又分别覆盖 numerical-reference guidance 和残差专属监督。因此“保护数值基底，再用辅助模态修正潜状态”已经不是空白。本次核查没有建立一个足以越过这些完整先例与普通修复的新状态／转移机制候选。这个结论不来自通用 RNN/Bayes 的理论表达能力，也不要求动作、因果状态、零样本单 checkpoint 或无限算力。建议保留一个小型定位实验，用于判断文本是否先被接口丢失，以及恢复后的收益是否还需要新的状态目标；不要把这个定位实验包装为新方法。

## 1. 冻结问题与适用边界

以下三个问题保持原 brief 的含义，不因检索结果加码：

1. 最近六个月的 WorldTS 及可执行相近框架，实际如何把异质数值／文本／图像条件用于状态形成和转移学习？有什么具体学习目标、表示接口或可复用能力差异，已经被真实训练和强控制支持？哪些所谓不足仅是接口／预处理／原文没有承诺的性质？
2. 在有限样本／计算下，是否有一个重要的状态／转移学习残差，未被最近完整方法和普通 direct/fusion/conditional state-space 解释？优先从作者真实实验的失败和限制，而非宏大原则或经典模型理论表达能力出发。已知组件组合允许成为贡献，但必须有可区分的交互与实际可证伪收益。
3. 哪个具体最小机制与单 5090 实验可以推进一个候选？必须给出最近完整先例、差异在哪里、最强普通对照、合法公开输入／任务／代码，以及什么结果会关闭它。若没有足够证据可以无候选；但不能只罗列论文或因为 Kalman／通用 RNN 理论可表达就自动否定所有有限预算方法。

“可复用”在本任务中允许指训练框架和接口跨任务复用、每数据拟合；不自动意味着所有任务共享状态、参数或 checkpoint。本报告只读冻结 brief 和公开一手来源，没有读取 sibling notes，没有读取 C37 部分指标，也没有启动训练。root 补充的已关闭事实仅作为去重约束：Pythia 和 REST-TS 在 C34/C35 已知，C35 的 solar/Jena protected latent residual 具体配置保持关闭。

## 2. 检索与核实方法

先宽搜 `WorldTS / time-series world model / multimodal predictive state`，再沿失败和控制收窄为 `text collapse / constrained fusion / temporal alignment / context quality / direct forecasting / training budget`。分别检查三个问题：最近完整方法已经做了什么；原作者观察到什么具体退化；这些退化是否先被更普通的信息或优化解释覆盖。

纳入下文实际使用的 12 篇原论文及其可访问作者代码。WorldTS 固定核查公共 commit `269426ea6217d7152a244da0e209d922acd46d4f`，没有把“未找到代码”当批评。主窗口内初稿与窗口内修订分开记录：LatentTSF 初稿 2026-01-30、v2 2026-05-12；CFA 初稿 2026-03-23、v2 2026-06-16、v3 2026-07-15；UniCA 初稿 2025-06-27、v2 2026-03-24，属于基础先例。日期已核 arXiv submission history。[2–4]

所有结果均区分“作者报告”与“本轮独立重跑”；本轮没有独立重跑。CFA 的 arXiv 全文接口读取失败，故仅从作者仓库确认公开实验接口和对照种类，不转述未读到的细节或分数。Pythia 正文仍标注 code/model to be released，因此它是完整论文先例，不被写成当前可直接运行的公开底座。[4–5]

## 3. 按失败发生位置划分证据

分类单位是失败位置，不是论文；同一系统可以覆盖多个位置。

| 位置 | 需要辨别的现象 | 本轮主要交叉证据 | 对新候选的含义 |
|---|---|---|---|
| 输入／接口 | 原始辅助信息是否到达模型；发布时刻与事件时刻是否混淆 | WorldTS、UniCA、TESS、ScenarioDiff | 先排除编码、对齐和信息不公平；不能据此直接宣布新 transition objective |
| 条件状态／优化 | 同样信息是否被状态目标、融合和监督分配有效利用 | LatentTSF、WorldTS、Pythia、REST-TS、CFA | 可以是有限预算贡献，但需击败已知组合和 ordinary repair |
| 证据／任务 | 指标究竟测预测、语义依赖、真实日期配对还是动作响应 | TimesX、MUSE、MMTT-Bench、Divergence | 不同合同不能互相替代，也不能给用户扩大成功标准 |

三类之间不能跳步：一个模型对随机文本敏感，只说明输入改变会影响它；尚未说明它利用的是正确语义，更未说明它学到了更好的转移结构。相反，一个表征可以没有可解释物理坐标，却在有限数据下提供稳定、可复用的预测收益。

## 4. 状态学习已经覆盖到哪里

### 4.1 完整先例排除了哪些宽泛创新声明

下表的结论是：异质输入接入、受约束融合、潜目标学习和监督分配均已有可核实实现或完整学习方案；区别必须落到具体交互，不能靠重新命名。

| 方法 | 具体接口／学习机制 | 已报告控制或边界 | 对本任务的最近关系 |
|---|---|---|---|
| LatentTSF | 预训练并冻结 AutoEncoder；预测未来编码，使用 latent prediction 与 cosine alignment | 原文 §3、Eq.5；作者代码同时提供 latent 与原始 observation 训练路径 | “先编码成状态，再预测状态”已有完整可执行先例 [2] |
| UniCA | 异质 covariate homogenization；在冻结 TSFM 前后接注意力融合 | §4；v2 App.I 承认对齐近似、冲突／噪声条件可退化 | 单纯“统一异质接口＋保持数值底座”已有完整先例 [3] |
| CFA | 作者仓库公开低秩受约束融合，与 add/concat、gating、FiLM、orthogonal 比较 | README：14 个 backbone、4 个 text encoder、明确 LR grid | gating／低秩 bottleneck 必须作为普通强对照，不能只对 concat [4] |
| WorldTS | 因果 patch 编码；数值历史融合、文本／图像 CFA；未来数值条件修正预测状态；latent L1＋cosine＋VICReg 后冻结，训练预测／重构 decoder | §4、Eqs.13–21；Table 4 和 App.A.4 有同结构 one-stage 对照 | 已经实现 multimodal-conditioned latent forecasting [1] |
| REST-TS | EMA 数值预测定义残差；分解为 trend/noise/event，仅文本头接受 trend/event 专属监督 | §3.3–3.6、Eqs.4–16；App.D 明示残差缩小与预设分解的限制 | “额外文本监督防止数值通路吸收梯度”已有直接近邻 [6] |
| Pythia | 预测数值 future latent 后，用 context cross-attention residual 修正；detached numerical reference 调整 context-only 梯度；冻结表示，再适配概率读出 | §3.2–3.4；App.F 同预算 direct、numerical-only、no-reference 等控制 | protected residual／reference-guided state correction 已覆盖；代码尚待发布 [5] |

LatentTSF 和 WorldTS 的差别在于固定 AE 目标与联合学习状态目标、以及异质条件进入预测路径；WorldTS 与 Pythia 的差别又包括每数据拟合和跨数据预训练、数值预训练参考与独立概率读出。这些是可核实的训练和接口差异，不能压成“都是 JEPA，所以无贡献”，也不能放大成“只有带动作才能叫状态模型”。[1–2,5]

### 4.2 WorldTS 的证据成立到何种程度

WorldTS Table 4 的 MMSP MAE 为 full 0.171、one-stage 0.217、without covariates 0.279；App.A.4 明确 one-stage 训练 `N1+N2` epochs，不能声称作者只给 direct 跑了半程。与此同时，without covariates 去掉的是条件集合，不是单独移除图像；baseline 分数从既有工作承接、App.A.3 采用单一 seed。因此它支持其配方有效，尚不能把所有收益唯一归给视觉线索或 state objective。相较之下，Pythia 的数值／上下文／reference 分离控制更细，但也不能替代 WorldTS 同数据重跑。[1,5]

重要的概念限界来自数学接口，而不是标签：WorldTS 的 target state 为 `E(x_future)`，没有要求保留所有图像／文本内容。这对以未来数值为目标的任务是合理的信息取舍；只有在另一个明确下游任务需要被丢弃的信息时，才能把它认定为复用残差。不能要求一个预测状态同时成为可逆的多模态全信息状态。[1,3]

## 5. 真实失败揭示了什么，又没有揭示什么

### 5.1 文本顺序信息在接口丢失：精确，但属于普通修复

WorldTS 公共源码的文本入口只调用 frozen GPT-2 input embedding，随后对有效 token 做 avg/max/min pooling；公开脚本使用 avg。具体位置：[data_loader.py L129–184](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/data_provider/data_loader.py#L129)、[worldts.sh L46–60](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench/scripts/worldts.sh#L46)。这是 bag-of-token 接口：在截断后的 token 多重集相同这一条件下，改变顺序不会改变均值。依赖实体角色、否定范围或先后关系的差异可能已经不可恢复。该判断由代码直接推出，并非作者观察到该特定错误。

但 UniCA 已比较多种文本 encoder，TESS 与 ScenarioDiff 又提供语义提取／结构化条件先例。因此这里的第一对照应是缓存一个 contextual sentence encoder 或结构化事件表示，再接现有 CFA／direct 模型；若这样就补回收益，缺口是 encoder/interface，不是新状态／转移学习。[3,11–12]

图像方面，不能从“未来 weather 条件”推断图像也来自未来：固定 commit 的 [image model L140–170](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/MMSP-bench/models/adapter/worldts_image/model.py#L140) 要求 satellite tensor 与历史 target 对齐，仅把 `feat_dynamic_real` 的未来段传为 `x_future`；[module.py L139–149](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/MMSP-bench/models/adapter/worldts_image/module.py#L139) 也把二者长度分开。合法输入还需要数据提供方的 issue-time 记录，这段代码本身既不证明所有 NWP 完全合法，也不证明泄漏。

### 5.2 上下文质量、日期配对和负迁移是实证问题，不自动成为状态残差

TimesX 在固定数字、模型、prompt 下替换 Time-MMD 文本，作者报告 aggregate MASE 从 0.906 到 0.840；其 §4.5 又发现简单等权 AvgEns 优于三种 revision。这与 MUSE 的“正确上下文通常改善、错误／错时上下文退化”相容：两者共同反对把复杂融合的得失全部归给 transition 学习，但任务规模、模型和上下文不同，不能直接比较分数。[7–8]

MMTT-Bench 则把“文本含额外信息”和“文本与这个日期配对有信息”分开：§4.2、App.G.2 Tables 17–18 中，多数被测 Time-MMD 配对对比不显著；这只约束它的表示、估计器和下游设定。它还显示系统性反向描述仍可统计可预测，因此 MI 高不是语义真实度证书。TimesX 的文本替换改善与此并不矛盾：原配对可能太弱，改良语料可以改变结论。[7,9]

这些来源足以要求日期／内容与普通融合审计，却不足以断言“文本无用”或“所有模型都缺少新的异质状态”。反例也已存在：Pythia 正常上下文改善 aggregate，但 §4.4 明说部分 finance/healthcare groups 仍偏好 numerical-only；REST-TS 也报告部分 backbone/domain 退化。这是任务条件化的负迁移，不是单一统一瓶颈。[5–6]

### 5.3 更多梯度、潜空间几何与有效状态不是同一个命题

REST-TS 的专属监督防止数值参数直接降低文本辅助损失，但这并不逻辑保证内容依赖：常量或领域平均残差也能被文本头部分拟合。Pythia 增加 null/random、source-removal 和相同输入下 no-reference 控制，比单看梯度范数更接近回答利用问题。两者都不能让任意未来候选免于 `joint + extra numerical loss`、普通 gate、以及 train-time shuffled-context 对照。[5–6]

LatentTSF 的时序邻接距离、频谱和 t-SNE 是一种有用诊断，却不直接验证物理状态或条件充分性。独立推论：将一个已训练 latent 作可逆线性变换并相应变换 predictor/decoder，可以改变距离与方向分数，而保持输入输出预测不变。因此新候选要靠同信息有限预算的预测／复用收益建立价值；不能只以“更平滑”替代能力证据。这也不否认 LatentTSF 或 WorldTS 已报告的任务收益。[1–2]

### 5.4 undertraining 审计应有限、对称、可结束

WorldTS 的 one-stage epoch 数已经匹配；合理追问是实际 updates、样本暴露、早停／LR 搜索、模态缓存与 decoder 适配是否同等，而不是笼统宣布 direct 没训够。Pythia 的容量×预算实验显示更长训练并不单调更好，作者也明确不把它解释为 scaling law 或穷尽容量。有限预算配方改善可以成立，不需要证明任何可表示同函数的 RNN 经过无限训练仍失败。[1,5]

评审上应预先给 strongest ordinary control 一个有限的 1×／2× 预算和同数量验证选择，报告成本—性能曲线；若候选胜出的只是 ordinary baseline 更差的 schedule、没有额外数值监督或错误 decoder 接口，关闭该机制解释。在已关闭 C35 配置上继续换 loss 名称或换一轮 seed，不构成本轮新候选。

## 6. 哪些标准是人为加码

| 加码要求 | 本任务允许的要求 | 不能作出的推论 |
|---|---|---|
| 必须有 actions／planning | 可对被动系统学习条件预测状态与未来转移 | 无动作不等于无状态学习贡献 |
| 必须识别真实因果／物理状态 | 针对声明任务证明状态表示有实用价值 | 无识别定理不等于有限预算配方无效 |
| 必须单 checkpoint 跨域 zero-shot | 可复用框架，可每数据拟合 | dataset-specific 不是本 brief 的失败 |
| 必须包含 text＋image | 多模态用于补查盲区 | 数值方案不因缺模态而出局 |
| 必须超过任意 Bayes／RNN 表达集合 | 超过明确、有限、强的普通实现 | 理论包含关系不能替代训练实验 |
| 必须有全新基础组件 | 已知组件的交互可以贡献 | “组合”本身不是否决理由；没有交互证据才是 |
| 所有 future covariate 都是假未来真值 | 按 forecast origin／issue time 判合法性 | 不能把已发行预报和事后实现值混称泄漏 |
| 必须全面降低 latent geometric score | 指标与明确能力合同对应 | 坐标依赖诊断不应成为唯一录用门槛 |

近期 Divergence 论文研究的是另外一个已声明合同：执行计划下 forecast error 与声明动作方向的一致性。其 §2 明说方向由先验规定而非从混杂数据识别；App.B.2 还说明 actions/exogenous 贡献未分开，held-out 同时选 checkpoint 并报告 MAE。它提供动作任务的探针，不能用来给本 brief 强加动作，也不是 WorldTS 因果失败的直接实验。[10]

## 7. RQ3：本轮无新机制候选；给一个可关闭动机的单卡定位实验

**不提出 protected residual、reference loss、文本残差专属监督的新版本。**最近完整先例分别是 Pythia、REST-TS 和 WorldTS；C35 已关闭的具体配置不得复活。本轮唯一值得准备的小实验是“编码接口与状态目标的析因定位”，它是 falsifier，不是新方法。

### 7.1 合法输入与最小任务

第一任务固定公开 Time-MMD Climate，history 8、horizon 12，按 WorldTS 70/10/20 时间划分；只取 forecast origin 之前可获得的 reports，读取保留的 `start_date/end_date` 后核实际发布时间。若无法确定报道何时发行，先将相应记录置缺失，不从事件归属日期推断可见。代码入口为 WorldTS `TimeMMD-bench/exp/exp_worldts_forecasting.py`；[作者 README](https://github.com/decisionintelligence/WorldTS/tree/269426ea6217d7152a244da0e209d922acd46d4f/TimeMMD-bench) 与 [CFA 作者仓库](https://github.com/seunghan96/cfa) 提供公开实现／数据使用路径。不要重新调用文本生成服务，不给普通对照较少的原始文档。

先用已本地具备的 GPT-2 embedding 与一个可合法使用的小型 contextual encoder 缓存同一批原文；若后者本地不可用，在 root 决定实验前明确依赖，不能暗中换入更大的 API 模型或批量拉权重。单 5090 只训练小型预测器；无需复现 Pythia 的大规模预训练。

### 7.2 最少需要分开的变量

| 轴 | 水平 | 保持不变 |
|---|---|---|
| 文本接口 | 当前 bag embedding；contextual sentence embedding | 原文、截断、可见时间、缓存 dtype |
| 预测学习 | 原生 WorldTS 两阶段；同结构 direct/CFA＋额外 numerical-only forecasting loss | 历史、horizon、latent／hidden 宽度、总 updates、有效 batch、验证选择数 |

另列 numerical-only、普通 conditional GRU/SSM（相同 cached embeddings）作为 ordinary controls。这里“额外 numerical loss”是排除监督分配解释，不是新提案；对照的强度不能靠省掉这条分支人为降低。先做 3 seeds；学习率仅在预先声明的有限集合内用 validation 选，direct 允许 1×／2× 预算，报告真实时间与 sample exposures。采用作者小型 PatchTST 配置，无必要下载 TSFM。不能在看到 test 结果后换主任务、horizon 或配对规则。

### 7.3 结果如何关闭解释

1. **contextual encoder 同时改善 latent 与 direct，而两者相当**：关闭“需要新 state objective 才能利用语义”的解释，记为普通接口修复。
2. **同信息 direct＋额外数值监督或 conditional GRU/SSM 追平**：关闭该有限预算状态机制候选；不要用更好 latent 图形挽救。
3. **训练和评测都采用同域错时文本后收益仍相当**：关闭“精确日期上的外部事件改变转移”解释；仍可能存在领域／慢变信息，不宣称文本完全无用。错时控制应保留领域与大致季节，使用长度大于 forecast horizon 的 block shift，避免逐点随机把所有边际结构打散。
4. **只在给 ordinary control 少量训练时胜出，1×／2× 有限补训后消失**：关闭独立机制解释，记为训练配置差异。
5. **所有合法文本模型都不胜 numerical-only**：关闭这个任务作为候选证据载体；不能据一个小任务否定整个多模态方向。
6. **只有恢复语义后的 latent 目标稳定超过所有 ordinary controls**：才值得在预先约定的第二任务（例如 Health，history 36、horizon 12）复核，并测训练时 aligned／block-shifted 差异。三 seeds 与重叠窗口不等于足够独立样本；不确定区间应按时间块配对 bootstrap，不把窗口当独立重复。

本报告没有给新机制命名，也没有宣布这个析因设计已提供新颖性；其作用是以有限预算关闭错误动机。若复核后仍有残差，再为具体机制建候选合同。仅凭现有论文的 negative transfer 或“text collapse”标签，不能预先指定机制会成功。

## 8. 结论：逐项回答冻结 RQ

**RQ1。**最近框架确实把异质条件编码、融合进预测状态，并有真实训练支持。完整近邻已经覆盖统一协变量接口、受约束融合、latent future target、残差专属监督和 numerical-reference guidance。WorldTS 的公开代码可执行；具体文本顺序损失是接口问题，历史图像与未来数值协变量应分开。其任务没有承诺因果状态、控制或单 checkpoint 迁移。

**RQ2。**存在明确的实践失败：弱／错时上下文、模态负迁移、监督分配和读出不兼容。但本轮没有把其中一个隔离为普通 direct/fusion/conditional state-space 与最近完整方法尚不能解释的重要状态／转移学习残差。不能因此宣布残差不存在；也不能将未隔离的失败提升为新框架动机。

**RQ3。**本轮无足够证据推荐新机制候选。可以准备上述单卡析因定位实验；若收益被 contextual encoder、extra numerical loss、普通 conditional GRU/SSM 或有限补训解释，就关闭相应机制叙述。若保留稳定剩余，下一轮才有依据提出具体组合和交互。这个判断保持 C35 关闭、不使用 C37 部分结果，也不附加 action／causal／zero-shot／无限计算门槛。

## 参考文献与精确位置

[1] Zhu, Y., Qiu, X., Cheng, H., et al. “WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting.” arXiv:2609.31162v1, 2026-09-25。[原文](https://arxiv.org/html/2609.31162v1)：§3.1；§4.3–4.5，Eqs.13–21；§5.4 Table 4；App.A.3 Eq.22、single seed；App.A.4 Algorithm 2；App.C。[固定代码](https://github.com/decisionintelligence/WorldTS/tree/269426ea6217d7152a244da0e209d922acd46d4f)。

[2] Yang, J., Hu, Y., Li, Y., et al. “From Observations to States: Latent Time Series Forecasting.” ICML 2026；arXiv:2602.00297v2, 2026-05-12。[原文](https://arxiv.org/html/2602.00297v2)：§3 Eq.5、Remark 3.1；App.A–B；[日期](https://arxiv.org/abs/2602.00297)；[作者代码](https://github.com/Muyiiiii/LatentTSF)，README Original baseline mode／pretrained AE。

[3] Han, L., Liu, Y., Li, L., et al. “UniCA: Unified Covariate Adaptation for Time Series Foundation Model.” ICLR 2026；arXiv:2506.22039v2, 2026-03-24。[v2 原文](https://arxiv.org/html/2506.22039v2)：§4 Methodology；App.G.2 Table 13 的 modality encoder 控制；App.I Discussion of Limitation；[日期](https://arxiv.org/abs/2506.22039)；[作者代码](https://github.com/hanlu-nju/UniCA)。

[4] Lee, S., Seo, J., Lee, J., et al. “Rethinking Multimodal Fusion for Time Series: Text Modalities Need Constrained Fusion.” arXiv:2603.22372；作者仓库使用 “Auxiliary Modalities” 标题。[身份／版本记录](https://arxiv.org/abs/2603.22372)；[作者代码](https://github.com/seunghan96/cfa)：README Overview、Fusion Methods、Learning Rates、Running Experiments。全文接口未成功读取，本文不引用未核分数。

[5] Dai, X., Chen, H., Hu, Y., et al. “Pythia: Toward Foundation World Models for Multimodal Time Series.” arXiv:2610.05240v1, 2026-10-04。[原文](https://arxiv.org/html/2610.05240v1)：首页 code/model 状态；§3.2–3.4 Eqs.1–5；§4.3–4.6；App.E.2–E.3；App.F.1 Tables 9–12；App.G.1 Table 15。

[6] Nguyen, H. H., Nguyen, M. H., Nguyen, D., et al. “Does Text Actually Help? Uncovering and Resolving Text Collapse in Multimodal Time Series Forecasting.” arXiv:2606.19413v1, 2026-06-17。[身份](https://arxiv.org/abs/2606.19413)；[原文](https://arxiv.org/html/2606.19413v1)：§3.3–3.6 Eqs.4–16；§4.2–4.4；App.A.1；App.D Limitations。

[7] Liu, H., Zhou, Y., Sen, R., et al. “Rethinking Multimodal Time-Series Forecasting Evaluation.” ICML 2026；arXiv:2607.06973v1, 2026-07-08。[原文](https://arxiv.org/html/2607.06973v1)：§3.1.1–3.1.3、Table 3；§4.1 的 announcement-date 截止规则；§4.5–4.6；App.S。文中 leakage-free 是作者的时点隔离合同，不被本文扩成对所有未来模型的保证。

[8] Chen, P., Zhuang, Z., Chen, H., et al. “Beyond Numerical Time Series: A Unified Benchmark for Multimodal Forecasting with Heterogeneous Context.” arXiv:2609.15087v1, 2026-09-14。[原文](https://arxiv.org/html/2609.15087v1)：§4.2.2 Table 4 的 Original/Noisy/Wrong Context；App.C.1 与其作者仓库／checkpoint 对照表。

[9] Andrews, E., and Mengaldo, G. “When Does Text Inform? Benchmarking Information-Theoretic Metrics for Multimodal Time-Series Forecasting.” arXiv:2609.11282v1, 2026-09-10。[原文](https://arxiv.org/html/2609.11282v1)：§4.1 的 inverted signal；§4.2；§5；App.G.2 Tables 17–18；[作者代码](https://github.com/MathEXLab/MMTT-Bench)：README Reproducing the paper，含 self-contained synthetic splits 与 Time-MMD 路径合同、MIT license。

[10] Zhang, H., Guo, J., Xu, Z., et al. “On the Divergence of Accuracy and Mechanism Consistency in Time Series World Models.” arXiv:2610.01842v1, 2026-10-01。[原文](https://arxiv.org/html/2610.01842v1)：§2 的 imposed mechanism sign；§3.1、§4.3、§5.1；App.B.2 的 held-out／plan 合同；App.C 的训练协议。

[11] Li, L., Wang, Y., Yan, J., et al. “From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space.” arXiv:2603.12664v1, 2026-03-13，基础先例。[原文](https://arxiv.org/html/2603.12664v1)：§3.2 的 future-statistics-derived semi-synthetic text；§4.1–4.3 的 primitives 与 confidence gating；§5 ablation。其诊断用未来统计生成文本是显式半合成合同，不能混称为真实历史新闻可见性。

[12] Tran, T.-B., Nguyen Cong, D., Le, D.-T., et al. “ScenarioDiff: A Scenario-level Guidance Framework for Multimodal Time Series Forecasting.” arXiv:2608.17164v1, 2026-08-17。[原文](https://arxiv.org/html/2608.17164v1)：§IV-B 的 Scenario/Anchor agents；§V-A 的 offline cached Gemini 2.5 Flash、BERT 和训练合同；[作者公开实现入口](https://anonymous.4open.science/r/ScenarioDiff_ICDM-2C4C) 由 arXiv abstract 给出。本文未检查该匿名仓库，不将其记为已独立运行。
