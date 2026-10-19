# C41 独立建设性／相邻视角：把“学状态”改写成可检验的机制交互

日期：2026-10-17。范围：独立阅读本轮 RESEARCH_BRIEF，并检索公开一手论文；没有读取其他代理或旧轮笔记，没有连接服务器、训练模型或下载数据／权重。下述内容是两个候选及实验合同，不是已证明的新架构。

## 摘要

本视角保留两个不同方向。候选 A 面向复用：**可更换的观测推断接口 × 可组合的动态机制**，检验它们是否在未见“观测方式—动态系统”组合上产生超过各自独立效果的收益。候选 B 面向模拟：**具有共享粗路径的条件生成 × 多层 Monte Carlo 查询**，检验同一个模型能否在一组路径函数上降低误差—成本，而不是仅让若干内部 law moments 更一致。A 与本轮状态问题联系更直接，优先做一个小型已知真值筛选；B 是状态路线缺乏收益时的真实替代，不需要以新状态理论为前提。

两者的材料都已存在。贡献只能来自被公平验证的必要交互、完整训练／推断配方和跨域实际收益。NCDSSM 已经拆分 recognition 与 dynamics，modular NRI 已经学习可复用动态模块；mr-Diff 已经粗到细生成时间序列，MLMC diffusion 已经研究生成模型的估计成本。因此，仅画出这些模块的拼接图，没有贡献。

## 1. 冻结问题和检索方法

本报告回答三个问题：

1. 哪一种可复用接口会使“状态”承担普通单次条件生成之外的实际职责？
2. 哪个最小已知状态实验能分开表示信息、动态近似和推断近似，并允许普通方法取胜？
3. 与完整最近方法比较后，是否还存在能在一张 RTX5090 上筛选的机制交互？

检索沿两个相邻分支进行：一是 compositional interfaces、modular relational dynamics、continuous-discrete SSM；二是 multiresolution temporal generation、multilevel Monte Carlo、conditional functional estimation。随后补查已知 close references 的全文。引用均用标题／作者核对存在性，再核对摘要或方法正文。六个决定性来源列于第 6 节，另有两个边界／近邻补充来源。没有把搜索未命中解释成“从未有人做过”。

## 2. 最近工作把空白限制在哪里

| 决定性比较 | 已有证据 | 本轮不能照搬的说法 | 尚需直接测试的条件 |
|---|---|---|---|
| NCDSSM vs modular NRI | 前者将复杂观测经 auxiliary variables 接入连续时间 Bayesian state inference；后者把关系推断／动态预测写成可组合模块的 meta-learning。[1][2] | “观测与动态分离是新的”“共享动态模块是新的” | 在相同推断预算下，二者的结合是否改善未见观测—机制组合，而不是只改善 seen combinations |
| LIFTS vs FlowNP | 前者统一部分观测下的 latent-driven generation 与前后向推断；后者对任意 context/target 点集合学习联合条件生成。[3][4] | “一般生成器不能利用新观测”“多任务条件预测天然要求我们的状态” | 持续复用的机制接口是否带来质量／适应成本优势；二者均必须作为完整控制 |
| mr-Diff vs MLMC diffusion | 前者已经按粗趋势到细节条件生成；后者已经耦合不同精度生成采样器以减少 Monte Carlo 计算。[5][6] | “多尺度生成是新的”“把 MLMC 用到 diffusion 是新的” | 时间分辨率层次能否与共享条件信息共同降低一组路径函数的实际估计误差／计算成本 |

MCJEPA 还明确讨论了概率 context state、Markov transition、transition composition，以及 predictive bottleneck 与 residual predictability。[7] 其有限 HMM 构造并不解决所有有限数据、神经网络优化和跨域接口问题；但它足以阻止把“概率状态＋Chapman–Kolmogorov”重新包装为这里的候选。类似地，LIFTS 的结构子类识别定理不是任意观测变换下的神经框架保证，而其现有条件生成能力也不能被忽略。

## 3. 候选 A：观测接口的复用与动态机制的复用

### 3.1 明确交互，而非泛泛“分离 state 与 observation”

考虑同一类动态系统被不同传感器集合、测量混合、噪声水平和记录频率观察。框架包含两类独立可替换的部分：

- 观测接口以某个 view 的观测及 mask/time 为输入，给出动态辅助量的**分布性证据**，而非直接将它当作无噪声状态；
- 动态部分由跨系统复用的局部 transition modules 及组合规则组成。这里不要求图具有因果含义，也不要求变量具有统一物理语义；实体／分组若使用，必须事先声明或由同样输入学习。

**待证假设 A：**有限数据时，如果观测接口和动态模块同时复用，动态模块可利用多种 view 对同一机制的约束，观测接口也可利用多种机制中的重复结构；这种交叉约束有可能降低未见 view–mechanism 组合的条件预测误差。只复用任一侧时，另一个单体网络可能把 view 差异编码成动态差异，或者把动态差异编码成观测差异。

这是关于**样本效率和组合复用**的假设，不是“滤波不够强”的失败故事。NCDSSM 提供可靠的观测—动态接口，modular NRI 提供动态组合学习，但两篇的现有结果不能直接推出上述交叉收益。[1][2] 普通 hierarchical SSM 本来就能写出这个 factorization，所以该普通模型是强竞争者；不能把它排除后声称状态有用。

框架层面的可能产物是：一套训练时交叉共享、适应时只更换所需接口、预测时保留证据不确定性的生成配方。**没有必要预先发明额外 consistency loss、teacher、因果标签或新数学算子。** 若普通的模块化 SSM 配方已经实现全部收益，这就是应当采用的实现，研究问题变为是否有足够强而新的交互证据，而不是强行添加模块。

### 3.2 最小已知真值实验

先做一个保守的低维系统，不以有利的复杂非线性开局：

1. 隐状态为 4 维，由两个 2 维局部稳定动态和一个弱耦合项组成。用 3 种局部 transition primitives 和 2 种耦合机制组成至少 8 个系统；稳定性边界在生成前固定。
2. 构造 4 个线性观测 view，包含可逆混合、部分传感器选择和不同观测噪声；所有 view 在总体上可观测。模型获得 view ID／时间／mask；观测矩阵不提供给学习模型，真值 oracle 保留它。各模型得到相同的少量同步多 view 训练片段，不给任何模型特权。
3. 以棋盘式留出 view–system 组合：每个 view、每个动态 primitive 都在训练中出现，但一部分组合从未出现。测试同时报告 seen 组合、held-out 组合、全新动态 primitive 三层；最后一层是“复用假设不成立”的压力测试。
4. 每格约 512 条、长度 128 的轨迹，先 2 个 seeds 做筛选，进入确认后固定配置跑 5 个 seeds。初始模型控制在约 0.3–2M 参数；序列和模型规模是预算建议，未测运行时间。

线性层便于 Kalman oracle 给出精确条件 law。它也故意让普通 EM／Kalman 模型有取胜机会；若此层完全被它解释，不得把它改名为“太简单不体现我们优势”。只有存在可归因的交互，才进入低维 nonlinear emission／nonlinear dynamics 扩展，用足够粒子及收敛审计代替精确 oracle。非线性不是为挽救失败而随手调参。

### 3.3 关键对照和裁决

核心是同一训练数据、同一 latent capacity、同一损失、同一推断器的 2×2：

| 观测接口 | 动态模型 | 用途 |
|---|---|---|
| 单体 | 单体 | 00 |
| view 模块化 | 单体 | 10 |
| 单体 | primitive 模块化 | 01 |
| view 模块化 | primitive 模块化 | 11 |

再加入完整控制：NCDSSM 风格模型、modular NRI 风格模型、按 view／system 条件化的普通 hierarchical LGSSM/SSM（含同样模块共享及联合参数估计的强版本）、FlowNP、GRU-mixture；LIFTS 经父流程资格确认后加入。给普通 SSM 同样的噪声建模、足够状态维度、联合推断和调参机会。按参数量匹配一次、按训练／推断计算匹配一次，分别报告，不能假装两种预算天然相同。[1–4]

以 held-out 组合的未来 observation joint NLL 为主终点（可算时），sample-based 模型报告 energy score 和预先固定的 trajectory projection scores；辅以状态 oracle 相对 excess risk 和适应曲线。潜变量坐标距离不做主终点。

若 NLL 越小越好，交互量定义为

$$I=(R_{10}-R_{11})-(R_{00}-R_{01}).$$

I>0 表示增加动态模块化，在观测也模块化时收益更大。要看跨 seeds 的区间和 held-out 相对 seen 的差异，不能只看“11 最好”。预先按工程价值定义最小实用收益，例如 held-out excess risk 相对最强普通控制下降 10%，并在两个观测强度上方向一致；10% 是筛选门槛，不是理论常数。

**明确否证：**11 只有 seen 改善；交互与 0 相容；收益只来自参数减少或更多调参；ordinary shared-parameter SSM 吸收全部收益；去除组合结构后仍有同样收益。任何一项成立，都不能宣布“可复用状态机制已成立”。不要通过选一个 site/seed、缩小测试任务或调噪声阈值救回。

### 3.4 把信息、转移和推断分开

这里只作诊断性替换，不把真值 state 当作新训练目标：

- **推断误差：**固定一个已经学得的 generative model，比较其原生 inference 与相同模型下更强、收敛检查过的 smoothing/filtering。两者预测差是 inference 近似问题；更强推断是允许的普通修复。
- **动态误差：**在 oracle 真状态上学习／评估 matched transition family，比较其多步 observation law 与真 transition 所诱导 law。此项只回答容量与估计问题，不要求 learned latent 与真坐标对齐。
- **表示信息：**固定各模型的实际预测 state，训练相同预算的 distributional diagnostic decoder，并比较加旧历史后 proper-score 的增益；同时做 decoder-capacity sweep。历史仍有增益只能证伪当前表示＋decoder 的充分性；零增益不能证明充分。此诊断早有预测状态文献先例，不是贡献。[7]

这三项一般不是可直接相加的误差分解。尤其不能把一个弱 probe 的误差命名为信息损失，也不能用真状态替换后预测变好就宣称 learned state “没有物理可识别性”。

### 3.5 后续真实跨域证据与失效范围

最低真实证据要求：同一实现和冻结选择规则，在至少 3 个不同领域的多变量序列上重复“传感器 view × 系统／群体”留出；例如环境传感器网络、能耗／工业系统、可穿戴运动。先使用真实同时记录的传感器子集和自然群体划分；任意线性混合作为额外压力测试，不能充当全部实际验证。可采用跨域预训练后有限适应，也可分别训练同一框架；两种结论必须分开。

除了预测／缺失重建，还报告固定动态部分时接入新 view 的样本—误差曲线和适应成本。如果接入 view 必须重新训练全部 backbone，复用故事就削弱。若三个领域只有一个存在模块重组规律，应该把适用范围收窄到存在可复用局部规律的系统；不能称普遍 TS world model 的改进。真实数据没有共同物理 state 时，主张只能是可复用预测接口，不是跨域共享真实状态语义。

**最近重叠挑战：**NCDSSM＋modular NRI 的直接组合、普通 mixed-effects／hierarchical SSM、共享 encoder/decoder 的多域预测器，可能已经是完整解。A 的开发价值来自是否能确认上述有限数据交互；不是靠说这些方法“原本不叫 world model”取得新颖性。

## 4. 候选 B：粗到细路径 law 与多层路径查询

### 4.1 任务与精确机制

一个时间序列生成框架除了输出固定 horizon 轨迹，还可以回答一组由用户明确给定的路径函数：累计量、区间平均、占用时间、越界概率等。对多个条件／多个查询反复调用时，完整细路径的 Monte Carlo 成本可能成为瓶颈。问题是能否学习一个**可复用的路径生成与查询接口**，让计算分配随函数所需时间尺度变化。

待证组合包含两项：

- 两层条件生成 law：先由历史证据生成低分辨率路径 c，再生成细节 d；用固定可逆 lifting／wavelet 变换回到细路径 y，保证细路径重新聚合回 c。粗生成器可单独调用，不必运行细节网络。
- 在同一已拟合条件 law 下，估计 E[f(y)] 时使用多层 telescoping estimator，并在 validation 上根据各函数层间差的方差和成本分配样本。训练一次生成 law；新函数不重新拟合动态 backbone。

例如：

$$\mathbb E[f(Y_{\rm fine})]
=\mathbb E[f(P C)]
 +\mathbb E[f(Y_{\rm fine})-f(P C)],$$

其中 C 与 Y_fine 在第二项来自同一个 joint sampler，P 是事先固定的 coarse-to-fine reconstruction。粗样本独立批次用于第一项。**两项中的 C 边际必须完全一致**；仅声称两个模型“接近一致”不够，否则 telescoping 有偏。式子本身就是 MLMC／control variate，不是新算子。

mr-Diff 已有粗到细条件生成；MLMC diffusion 已有多精度生成器耦合。[5][6] WaveletDiff 也已经直接在 wavelet coefficients 上做跨层联合 denoising，并用 inverse transform 重建；所以“wavelet 表示保证可逆聚合”不是候选的新颖部分。[8] 候选差别是把**真实时间分辨率中可单独、便宜调用的 shared coarse marginal**作为可查询条件 law 的结构部分，再检验其与 query-dependent allocation 的交互。WaveletDiff 的全文式 (8) 在每个 reverse step 让各层依赖所有层；仅截取其 coarse coefficients 不自动省掉细层 denoising 成本。这个工程区别需要实测而非推断成性能优势。候选不是对若干输出 quantile 做 rank shuffle，也不依赖证明某个现有生成器内部不一致。

**待证假设 B：**共享粗路径使某些实际函数的 fine-minus-coarse correction 方差显著低于 fine 函数方差，且 coarse 分支足够便宜，因而获得实质 Monte Carlo 加速。层次生成结构本身不保证此点；对高频越界函数可能完全无效。

### 4.2 最小已知真值实验：先分清采样加速与模型变好

1. 先用已知 2–4 维 stochastic system：线性 Gaussian 作为校准项，弱非线性双稳态或非线性局部耦合作为主项；固定离散时间网格，避免把数值 SDE 离散误差混入学习收益。
2. 历史长度 64、未来长度 128，粗层下采样 4 倍。生成样本数约 20k，学习模型总量 1–3M 参数，先 2 seeds；确认阶段用 5 seeds 和 held-out 系统参数。与 A 可以共用数据／oracle，无需现在新增大规模训练。
3. 函数集合在训练前固定：累计量、平均量、max 的平滑版本、占用时间、真实离散越界 indicator。阈值依训练分位数固定，并把越界函数作为预期可能失败的对照。不能只展示能由 coarse 精确计算的函数。
4. 有两个真值基准：已知系统的足够高精度 oracle，用于估计 model bias；同一 fitted generator 的高样本参考，用于估计 Monte Carlo 误差。两者不可混用。

需要的完整对照：ordinary state-space sampler＋常规 MC／MLMC（可用时）、强 GRU-mixture＋批量 MC、native FlowNP＋MC、LIFTS＋MC、原生 mr-Diff＋MC，以及按 diffusion integration 精度构造层次的现有 MLMC 方案。[3–6] 另以 WaveletDiff 的 joint multilevel architecture 做无条件生成结构控制；若将其改为历史条件版，必须公开该适配并给它同等训练预算，不能称原生结果。[8] 若一个便宜 GRU-mixture 直接以更低成本取得同等真实误差，候选 B 没有实用胜点。

2×2 主比较为“共享粗 marginal 的路径参数化／matched flat sampler”×“常规 MC／MLMC”。flat 的 MLMC 可使用预先固定的标准 noise coupling／生成步数层次，不能只给它 independent pairing。训练成本、样本生成成本和 pilot allocation 成本全部入账。

### 4.3 裁决规则

主图是每个函数的 total RMSE／校准误差对实测 GPU wall-clock 的 Pareto 曲线，至少包含 3 个预算；同步报告 fine joint energy score 和模型偏差，防止以更差模型换便宜估计。

预先把确认门槛写成：在至少两个**非平凡**函数族上，达到相同真实误差所需的端到端时间减少至少 2 倍，并优于最强普通 sampler／MLMC 控制；另外至少一个 discontinuous 函数公开失败或成功，不藏掉。2 倍是实践门槛，不是文献保证。

对 B 最关键的机制测量是

$$\operatorname{Var}[f(Y_{\rm fine})-f(P C)] \quad\text{和}\quad
C_{\rm coarse}/C_{\rm fine},$$

而不是单看 coarse 与 fine 的均值相近。理论允许在这些量有利时加速；有限网络是否实现有利量必须实测。

**明确否证：**不运行细分支也同样准确（只是简单 coarse readout）；额外 overhead 吃掉 variance 收益；现有 MLMC coupling 同样好；训练 bias 成为主误差；只能对平均量加速；强便宜 ordinary generative model 支配 Pareto。出现这些结果就停止，不把内部一致性升格为实际效用。

### 4.4 后续跨域证据与最近重叠挑战

在至少三个领域的一组条件窗口上，用同一训练／allocation 规则同时测试 prediction、gap imputation 后的 path functional estimation，并在独立真实未来上检验预先固定的 event Brier score、累计量误差及覆盖率。单个真实样本不能给出该窗口的精确 event probability；因此校准／Brier 要跨独立窗口汇总，不能以“真实条件概率 RMSE”伪装它。

这个方向的最大重叠是 mr-Diff／wavelet hierarchy 与 generative MLMC 的直接组合。若不比这个直接组合更强，仍可能形成一个有证据的实用配方，但不足以宣称新生成机制。更值得开发的点是：一个有相同边际、可单独调用粗层、可复用条件状态的**完整条件路径 law**，在跨域多函数查询中确实比现成生成框架省计算；仅加一项 consistency loss 不满足这一点。

## 5. 本轮决策与开放风险

**建议先做 A 的 known-truth 2×2，B 保留为真正不同的备选。** A 的最小实验能直接回答状态是否作为复用接口增加价值，而且允许精确 Kalman／普通 SSM 解释结果。B 可以共用已知随机系统，验证“模拟与查询成本”这一替代贡献，但不应在用户还未接受这些查询是实际需求前把整条项目强制转到罕见事件或数值金融。

两条路线都不能从“原料已知”直接判死，也不能从“组合未搜到一模一样标题”宣布新颖。开发门槛应是：先有相同预算下的交互证据，再有完整最近方法比较，再有三个真实领域的一致收益。没有通过前，项目状态只是候选机制，不能写作已验证框架。

对原研究问题的回答：

- RQ1：已有方法提供部分机制与完整条件预测能力；本次来源没有直接建立 A 的跨 view–mechanism 有限样本交互，也没有直接建立 B 的时间路径层次在多域函数查询中的端到端收益。该结论是本次检索范围内的证据缺口，不是全球优先权判断。
- RQ2：A 的线性 Gaussian 交叉设计，加上同一模型的 inference 替换和真状态 transition 诊断，可定位收益来自哪里；B 另外把模型偏差与 Monte Carlo 误差分开。二者均无需物理 latent 的唯一识别。
- RQ3：A 有一个可执行的强否证试验；B 有不同的机制和价值标准。现阶段没有足够证据推荐新增宏大架构，也没有理由只因经典组成部分存在就拒绝一次小型、严格的交互筛选。

## 6. 已核验一手引用与证据边界

以下六项为决定性来源，标题、作者和方法陈述均已在官方全文／会议页核对；没有运行其代码或复现实验。

1. **Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh. Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series. ICML 2023.** [官方论文页](https://proceedings.mlr.press/v202/ansari23a.html)，[全文 §§3.1–3.2](https://proceedings.mlr.press/v202/ansari23a/ansari23a.pdf)。证据：auxiliary recognition 与 latent dynamics 分开；LTI、locally-linear、nonlinear 参数化及连续离散推断。没有据此声称完成未知 view×机制的组合泛化。
2. **Ferran Alet, Erica Weng, Tomás Lozano-Pérez, Leslie Pack Kaelbling. Neural Relational Inference with Fast Modular Meta-learning. NeurIPS 2019.** [官方论文页](https://papers.neurips.cc/paper_files/paper/2019/hash/b294504229c668e750dfcc4ea9617f0a-Abstract.html)，[全文](https://openreview.net/pdf?id=BylKFBreLS)。证据：动态模块重组、关系推断的 meta-learning、短观测新系统预测；不将其对象／关系设定等同任意异质传感器时间序列。
3. **Sanyou Wu et al. Latent-driven Imputation and Forecasting for partially observed Time Series (LIFTS). arXiv:2610.03916v1, 2026.** [全文](https://arxiv.org/html/2610.03916v1)。证据：joint measurement/mask latent model、forward–backward tasks、structural-subclass identification；附录 S6.1 明确 synthetic encoder-calibration 没有系统收益而后续设 λ_h=0。其理论与实测结论均不外推至任意观测变换或所有神经训练结果。
4. **Hussen Abu Hamad, Dan Rosenbaum. Flow Matching Neural Processes. NeurIPS 2025 / arXiv:2512.23853v1.** [全文](https://arxiv.org/html/2512.23853v1)，[作者代码](https://github.com/danrsm/flowNP)。证据：集合 context/target 条件联合生成，GP、image、ERA5 实验；没有把 NP 内部一致性限制当作实际预测质量失败的证据。
5. **Lifeng Shen, Weiyu Chen, James T. Kwok. Multi-Resolution Diffusion Models for Time Series Forecasting. ICLR 2024.** [官方论文页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d64740dd69bcc90ba225a182984b81ba-Abstract-Conference.html)，[全文](https://openreview.net/pdf/fa6b63dce52e32160942f666f91e9ed633304dca.pdf)。证据：seasonal–trend 的 fine-to-coarse forward decomposition 和 coarse-to-fine conditional denoising；没有从其 forecast 结果推出 MLMC 的查询方差性质。
6. **Abdul-Lateef Haji-Ali, Marcelo Pereyra, Luke Shaw, Konstantinos Zygalakis. Bayesian computation with generative diffusion models by Multilevel Monte Carlo. arXiv:2409.15511v2, 2025.** [全文 §§2.3–2.4](https://arxiv.org/html/2409.15511v2)，[作者代码](https://github.com/lshaw8317/MLMCforDMs)。证据：不同采样精度的耦合、allocation 和 imaging inverse-problem 数值收益。这里只使用它限制“MLMC＋生成器”的新颖性，不把 imaging 加速数字转成时间序列收益。期刊版 DOI [10.1098/rsta.2024.0333](https://doi.org/10.1098/rsta.2024.0333) 的作者排序与此 arXiv 版不同；本报告固定引用 arXiv 版本。

边界／近邻补充来源：

7. **Yongchao Huang. Your Probabilistic JEPA Is Secretly a Hidden Markov Model: A State-Space Interpretation of Joint-Embedding Predictive Learning. arXiv:2608.13621v1, 2026.** [全文 §§5–7](https://arxiv.org/html/2608.13621v1)。证据：HMM 对应的分层条件、MCJEPA transition composition、predictive sufficiency/residual diagnostics、受控 HMM 实验。它不是 A 或 B 的完整工程实现，也不是“有限数据下任何状态组合都无效”的定理。
8. **Yu-Hsiang Wang, Olgica Milenkovic. WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation. arXiv:2510.11839v2, 2025.** [全文 §§3.1–3.2](https://arxiv.org/html/2510.11839v2)。证据：wavelet coefficient generation、可逆重建和跨层 joint reverse model。其论文的 generation 结果不自动给出条件查询／可单独调用 cheap coarse marginal 的收益；后者是本候选必须另外验证的条件。
