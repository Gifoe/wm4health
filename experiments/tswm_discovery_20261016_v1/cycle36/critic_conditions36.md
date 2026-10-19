# C36 独立批评视角：外部未来预测的不确定性，究竟还留下什么学习问题

日期：2026-10-17。依据冻结的 `RESEARCH_BRIEF.md`；主窗口为 2026-04-17—2026-10-17。以下是 intake 判断和可证伪协议，不是已运行结果，也不批准新方向。未训练、未启动服务、未下载数据或模型、未读取或执行 cycle6 launcher；只写本文件。

## 1. 检索范围与证据强度

采用 deep-research 的先宽后窄及原文核验方法。本视角先分别检索 `future covariates / forecasted covariates / covariate uncertainty`、`joint probabilistic forecasting / copula / marginalization`、`weather ensemble / calibration / downstream load`、`uncertain-input GP / state-control covariance`，再按命中的模型名、论文名和数据提供方收窄。检索覆盖了近期 TSFM 评测、PV 结构研究、概率状态模型、联合分布、天气后处理和原始数据合同。论文身份、日期与具体主张分开核查；只以原论文全文、作者代码及原始数据提供方作为判断依据。

这里的“已覆盖”表示某个数学操作、输入范式或实验证据已经存在，不表示任何有限预算模型都达到最优。反过来，一项工作只报告 synthetic noise 或只有 marginal quantiles，也不能被扩大解释成已经解决真实 vintage 下的 joint uncertainty。部分作者代码仅有链接、本轮未执行；公开可访问不等于可直接复现实验。

## 2. 近期最接近的普通方法与实验，分别覆盖了什么

| 工作与日期 | 全文位置、已核查内容 | 对 C36 的约束与公平边界 |
|---|---|---|
| [Kaljevic, Arzola & Zhang, *Benchmarking Time Series Foundation Models for Load Forecasting Under Covariate Uncertainty*, 2026-10-05](https://arxiv.org/html/2610.07232v1) | §II 的问题定义直接学习 target 在历史与 future forecasts 条件下的点值或条件分布；§II-C 用 train ARIMA 和 validation residual scale 生成独立高斯扰动；§III-B 比较 Informer、TimesNet、Chronos-2 的噪声退化。 | “future covariate 不是精确真值”已是近期明确问题。其扰动不是实际发布的 weather vintage；作者承认缺少真实误差的 bias、temporal dependence、non-Gaussianity。§II-D 允许各模型按能力使用不同输入/上下文，未给出每个噪声部署合同对应的公平重训练证据，因此不能从其排序推出强 direct 在健康训练后仍有结构性缺陷。 |
| [Zhang et al., *Rethinking the use of deep learning methods for photovoltaic power forecasting*, Nature Communications, 2026-06-12](https://www.nature.com/articles/s41467-026-73817-3) | Results 的 Problem definition、Dataset configuration、Necessity of encoder-decoder mechanism，以及 Fig. 2；Cross-Unet 把历史状态与 forward weather forecasts 分流，通过 channel dependence 和 hierarchical cross-attention 融合。 | 已覆盖“历史观测和有偏未来预测异质，需要专门交互结构”，不能只与把两者胡乱拼接的弱基线比。作者区分 PVOD 在 >2day 的 forecast-skill 高估、future satellite upper bound、Earth-2 daily-issued 构造。未见显式 ensemble-distribution-to-transition 学习；daily issuance 叙述也不是历史公开下载时间的独立证明。 |
| [Chen, Zhao & Chen, *Robustness of Deep Learning Models for PV Power Forecasting under NWP Forecast Errors*, 2026-07-14](https://arxiv.org/html/2607.12954v1) | §2.1–2.3、§3、Data Availability；用 pvlib virtual PV、HRRR residual calibration、AR 型相关扰动、sky-dependent heteroscedasticity 与 Erbs radiation consistency，比较含 LightGBM、GRU、N-HITS、PatchTST 的六类模型。 | “不要只做 iid input noise”“噪声下会向历史/physical priors 重新分配依赖”不是新发现。它是受控模拟，processed data 需向作者索取；不能替代真实厂站与真实 forecast vintages。单 seed/common settings 也不能证明任何 architecture 必然更鲁棒。 |
| [Pan & Ezzat, *Evaluating Time Series Foundation Models for Electricity Price Forecasting*, 2026-07-02](https://arxiv.org/html/2607.02623v1) | §2.1 Eq. (1) 用 issuance-time information set；§2.2–2.4 纳入 Chronos-2、TFT、TiDE、TSMixer 与 LEAR/DNN/CING-LEAR；§3.1–3.3 考察简单组合、tails 与 spikes。 | 直接输入 forecast 信息、task-specific ordinary methods、简单组合均须认真纳入。GEFCom2014-P 与 GridStatus2025 的可用信息合同不同；原文的总体 “available information” 定义本身不证明每个 GridStatus covariate 的逐版本发布时间。不能用单纯移除 covariates 的 univariate TSFM 代表最强普通方法。 |
| [Ye et al., *SPECTRA*, 2026-07-22](https://arxiv.org/html/2607.20587v1) | §II-A Eq. (1)、§III 的 MTPD/ECS/STSSE/SBE、§V；exogenous context 进入分离的 backbone/residual streams，SSM refine 后输出 ordered quantiles。 | “context 对中心和波动分别起作用”“state-space + probabilistic head”已是完整近期邻近架构。它把 historical/known future exogenous sequence 作为张量，结论仍把 imperfect exogenous information 列为后续工作；未验证其已处理真实 forecast distribution。 |
| [Klötergens et al., *Probabilistic Circuits for Irregular Multivariate Time Series Forecasting* / CircuITS, 2026-04-30](https://arxiv.org/html/2604.27814v1) | §3.1、§4.1–4.4、Appendix A；context-conditioned joint channel/query distribution，recursive sum-product、Gaussian-copula leaves，证明 marginalization consistency；§1 脚注给[作者匿名代码](https://anonymous.4open.science/r/CircuITS-for-IMTS-2CD1)。 | target 与未来未知 covariates 可以作为 joint random variables 建模与边缘化；“联合分布不存在普通实现”不成立。它的四个实测数据是 irregular observations，不是外部 forecast-vintage 任务。将 forecast tensor 纳入 context 是可实现适配，不应虚称原文已验证。 |
| [Vercellino, *WIRED*, 2026-08-13](https://arxiv.org/html/2608.12998v1) | §6.2、§7 Tables 3–6、§9–10；marginal expert mixtures + Gaussian/Student-t copula，显式 equal-weight 与 Gaussian-copula bootstrap 基线。 | coherent scenarios 与 adaptive mixtures 已有现成邻近实现；其自身试验里简单组合经常优于复杂权重。这是对“学了更多 uncertainty modules 就会更好”的实际反例。数据范围小、主要 synthetic，不能把其负结果泛化为所有学习组合无效。 |
| [Chintalapudi et al., *Pollutant Adaptive Forecasting…Bayesian Dynamic Regression and Conformal Prediction*, 2026-08-17](https://link.springer.com/article/10.1007/s40825-026-00308-8) | Methods 的 Eq. (21)–(24)、§4–5；origin-wise internally forecast covariates、delta-method variance propagation 与 conformal residual calibration。 | “input variance 加进 uncertainty 再校准”是现成统计基线。它假定不同 covariate forecast errors 独立，明确不覆盖结构性偏差和 joint errors；annual、很小样本与事后版本资料也不是历史公开信息验证。普通校准改善 coverage，并不识别 latent state。 |

上述来源均在主窗口内；其中预印本与已发表论文的地位不混同。未因标题带 world model、state-space 或 uncertainty 就认定它已解决本题；也未因为它没有这几个词就排除它。

## 3. 旧基础必须保留：它们关闭的是哪些具体主张

| 较早来源 | 精确位置与可用事实 | 不能外推的部分 |
|---|---|---|
| [Chronos-2, 2025-10-17 v1](https://arxiv.org/html/2510.15821v1)；[作者 pipeline 输入合同](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py) | §3 定义 covariate-conditioned probabilistic forecast；§5.3/Fig. 6 的 EPF-DE 输入明确包含 day-ahead load、solar/wind generation **forecasts**。pipeline `predict` 支持 past/future covariates、multivariate grouped series。 | “future_covariates”这个 API 名称不等于只能塞无误差真值。该接口本身不接受一个带显式概率语义的天气 ensemble 分布；实测使用外部预测也不证明有 ensemble-level calibration 或跨版本适应保证。 |
| [Ludwig, Arora & Taylor, accepted manuscript 2022；JORS 74(3)](https://users.ox.ac.uk/~mast0315/LoadEnsembles_JORS.pdf)；[原始元数据](https://ora.ox.ac.uk/objects/uuid%3Ac4d76c24-d3f7-4119-92e7-9abbb10e3658) | §2.2 用 ECMWF 51-member + HRES；§3.1 EMOS、§3.2 ECC 保留跨变量/时间 rank dependence；§4 将校准后的 multivariate weather draws 经 load model 传播，1000 draws，另建 load residual dynamics；§5 比较 raw、post-processed、actual、no weather。 | 输入校准、coherent trajectories、非线性 weather-to-load 转换及 downstream density 已构成完整 pipeline。它只做 GB midday、1–6day、特定回归，不证明可重用 neural state 最优；原文结果也不支持 calibration 在所有 lead/metric 必胜。 |
| [Girard et al., NIPS 2002 / volume published 2003](https://proceedings.neurips.cc/paper/2002/file/f3ac63c91272f19ce97c7397825cc15f-Paper.pdf) | §3 Eq. (5) 对 random input 积分，§3.1 用 Gaussian approximation，后续应用于多步 time-series regressor uncertainty。 | 单纯“条件是分布而非点，经过非线性 transition 积分”不能作为新机制；但 Gaussian approximate propagation 没有消灭高维、多模态和有限预算的误差。 |
| [Deisenroth & Rasmussen, PILCO, 2011](https://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf) | §2.2 Eq. (9)–(23) 传播 joint state-control uncertainty，包含 state/control 与 state/increment cross-covariances，Gaussian moment matching。 | 不能把“保留状态与条件的相关性”本身说成首次。PILCO 的 GP/Gaussian/control设定并不提供真实外部 forecast tensor 的一键最优解；不应以无限算力 GP oracle 当否定所有有限模型的标准。 |
| [Rangapuram et al., DeepState, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/5cf68969fb67aa6082363a6d4e6468e2-Paper.pdf) | §3 Eq. (1) 预测 future trajectories、假设 prediction-range covariates 已给；§4 Eq. (5)–(6)、Fig. 1 学习共享 covariate-to-SSM-parameters mapping，边缘化 latent state。 | state、transition、observation 和 covariate-driven dynamics 早有联合学习先例。DeepState 的 fixed future-input contract 没有显式 forecast measurement distribution，不能直接说它原生解决外部天气不确定性。 |
| [Ashok et al., TACTiS-2, 2023/ICLR 2024](https://arxiv.org/html/2310.01327v2) | §2 Eq. (1) arbitrary observed/missing mask 条件下的 joint density；§3 two-stage copula curriculum，§4 dual encoder。 | 把目标/协变量作为 joint missing variables 的路线是完整方法，非空白。其 non-stochastic context/covariates 需要显式加入 forecast vintage，不能让未知真实天气冒充 observed token。官方 Appendix C.3 的大规模 HPO预算也不应要求本轮照搬。 |

这组基础不构成 blanket old-component veto。它们只要求把具体改动与完整邻近方法区分清楚：一个新条件编码/有限预算交互可以值得测，但“积分、校准、joint uncertainty、state-input covariance”这些概念本身没有新颖性。

## 4. 同一信息集下，真正需要推翻的普通模型是什么

令 `o` 为 forecast origin；`H_o` 为在 `o` 已可获得的历史，`F_o` 为当时发布的完整 forecast packet（包括变量、lead、member轨迹、source/version/age），`I_o=(H_o,F_o)`。`C^+` 是未来真实条件，`Y^+` 是预测目标。

**直接模型的合法目标是 `p(Y^+ | I_o)`。** 在历史训练 origins 中输入对应版本 `F_o`、按 proper score 或 quantile loss 学习，模型原则上可以直接学会 forecast bias、spread-skill关系、历史与预测的可靠性切换以及最终条件不确定性。它不必先恢复真实未来天气。因而，把 `F_o` 当输入不等于“假定 `F_o=C^+`”；经典 errors-in-variables 对物理斜率的 attenuation 也不自动证明该预测条件分布有误。

常见分解为：

`p(Y^+ | H_o,F_o) = ∫ p(Y^+ | H_o,C^+,F_o) p(C^+ | H_o,F_o) dC^+`。

若删除第一个因子中的 `F_o`，另需 `Y^+ ⟂ F_o | (H_o,C^+)` 这一预测充分性假设。没有因果宣称也要讲清该假设：weather provider 的 errors、分辨率或更新时间可能提示 target-relevant latent regime。直接模型可使用这份提示，简化后的积分模型反而会丢信息。若积分含更多未知参数与更高 variance，有限样本下它也可能更差。

据此，下面几类“胜利”并不能留下 state/transition 的学习残差：

1. **truth-train / forecast-test 的胜利**：用真实未来条件训练基线、只在 test 喂误差条件，再让新方法在 vintage-train 上学习。先修正 input contract，不计方法收益。
2. **额外信息的胜利**：新方法拿 ensemble、source id、forecast age、过去已成熟误差，而 direct 只拿均值。所有能在 `o` 取得的字段必须给相同。
3. **普通推断的胜利**：对 `K` 个 condition scenarios 各跑一次普通 predictor，再取 mixture。该操作就是 uncertain-input integration；若没更改 representation/transition learning，只能主张预算化实现或应用结果。
4. **校准与误差项的胜利**：只加输出方差、quantile recalibration、EMOS/ECC、ensemble averaging。需要和相同校准后的 direct/joint 方法比，不能把 coverage 修好当成新状态理论。
5. **错误相关结构的胜利**：对照逐时独立抽样，新方法保留 member的完整轨迹。coherence 是必须补齐的基线，不是新机制；Ludwig et al. 已给完整先例。
6. **不健康训练的胜利**：epoch1 checkpoint、训不动、loss scales 不平衡、某一方法独占额外数据/训练passes。应先使主要对照达到合理收敛，再比较 finite-budget 表现。

对 uncertainty-aware **训练**，还应区分 `-log E_C[pθ(Y|C)]` 与 `E_C[-log pθ(Y|C)]`：二者不是相同目标，后者倾向要求每个情景都解释同一个目标。仅修正这一 Jensen/augmentation 错配，不等于找到新的 world-model learning mechanism。

## 5. 最小科学问题：可以测，但目前不能宣称已有残差

值得保留为诊断的问题只有下面这个有限命题：

> 在相同已发布 forecast packets、相同训练 origins 和健康训练下，coherent-condition-distribution 进入 compact latent transition 的结构偏置，是否比能读取相同 packet 的 direct joint predictor，以更少监督样本或更低 inference budget 获得更好的 target trajectory distribution？

它是一个样本效率/计算效率问题，不是表达能力不可能性、因果辨识或“直接条件模型不能处理 uncertainty”。要提升为方法方向，必须明确具体 transition interaction；仅写“分布编码 + 状态转移 + marginalization”仍是未定结构，并可能完全等同于已知 uncertain-regressor pipeline。

### 建议的最小证伪矩阵

数据合同过关后，先做一个任务、固定24/48step horizon、固定 chronological train/val/test。历史上下文可固定7天，训练期覆盖不少于两个完整季节；按**origin**拆分，所有 horizon 不能越 split 取label。第二个异质任务只在第一个有稳定信号后追加。下列三臂必须存在，再测一个候选臂；不需要把所有 classical families穷举到无限算力。

| 臂 | 输入与学习合同 | 要排除的解释 |
|---|---|---|
| D：强 direct joint | 小型 encoder-decoder/TCN/MLP，读取 history + packet；对于 members，用共享 trajectory encoder + permutation-invariant pooling，而非只取均值；输出有 temporal dependence 的 joint Gaussian/mixture或同一共享copula；训练部署均用真实 vintage。 | 普通 supervised conditional learning 已足够。Cross-Unet式 dual streams 是可用设计；可另报 task-trained LightGBM 的边际分数作为廉价检查。 |
| J：joint target-condition | 共用容量/encoder，建 `p(Y^+,C^+|I_o)`，对 observation可见的训练变量做 likelihood；预测只取 target marginal。未知 future condition 不当作 test truth输入。 | 收益只来自 joint supervision/auxiliary task，而非新 state interaction。条件truth若用于辅助监督，D也有相同辅助loss变体。 |
| M：calibrated coherent marginalization | 相同历史state/ordinary predictor，`q(C^+|I_o)` 在 train-fit/val选定；同一批 coherent trajectories，`K=8/16` 推理，保留 downstream residual分布；同时允许 ordinary model 使用原始packet。 | 收益只是 calibration、trajectory sampling或 nonlinear integration。所有臂共享 calibration训练资料和同一packet。 |
| T：候选 transition | 在 D/M 相同的条件编码与联合输出族上，只改变一个可说清楚的 latent transition interaction。 | 若混入新输出头、额外metadata、不同loss、额外passes，无法识别 transition 的贡献。 |

具体冻结约束：每臂约1–3M参数；相同 history、训练origins、有效target数、max训练passes和预设GPU-hour上限；3 seeds；不进行 per-arm 大规模 HPO，只给相同小学习率网格与validation选择机会。记录 wall-clock 与 peak memory；`K` 次推理算入budget，可在共同latency下比较，也可单独报告accuracy-cost曲线，不能隐去额外MC成本。单5090可先排程12个小模型fits，但本轮未测runtime，不承诺小时数。

主要分数用 target marginal CRPS加 joint energy/variogram；若所有臂同为有密度模型，报告 NLL。另预先指定24h sum与 ramp/max事件的proper score；只比较MAE/coverage无法检验trajectory uncertainty。按days或weeks做paired block bootstrap，block长度在validation固定；报告seed与origin两层变异。未来真实forecast error分箱只做事后诊断，deployment gating须用origin可计算的spread、age、past matured errors。

**负例与关闭规则**：若 D 或 J 在共同预算下与 T 无稳定差别，或 T 的改善在同一校准/相同完整packet给D后消失，关闭“需要新 uncertainty-conditioned state/transition”这一具体命题。若 M 已解释所有收益，则归类为普通 uncertain-input integration；若只存在低样本/低预算优势，主张限于该区间。预先可设实际效应阈值如主CRPS相对1%，但阈值是协议选择，不是现有证据。任一任务失败必须保留；不得复用 C35 的solar/Jena结果来证明本题。

## 6. 公开 vintage 数据：有候选，但不能把 initialization 当 availability

| 原始提供方证据 | 当前核实到什么 | 本轮仍未过的合同 |
|---|---|---|
| [NYISO P-7A Weather Forecast](https://mis.nyiso.com/public/P-7Alist.htm)；[当前单日CSV链接](https://mis.nyiso.com/public/csv/lfweather/20261015lfweather.csv) | 官方索引有daily weather forecast文件与月度archive，索引跨度可见2008—2026；2026-10-15条目显示 `10/15/26 08:01 EDT`。这是比只看到valid_time更直接的发布线索。 | web不解析CSV，本轮未下载或核字段；索引的当前Last Updated不是每份历史成员的immutable first-release。需核CSV内部issue/version、修订规则、station聚合和DST；尚不能宣称严格 `available_at≤origin` 数据集已就绪。 |
| [NOAA NCEI GEFS 原始说明](https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast)；[NCEP GEFS页面](https://emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php/) | NOAA确认 operational GEFS公开云资料2017至今、00/06/12/18UTC cycles、完整forecast leads；NCEP分别链接operational、archive和reforecast，不能混淆。 | cycle是initialization；每个file需要creation/release或保守、可核的historical operational latency规则，且history target/weather也要有发布延迟。NCEI明确NODD并非官方archive，需manifest与缺失审计。不能把现在重传的对象LastModified当原发布时间。 |
| [NOAA GEFSv12 reforecast storage说明](https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf)；[PSL官方reforecast页面](https://www.psl.noaa.gov/forecasts/reforecast2/index.html) | 官方明确该产品不是当年实时预测archive，2000—2019是固定后来的系统生成的retrospective forecasts。 | 可做clean controlled hindcast，但不能称那些forecast packets在对应历史origin已经发布。真正operational test与hindcast training分层可研究，必须把version差异交给各臂相同处理。 |
| [GEFCom2014组织者原文](https://pierrepinson.com/docs/Hongetal2016.pdf)；[IEEE PES数据条目](https://ieee-pes-data-sharing.org/datasets/detail/0e87366e-2e91-4024-b658-43f6b22faa69) | 组织者介绍competition逐任务提供预测问题，原始数据包含energy targets和外部forecast信息；近期EPF论文仍采用它。 | competition release与实际NWP issuance不是同一个合同。本轮未核全量字段、每个NWP origin/lead和许可；不把第三方合并CSV的列名当真实vintage证据。 |

母线程提示的 Open-Meteo Single Runs / Historical Forecast 区别、OpenSTEF simulated availability疑点，与本视角的结论一致，但本文件不以未独立重读的材料充当已核实证据。这里保留明确门槛：必须建立 `(provider, source_version, run_init, first_release_or_verified_delay, origin, valid_time, lead, member)` manifest，检查每一行可用性；所有方法读取同一manifest。**目前本视角没有核实出已可直接开跑、逐版本公开发布时间完整的双任务数据包。**这属于数据合同未完成，不能拿来主张方法新颖。

另有四个常见信息陷阱：

- ERA5/reanalysis、事后修正负荷、最新网页覆盖后的forecast都可作离线label/verification，不能默认为origin时刻的历史输入；原点需留足观测延迟或使用已冻结历史版本。
- 不能在某个origin的多步rollout中接入之后发布的更新run；若研究rolling reforecast，所有方法必须同步得到新packet并将任务定义为同样的rolling更新。
- 日历是future-known；forecast tensor的数值是在origin已知，但对应真实天气未知。把这两种“known”分清即可，不必赋予因果或行动语义。
- 若使用trained TSFM或later weather model的retrospective output，预训练截止与目标时期污染风险要单列；它不等于numerical leakage，但会影响“当年可部署”的主张。

## 7. 独立结论

目前证据足以关闭三种泛化叙述：**“普通预测器只能把future covariates当真值”“联合条件分布或校准传播尚无完整先例”“只要把条件不确定性放进state/transition就有新机制”**。Chronos-2的实际forecast-as-input、Cross-Unet的异质流交互、Ludwig的calibration+coherent propagation、TACTiS-2/CircuITS的joint distributions、GP/PILCO的uncertain-input covariance分别构成具体反证。

证据还不支持关闭所有有限预算的条件表示或world-model学习研究。尚可检验的是：在相同信息与健康训练后，某个明确结构偏置是否提高样本/推理效率，并且收益不能被joint supervision、calibration或MC传播解释。该命题目前没有已测残差，也没有已冻结的真实vintage双任务合同。**建议保留上述四臂诊断作为证伪工具；本视角接受的新方向数为0。**
