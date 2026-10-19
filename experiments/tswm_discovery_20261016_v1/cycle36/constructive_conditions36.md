# C36 建设性相邻机制：条件、最小架构与关闭边界

检索日：2026-10-17。责任范围：独立建设性视角；未读其他视角的 note，合并前仅交换重要 overlap 与数据线索。主窗口为 2026-04-17 至 2026-10-17；早期基础工作逐项标明。只读原论文、作者代码、官方 proceedings 和数据提供方。未训练、未启动服务器、未批量下载、未接触 cycle6。唯一写入为本文件。

## 结论与研究问题

**当前没有可以接受的新候选。** “外部未来预测不等于真值”确实是输入合同的重要区别，但从不确定外生输入、预测修订进入状态，到对预测分布积分，均有直接先例。新增 forecast-observation channel、冻结 target dynamics 后仅校准外部预测，在模型条件相同的情况下就是普通两阶段校准与积分；不能靠 latent rollout 改名形成方法差异。

仍有一个可提出、但尚未挣得立项的有限预算问题：**当外部未来误差为高维、相关、目标敏感，而推理只允许很少情景时，联合学习“供转移消费的不确定性低维表示”能否优于同信息的直接模型、普通校准加积分，以及固定/学习的情景压缩？** 这是有可能失败的统计和计算问题；不是“未知输入第一次进入 world model”。目前缺少压力条件证据、胜过最强普通对照的证据，以及完全核验的数据发行链。因此以下架构只是可供后续一击关闭的实验对象，不是被接受的 Idea。

本视角回答三个问题：哪些完整先例覆盖宽泛主张；还剩什么具体有限预算交互；什么公开协议和普通对照能推翻它。

## 1. 先宽后窄检索及最接近先例

宽检索覆盖：uncertain future covariates、latent/state-space forecasting、forecast evolution、weather calibration、learned quadrature/sigma points、scenario compression、2026 covariate-aware probabilistic forecasting。随后按 DSSMF、KITE、MMFE、Viking/GAMLSS、Cross-Unet、OpenSTEF/GEFCom 缩窄。搜索无命中不视为不存在；以下只列已核实且改变判断的原始来源。

| 来源及时间 | 精确位置与核实内容 | 对本题的约束与未覆盖部分 |
|---|---|---|
| **DSSMF，IJCAI 2019，早期基础**：Li, Yan, Yang, Jin, [Learning Interpretable Deep State Space Model for Probabilistic Time Series Forecasting](https://www.ijcai.org/Proceedings/2019/0402.pdf)；[作者 arXiv HTML](https://arxiv.org/html/2102.00397v1) | §2.1–2.4：神经 emission/transition、过滤推断、ARD、未来外生变量分布与 MC。§2.4 明确以 weather forecast 为例；§3.2 以未来真值为中心加入随 horizon 增长的 Gaussian uncertainty。§2.1 脚注报告把外生变量用于 transition 可能过拟合且性能变差。 | 直接覆盖“深状态 + 外生相关性 + 外生预测不确定性 + 多步采样”。其未来误差实验不是 archived vintage，transition 输入处理还存在文本/公式不完全一致处，复现时须查实现。不能把 arXiv 2021 当作首次发表年份。 |
| **MMFE 状态空间，2021，早期基础**：[de Chalendar & Glynn, On incorporating forecasts into linear state space model Markov decision processes](https://pmc.ncbi.nlm.nih.gov/articles/PMC8182152/) | Introduction 对信息 filtration 的区分；§4 静态 forecast 条件下非平稳转移；§5 forecast evolution 纳入扩展 Markov state；§6 混合静态/动态 forecast。 | 直接覆盖“forecast 是比被预测变量历史更丰富的信息”“forecast/revision 与状态联合演化”。不能把它限定为必须有 action 才相关；本题只复用它的预测建模部分。线性/一致性假设不自动解决有限数据神经学习。 |
| **UKF-L，2012，早期基础**：[Turner & Rasmussen, Model based learning of sigma points in unscented Kalman filtering](https://mlg.eng.cam.ac.uk/pub/pdf/TurRas12.pdf)；[期刊原页](https://doi.org/10.1016/j.neucom.2011.07.029) | §3 sigma point collapse；§4 Model based learning 将 sigma 点配置作为数据驱动学习问题，以预测 marginal likelihood 选择。 | “有限几个代表点应随非线性目标学习，而非照搬默认点”不是新原则。它不是外部 forecast-vintage TS 框架，也不能据此断言所有端到端低秩组合都已解决。 |
| **KITE，ICML 2026，7 月**：[PMLR 正式页](https://proceedings.mlr.press/v306/cheng26t.html)、[正式 PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/cheng26t/cheng26t.pdf)、[作者代码](https://github.com/decisionintelligence/KITE) | PDF §3.2 明确估计给定历史和未来 exogenous 的条件分布；§4.2 Eq.(4)–(9) 是 history-conditioned 中心、尺度和 learnable low-rank manifold noise；§4.3–4.4 是统计先验调制 attention 与 CFG；§5.1.1 说未来 exogenous 已知或高精度，Table 4 概率实验只用历史 exogenous。 | 是现代完整 generative covariate control，尤其堵住“历史中心 + 低秩不确定性 + 少步生成”组合的宽泛新颖性。并未原生证明真实 forecast-vintage uncertainty 的状态学习。作者 README 可执行入口为 `scripts/w_future/KITE.sh`；本次未执行。 |
| **SPECTRA，2026-07-22，预印本**：[原文](https://arxiv.org/html/2607.20587v1) | §III 的趋势/残差流、外生对齐、state-space refinement 与 quantile 输出；§V 明确把 imperfect exogenous information 的 robustness 留作未来工作。 | 不能把 output quantiles 自动等同于对未来外部预测分布建模；同时不能声称 probabilistic state-space energy framework 尚不存在。 |
| **de Vilmarest, Dumas, Thorey，2026-09-15，预印本**：[Combining Weather Forecast Aggregation and State-Space Models for Adaptive Probabilistic Electricity Load Forecasting](https://arxiv.org/pdf/2609.17000) | §III-A Eq.(1a–c)：GAM 系数状态、Viking 自适应 covariance；§III-B：GAMLSS residual variance；§IV：多个 weather providers 的聚合与 dispersion。 | 已经有真实 forecast quality、provider diversity、state adaptation 和 uncertainty 的完整工程组合。不是本题所需的神经 latent transition 新颖性证明；其 weather 数据公开可得性本次未证实。 |
| **Cross-Unet，Nature Communications，2026-06-12**：[Rethinking the use of deep learning methods for photovoltaic power forecasting](https://www.nature.com/articles/s41467-026-73817-3) | Results 的问题设定、Dataset configuration、Methods/Cross-Unet。明确讨论 history 与 forward-looking forecasts 的角色、偏差/不确定性，并用跨通道和跨注意力融合。 | 直接 predictor 已正面处理本题所说的信息。不能把它贬为假设所有未来输入为 truth。长于两天的原 NWP 构造和 satellite 上界有作者明确限度；它没有证明输入分布积分或新 latent-state 机制。 |
| **Kaljevic, Arzola, Zhang，2026-10-05，预印本**：[Benchmarking Time Series Foundation Models for Load Forecasting Under Covariate Uncertainty](https://arxiv.org/html/2610.07232v1) | §II-C：train 拟合 ARIMA，以 validation residual standard deviation 定标，加 Gaussian noise 控制输入误差。 | 是近窗口直接 robustness 对照，但不是现实 forecast-error covariance/vintage 的完整验证。其结果不能外推为所有操作误差都由 Gaussian augmentation 解决。 |
| **Pavirani 等，arXiv v1 2026-08-10，预印本**：[Control-Oriented Scenario Tree Construction through Reinforcement Learning](https://arxiv.org/html/2608.09335v1) | §2、§4：按下游任务学习 forecast fan 到小 scenario tree 的 assignment；§6–7 承认实验 forecast 正确指定、control 线性等限制。HTML 页首仍写 April 2024，因此这里只据 arXiv 版本日期纳入窗口，不推断此前发表状态。 | 给出建设性原则：小预算组合可以有真实价值，不应仅因组件旧就否定。但其目标是 control cost，不能把 control 改成 target scoring 就宣布获得新 TS-state 方法；仍须证明 transition 学习特有的收益。 |
| **VINTAGE-TS，2026-09-23，预印本**：[Ahmad, Time-Series Foundation Models That Understand Data Revisions](https://arxiv.org/html/2609.28576v1) | Abstract 与 evaluation 限制：validity intervals、delayed labels、first/mature release joint targets；真实 ALFRED 和 Chronos-2 实验未执行。 | 进一步堵住 provenance、revision target、delayed-label filtering 本身的新颖性。它也没有完成这里需要的现实数据学习比较。 |

身份与主张分别验证：上述会议/期刊身份以 proceedings/出版方为准，arXiv 项目只称预印本。KITE PDF 因 web PDF endpoint 的 content-type 问题，以内存 HTTP 读取和 pypdf 核验，未保存文件。没有采用第三方摘要关于“KITE 已处理 noisy future covariates”的推断。

## 2. 两种看起来合理、但目前没有独立机制的路线

### 2.1 分离外部 forecast observer，再更新 observer

设目标历史为 H_o，发行时可见外部预测为 F_o，未来实际外部轨迹为 U，目标为 Y。冻结 target dynamics 与 decoder 为 θ，仅用后来到达的外部标签更新 ψ，则部署分布为

\[
p_{\theta,\psi_o}(Y\mid H_o,F_o)
=\int p_\theta(Y\mid H_o,U)\,q_{\psi_o}(U\mid F_o,H_o)\,dU.
\]

这是建模假设下的因子分解，不是对真实世界必然成立的条件独立性断言。普通 baseline 可以同样实现 q 的 multilead bias/scale/covariance calibration、同样冻结 conditional target generator、同样使用 maturity 合格的外部标签、同样做 MC/QMC。若新方法只是把 U 换成 latent forcing、把同一个积分塞进 recurrent transition，优化目标和可用资源未改变。target labels 尚未到达这一资源限制真实，但同时惠及上述 baseline。

要获得不同问题，至少需要以下之一有证据成立：joint posterior 中目标状态与 forcing error 的相关性会改变预测，而普通校准 baseline 缺少的不是被人为不给它的历史；或者给定固定推理/训练预算，联合表示确实比 modular 表示更有效。当前都未证明。不能通过不让普通校准器看到 H_o、只给它逐 lead independent errors、或不给其更新预算来制造差异。

### 2.2 将 forecast spread/revision 用作 transition gate

它可能是有效实现，却不构成足够的新机制。把 mean、spread、age、revision、lead 输入小 GRU/MLP/GAM 后直接预测 conditional Y，已经允许学习不可靠时减弱 forecast、依靠 history。latent state 也无法自动使这一功能新颖。

真正要测的是“同信息、同预算下，**条件转移的共享表示**带来多少增量”，而非比较有 spread 的新网络与只有 mean 的旧网络。输出 quantiles、noise augmentation、输入积分和 residual gating 均应以普通控制存在。

## 3. 唯一保留的条件性研究探针

可以写成以下窄命题，暂不写论文贡献：

> 在有诚实发行记录的未来外部预测上，外部预测误差的高方差方向可能并非目标转移最敏感的方向。固定小情景预算时，按目标多步转移损失联合学习低维 forcing representation，是否比输入分布保真压缩更有效；这一增量是否必须经由共享 latent transition 才成立？

这不是一个已知不存在 prior 的命题；UKF-L、KITE 和任务相关 scenario reduction 都说明其主要设计原则已经存在。这里只留下一个可测 interaction：**目标相关压缩 × 多步转移学习 × 小 K**。若只测前两者之一，无法区分该交互与普通降维/调参。

成立所需的现实压力有三项，必须先用 train/validation 核验，不能测试后挑数据：

1. 外部 forecast residual 有显著跨 lead/变量相关性，独立 noise 不是合适的普通基线；
2. 普通直接 forecast-as-input 与完整结构的 calibration+conditional generator 仍留有稳定、重要的误差；
3. 高 K 基线确实比小 K 好，且达到其性能需要违反预先声明的单次预测计算预算。若 K=8 已足够，则“有限预算困难”不存在。

**最小可实现架构（仅探针）**：小 history encoder 产生 z_0；所有 arms 共享一个已校准 trajectory distribution qψ(U|F_o,H_o)，保留跨 lead covariance。用 rank r∈{2,4} 表示 forcing error 的少量 joint modes，样本 w_h 经相同的 input-affine 条件转移

\[
z_{h+1}=f_\theta(z_h,c_h)+B_\theta(z_h,c_h)\,a_h+\eta_h,
\qquad Y_h\sim p_\theta(Y_h\mid z_h,c_h).
\]

其中 a_{1:H} 是压缩后的整段外部轨迹，不是每小时独立抽样；candidate 只在目标评分损失下联合训练其 mode map 与 transition。训练样本的 F_o 必须是真正的 origin-specific forecast，不能用未来实际 U 作为 mean 后再加噪声冒充发行。按 dataset 分别训练允许，不要求 checkpoint 转移、action、因果识别、物理限制、LLM、Loop 或 online adaptation。

这个架构的独立问题非常窄：共享的状态转移是否使 target-sensitive uncertainty modes 在小 K 和少 target labels 时更容易学习。它不是新 Bayesian 因子分解、不是新不确定性积分，也不是声称 mode 本身可辨识。

## 4. 最强普通控制与单 5090 证伪设计

未建议立即运行；先补第 5 节数据发行证据。若主线程仍决定做一次决定性实验，可冻结以下设计，避免多轮调参救活。

| Arm | 内容 | 要关闭的替代解释 |
|---|---|---|
| A | 小 TiDE/GRU direct predictor，全部诚实 F_o、history、lead/age/revision；概率输出与 candidate 相同 | ordinary direct forecast-as-input 已足够 |
| B | A 加 qψ 的 mean、scale、low-rank covariance summary，同样支持 future 输入 | 不确定性增益只是多给 summary features |
| C | 同 qψ + frozen conditional target generator，MC/QMC K=8；同一模型另测 K=64，只增加 inference | 普通 calibration+integration，及小 K 压力是否真实 |
| D | 与 candidate 相同 latent transition；固定 input-PCA modes，K=8 | state 架构本身或普通输入降维 |
| E | 上述联合 target-sensitive modes + latent transition，K=8 | 被检验探针 |
| F | 与 E 相同可学习 mode budget，但供 direct trajectory generator 消费，K=8 | 收益只是 supervised compression，state/transition 并非必要 |

所有 arms 获得相同的可用信息、qψ、forecast calibration 更新、标签 maturity、轨迹相关结构、训练步数和调参次数。额外加入便宜的 ridge/GAM/LightGBM quantile、Gaussian augmentation，以及简单 calibrated forecast combination，作为筛查；不能因其非神经而省略。DSSMF 是最近的完整早期机制对照，KITE 是现代 generative 对照；至少读代码确认与这里条件相同后再决定是否完整复现，不应把过时 weak implementation 当唯一 SOTA。

最小数据规模：两个任务族，优先真实 forecast-vintage 的 wind 与 solar/load；每族先用固定一个站点而非遍历后选最好站点。hourly target，lookback 168，horizon 24，固定 daily origin；按时间 train/val/test，预处理和 qψ 都只拟合 train，validation 用于一次选择，test 只评一次。学习预算可设 ≤0.5M trainable parameters、每 fit ≤2000 updates、相同 early-stop 规则，2 seeds 共 24 fits；每 fit 设 30 分钟硬上限，总 GPU 预算 12 小时。**这是可执行预算上限，不是已实测运行时保证。** baseline 亦获相同上限，若某一 arm 系统性撞限需报告实际 updates/FLOPs，不能以不同训练完成度宣称结构优越。

主评分：按 series 的 train scale 标准化 CRPS 或固定 quantile 网格的积分 pinball；辅助 MAE、90% coverage/width、joint energy score、每 origin latency/peak memory。相邻预测窗相关，uncertainty 由按周 block bootstrap 给出。不可用单个 high-error slice 或单 seed 取代主比较。

预先关闭规则（建议值，不是已有结果）：

- E 对 B/C/D/F 若无两任务一致的 ≥2% 主评分改进，或配对 block CI 包含 0，关闭该具体配置；
- E 只胜 A，不胜 B/C，关闭“需要 state-level uncertainty learning”；
- E 与 F 持平，关闭“state/transition 特有收益”，最多是 task-aware compression；
- 只在 K=8 胜而 K=64 不胜，可以支持计算折衷，但必须有真实 latency Pareto，不能声称更正确的概率分解；
- 用 iid augmentation 即能抹平收益，关闭“structured forcing uncertainty”动机；
- 若剥离 uncertainty 后 E 的 state 架构同样占优，则需要一个交互检验，不能把 backbone gain 算给 uncertain forecast。

这些规则不证明跨所有模型不存在方法；只防止本轮具体命题靠不停增加模块存活。

## 5. 公开数据：什么已够，什么还没有

### 5.1 GEFCom2014-W/S：发行合同较清楚，但不能验证所有机制

原组织者 [Hong et al., Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond](https://www.pierrepinson.com/docs/Hongetal2016.pdf)，**§6.1，pp.11–13** 说明 wind 为澳洲 10 farms，ECMWF 的 10/100m u/v，daily midnight issued、hourly 至 +24h，weather forecasts 在 training 与 evaluation 都提供且未变换；**§7.1，pp.13–14** 说明 solar 为 3 plants、12 个 ECMWF forecast variables，同样 daily midnight 至 +24h。**Appendix A，p.18** 指向 [正式 DOI 与数据附件](https://doi.org/10.1016/j.ijforecast.2016.02.001)。[组织者原博客](https://blog.drhongtao.com/2016/01/probabilistic-energy-forecasting-gefcom2014.html) 称完整数据约 120MB。[IEA Wind Task36 官方测试数据页](https://iea-wind.org/task36/task36-work-packages/wp2-power-and-uncertainty-forecasting/wp2-3-test-cases/) 进一步给下载和 Task15 为完整 wind 数据的说明。

它可以支持基于**原 benchmark 发行合同**的 `issued_at≤origin` day-ahead 协议。原文只说 midnight，本次未核验 timezone 或实际 provider ingestion latency，不能擅加 UTC 和逐行真实发布戳。它没有给出本探针需要的完整 multi-vintage ensembles 与匹配实际 weather labels，因此可立即用于 A/direct 等普通 controls 的外部未来信息测试，却不能单独完成 C–F 的输入误差建模比较，也不能检验 weather labels 比 target labels 更早到达的 adaptation 变体。

### 5.2 OpenSTEF/Liander2024：小规模候选，仍需构造链核验

[原 HF data card](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark/raw/main/README.md) 的 **§4 Versioned Weather Forecasts** 有 target `timestamp` 和 `available_at`，最长 7 天；**§1** 明确 solar/wind 的两天 delay 是 simulation；**Dataset Creation** 说给测量添加了 available_at，weather 来源只泛指 Open-Meteo。非 versioned `weather_forecasts/` 还明确警告不适合真实 availability benchmark。因而不能仅有该字段就称发布历史已被审计。

[OpenSTEF 官方教程，How versioning works / Filtering / Resolving versions](https://openstef.github.io/openstef/tutorials/datasets.html) 展示保留同一 target 的多个版本和按 origin 选版本，证明软件接口能正确表达信息集；它不是 weather 下载构造的独立证据。第三方 forecast-store 教程把两天 delay 称为真实 feed lag，与 data card 的 simulate 表述不一致，本报告以原 data card 为准，不采用第三方升级后的说法。

[Open-Meteo 官方 Historical Forecast 文档的 Which Historical Weather Data to Use?](https://open-meteo.com/en/docs/historical-forecast-api) 将三类产品明确分开：Historical Forecast 拼接每个 run 的初始小时；Previous Runs 是 fixed lead offsets；Single Runs 才保留以 initialization time 选择的完整 run。当前文档说 Single Runs 的 IFS HRES 从 2024-03 开始，其他模型从 2026-04-02 开始。**这提供核验 Liander 构造的具体问题，却不证明它已用正确 API。** initialization time 也不等于公网 available_at；需要原构造代码、模型/run 版本与发布延迟依据。

对于“weather 先到、target 后到”的变体，还缺 weather actual 的**真实 release/maturity**记录。Historical Weather 是 reanalysis/analysis 产品，不能因 event timestamp 已发生便认为当时可训练校准器。可以人为设定保守解析延迟形成声明清楚的 benchmark protocol，但必须标为合成 latency，不能冒充已验证的 DSO 实时事实。

### 5.3 当前实际可执行程度

- GEFCom 的 day-ahead direct baseline 具有清楚的原任务合同；没有经过本次逐字节下载审计，也没有完整 qψ 所需观测。
- Liander 的文件 schema 和版本接口很接近所需，但真实 forecast run 与 available_at 构造链仍未通过。
- 本轮没有一个数据集同时被本视角核实满足：完整 forecast run、保守发布时刻、实际 weather labels 的 maturity、目标 label delay，以及适合 C–F 的样本量。因此**不能称“单 5090 决定性实验已就绪”**。

## 6. 本轮可交接的判断

RQ1：早期 DSSMF 已原生区分不确定外生预测与确定日历，但它的误差实验为合成；现代 KITE/SPECTRA/直接 forecast-aware 网络也不是完全忽略该信息的空白。公开发行合同核验不能替代方法差异。

RQ2：observer-only adaptation 与同信息两阶段 calibration+frozen generator+integration 没有当前可辨认的新优化残差。小 K 下联合目标敏感表示与多步转移可能有有限预算价值，已有相邻先例既不证明其必然成立，也不允许宽泛首创。

RQ3：给出了一个最小架构及 6-arm、两任务、单卡有硬上限的证伪计划；但真实数据发行链和小 K 压力尚不齐全，故没有完成可接受候选。合理下一步是只补具体数据证据或先检查普通 controls 是否已解决，而不是先搭更大 world model 再找动机。

自我反审：这里的 no-candidate 是“未达到本轮接受证据标准”，不是“所有有限预算状态框架不可能有价值”。该判断没有依赖 action、causality、physical、zero-shot 或 LLM 门槛，也没有因组件已存在就自动否定组合；它依赖的是完整组合已有多条强先例、目前拟议变体与普通控制的目标等价，以及尚未验证的现实压力条件。
