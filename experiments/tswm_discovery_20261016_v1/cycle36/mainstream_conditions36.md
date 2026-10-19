# C36 主流视角：发行预报作为条件时，状态学习还剩什么问题

核验日期：2026-10-17。主窗口：2026-04-17—2026-10-17。本文是独立工作笔记，不是新方向立项书。冻结问题采用本目录 `RESEARCH_BRIEF.md` 的 RQ1—RQ3；不要求因果识别、物理正确性、动作、zero-shot 或通用 checkpoint。没有训练、远程作业、模型下载或数据集下载；仅阅读论文、作者代码、数据提供方文档，以及一个返回零行的 Elexon 两分钟查询。OpenSTEF 是主线程给出的补盲区线索，其余检索先独立展开。

## 摘要与判决

现有原生任务已经包含“已发布但会出错的未来预报”，尤其是电价任务中的负荷与风电预报。把 WorldTS、DAG、KITE 的未来协变量统一解释成 realized future truth，会误判现有工作。另一方面，近期论文对“历史发行版本、预报误差与真实天气”的区分确实不统一：有历史观测摘要、logged actions、持久值、实际发布预报、未来观测和真值加噪等不同合同。

这些证据支持做真实发行数据下的条件预测研究，尚不支持“必须新增一种 uncertainty-aware state/transition 机制”的主张。普通 direct forecast-as-input、联合预测、概率校准与误差状态模型必须先被认真击败。当前主流视角结论是 **NO_ACCEPTED_IDEA；得到一个可核查的数据候选和一个有明确关闭条件的筛选实验，不据此冻结新方法**。

## 检索方法与证据分级

宽搜交叉使用 `time series world model 2026 covariates`、`future covariates uncertainty 2026`、`forecast issued_at weather electricity dataset`。窄搜沿 WorldTS/Pythia/BeliefGraph-JEPA、Chronos-2/t0、DAG/KITE/GCGNet、covariate uncertainty benchmark 和真实 vintage 数据分支追踪。补盲区检索进入 ensemble postprocessing，找到当周正式发表的 PBC；不能只用 forecasting 榜单覆盖本问题。

正文仅使用原论文、作者仓库/主页、官方 proceedings 与数据提供方。`源码可见` 表示发现模型文件、环境和运行入口，**不等于本轮已运行成功**。`未定位代码` 是检索结果，不等于证明作者没有代码。预印本首次日期与窗口内修订/会议版本分列，避免把旧工作包装为本窗口新作。未进入结论的线索包括只查到题目/摘要而不能核查合同的文章。

## RQ1：原生合同究竟把什么当条件

### 先区分三个对象

令真实外部过程为 \(C_{t+h}\)，在发行时刻 \(r\) 发布的预报为 \(F^r_{t+h}\)，预测原点为 \(o\)。必须满足 \(r\le o\)，但不要求 \(F^r_{t+h}=C_{t+h}\)。确定日历、已发行预报数值、未来物理真值是不同对象。

直接学习 \(p(Y_{o+1:o+H}\mid H_o,F^r)\) 时，预报数值本身是预测原点已经观测到的信号；它对真实条件的不确定性可以体现在该条件分布里。这个信息论解释是本笔记的推导，不是给某篇论文附加作者没有声称的语义。若把预报数值当作未来物理输入而学习 \(p(Y\mid H,C)\)，再代入 \(F\)，则训练/部署的条件变量已经发生变化。这两种实现不能混写。

### 状态与条件演化模型

| 工作、日期 | 原文直接证据与位置 | 对本轮问题能与不能说明什么 |
|---|---|---|
| WorldTS；2026-09-25 首发，窗口内 | §3.1 Eq.(1)—(5)：covariate availability 用 \(\tau=T\) 或 \(T+H\) 表示；编码未来观测监督预测 latent states，随后冻结并训练 decoder。A.1 明确 NP/PJM/BE/FR/DE 使用负荷、风电或发电 **forecasts**；其他数值任务混有测量/ERA5。[全文](https://arxiv.org/html/2609.31162v1) | 已经是 forecast-conditioned latent prediction 的直接先例；不能声称它只用 future truth。合同不含发行轴；这些电价列是否满足每个 origin 的实际发布时间，仍须数据溯源。 |
| Pythia；2026-10-04 首发，窗口内 | §3.2、C.3：事件按 origin 截断，数值 covariate 只汇总历史观测；context correction 改变预测 state，另有概率 readout。首页 Code/Model 均写待发布。[全文](https://arxiv.org/html/2610.05240v1) | 是条件状态学习先例，但本身不评测 future external forecast uncertainty；不能拿其 multimodal 成绩替代发行预报实验。 |
| BeliefGraph-JEPA；2026-10-04 首发，窗口内 | §3：预测时用 logged future actions；未来 exogenous inputs 若使用则保持最后历史值。typed latent effects 随 driver rollout，再路由至 target。§5 与附录含匹配结构控制。[全文](https://arxiv.org/html/2610.05409v1) | 对“显式条件影响状态的跨步演化”很接近；其 exogenous 合同不是真实已发行预报，不能当本问题已被直接实验关闭。也不要求本轮引入其动作/图先验。 |
| ViBR-WM；2026-09-27 首发，窗口内 | §3.3 Eq.(2) 为 posterior parameter draw 与 disturbance 下的联合状态 rollout；§3.4 只用原点已知 covariates，且各 horizon 直接预测并不构成 joint path。[全文](https://arxiv.org/html/2609.33844v1) | 证明 Bayesian world modeling 已覆盖参数/扰动/变量选择不确定性；不等于已经对真实外部预报误差做 uncertain-input 积分。未定位可运行作者仓库。 |
| OutageDiT；2026-09-01 首发，窗口内 | §3 Experimental setup、A.1 明确天气在整个未来 horizon 已被观测；§4 将 weather-forecast error 列为后续工作。[全文](https://arxiv.org/html/2609.01896v1) | 其 horizon-aligned condition states 与整路径 flow generation 是相关机制，但当前成绩不能证明真实发行天气条件下同样有效。未定位作者代码。 |

WorldTS 的代码状态另从[作者 Xiangfei Qiu 主页](https://qiu69.github.io/)核查：WorldTS 条目只有论文链接，邻近已开源条目有明确 Code；本轮未定位其代码。Pythia 的待发布是作者原文声明，证据强于“搜索没找到”。BeliefGraph-JEPA 的论文/摘要未给出本轮可定位作者实现，因此只列机制先例，不能直接计入可运行对照。

### 强普通条件模型与原生误差实验

| 工作、版本日期 | 核查到的输入/实验合同 | 实现状态及边界 |
|---|---|---|
| t0；2026-09-21 首发 | §2 为 target history、past covariates、known-future covariates；§5.1.2 Table 5 用同 checkpoint 对比带/不带条件，30 个 future-covariate 任务中 19 个改善；Fig.7 在德国电价的两条 forecast covariates 上做时移/打乱实验。[完整论文](https://arxiv.org/pdf/2609.24559) | [作者代码](https://github.com/theforecastingcompany/tfc-t0) README“Forecasting with covariates”直接把 weather forecasts 列作输入，形状 `[B,F,T+H]`。这是很强的直接条件控制；上述扰动不是实际 vintage 误差研究。 |
| DAG；首发 2025-09-18，窗口内 v3=2026-05-12 / ICML 2026 | §3 定义 exogenous 可以提前估计；Eq.(1) 显式接受 future exogenous。时间/通道相关发现再注入目标预测。不能按旧版标题强加因果识别要求。[v3 全文](https://arxiv.org/html/2509.14933v3)、[日期](https://arxiv.org/abs/2509.14933) | [作者仓库](https://github.com/decisionintelligence/DAG)有 `scripts/w_future/DAG.sh`、依赖与模型实现。是既有方法及近期会议版本，不是 2026 首创。 |
| KITE；ICML 2026，2026-07-06—11 proceedings | 原 PDF p.3 §3.2 Eq.(3) 学 \(p(Y^{endo}\mid X^{endo},X^{exo},Y^{exo})\)；p.4—6 的 history source、统计先验 attention、CFG；p.6—7 将 future covariates 设为显式已知/高精度输入。它是条件输出分布，不自动等于已显式建模 covariate forecast error。[官方论文页及 PDF](https://proceedings.mlr.press/v306/cheng26t.html) | [作者仓库](https://github.com/decisionintelligence/KITE)实现/脚本可见；`ts_benchmark/baselines/kite/kite.py` L362—423 把 `exog_future` 切块并作为 `y_exo` 传入 inference。没有原生 `issued_at` 参数。 |
| GCGNet；2026-03-09 预印本，较早基础；ICLR 2026 版本 | A.1 确认 EPF 列是实际 forecast；§4.2/B.4 的 robustness 用零/随机替换的 missing exogenous，不是原始气象发行误差。[全文](https://arxiv.org/html/2603.08032v1) | [作者实现](https://github.com/decisionintelligence/GCGNet)含模型、依赖、w_future 脚本。其普通 MLP future fusion 也是应认真纳入的控制，见 A.3 Eq.(14)—(15)。 |
| Benchmarking TSFMs for Load Forecasting Under Covariate Uncertainty；2026-10-05 首发 | §II-C：训练 ARIMA、用 val 残差标准差，在预测时给真实 covariate 值加独立零均值 Gaussian；ISO-NE 的 DA_DEMD 本来是 day-ahead load forecast，例外不加噪。§III-B 显示较大噪声下 TimesNet 超过 Chronos-2。[完整论文](https://arxiv.org/html/2610.07232v1) | 直接命中本 RQ 的近期实验；但作者承认忽略时间相关、偏差、非高斯误差。没核实真实 forecast vintage，没定位代码；不得当作部署级残差证据。 |

较早但不能省略的控制是 Chronos-2（2025-10-17）：§3.1/3.4 对 unknown future 槽置 missing，对 supplied future 条件置数值；[完整报告](https://arxiv.org/html/2510.15821v1)。[作者 pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py) `predict()` 的字典合同要求 `target`、`past_covariates`、`future_covariates`，后者键为前者子集；代码不要求输入数值必须是未来物理真值。公开源码与 checkpoint 存在，本轮未加载模型。

RQ1 的回答：**已有模型容许且已有任务确实使用不完美的未来预报信号；针对严格历史发行版本的误差分布、条件状态演化和训练/部署匹配的联合原生评测，当前核查到的近期状态论文证据不足。**“证据不足”限定于此合同，不是宣布整个方向无人做过。

## RQ2：普通方法已覆盖什么，剩余命题如何收窄

### 最新的校准/组合先例不能省略

Guan 等 PBC 首稿恰为 2026-04-17，v3=09-04，[Nature 正式版](https://www.nature.com/articles/s41586-026-11126-x)为 10-14。其[可读全文](https://arxiv.org/html/2604.16238) §2.1/4.3 接收 raw forecast ensemble、滞后观测与 climatology，分别纠偏和回归组合，投影成合法 CDF 后平均；训练观测按 issuance 可见性截断。§2.4 有真实 AI Weather Quest 评测。[作者代码](https://github.com/mouatadid/pbc)已公开。这直接反驳“现有普通方法只把预报均值当未来真值、不会利用历史误差”的概括，但不证明其跨任务 latent state 已最优。

下面是本笔记的统计分析与实验要求，不把一般表达能力误写成某篇论文已经解决了一切。

| 普通控制 | 在信息相同且训练合同正确时，已经能解决的部分 | 仍需真实实验证明的有限预算问题 |
|---|---|---|
| Direct forecast-as-input：TiDE/MLP、DAG、KITE、微调 Chronos-2 | 把原点已知的预报数值当有噪声信号，直接最小化下游 point/proper score；无需先恢复“正确天气” | 在小样本、预报版本改变、多 horizon 相关误差下，何种归纳偏置更省样本/算力 |
| 普通联合生成 \(p(Y,C\mid H,F)\) | 同时学习目标和外部条件的条件分布、相关性与路径；不必强制假设独立天气误差 | 相同参数/时间预算下是否比专门的 transition 结构差；不能只与独立单变量生成器比较 |
| Ensemble marginalization | 以真实 ensemble member 路径传播非线性响应并平均，不只做 mean plug-in | ensemble 本身有偏/欠离散时的校准、以及每个条件路径采样的预算；这些不能通过“Jensen”命名新方法 |
| Calibration / forecast combination | 修正条件偏差、方差和 recent-error persistence；PBC 是本窗口完整实例 | 若优势只来自输出残差层，就属于校准收益，未证明必须改变状态形成/演化 |
| Bayesian uncertain-input / 状态误差模型 | 参数、扰动与 latent dynamics 可概率化；对输入另建分布后可进行积分 | 需要明确输入 posterior/ensemble 从何而来，及 train/test 如何一致；只写 Bayesian 并没有提供该证据 |

一个可讨论但**尚未得到支持**的有限预算假说是：从发行预报与已经兑现的旧误差中学习一个共享 error-regime state，令它改变未来条件对目标演化的作用，而非仅校正输出。它只有在以下匹配后才可能留下残差：direct 模型拥有相同全部历史与预报/lead/版本信息，普通 joint 模型允许相同未来路径依赖，输出校准模型也拥有旧误差历史，且预算接近。

这个假说很容易退化：error state 若只是预测一个 additive output bias，就是 residual calibration；若只是生成 noisy covariate paths 并平均，就是输入边缘化；若全部收益来自有无 `issued_at`、额外 ensemble spread 或更多输入行，是信息差。已知组件不自动否定其交互，但作者需要证明交互而非重新命名。当前检索没有给出足以越过这些普通解释的正证据。

## RQ3：数据与最小证伪协议

### A. 最优先的数据候选：Elexon 原生发布历史

官方[开发者入口](https://developer.data.elexon.co.uk/)明确 dataset endpoints 可按 published times 取 superseded forecasts。[当前前端](https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/WINDFOR)引用的官方规范为 [OpenAPI JSON](https://data.elexon.co.uk/swagger/v1/swagger.json)。本轮通过公开规范直接核查这些 JSON 位置：

| 对象 | 精确规范位置/字段 | 可做的严格过滤 |
|---|---|---|
| `/datasets/WINDFOR` | `paths['/datasets/WINDFOR'].get`；参数 `publishDateTimeFrom/To`；`components.schemas['Insights.Api.Models.Responses.Generation.DatasetRows.WindGenerationForecast']` 有 `dataset,publishTime,startTime,generation` | `issued_at=publishTime`，`valid_time=startTime`；对每个 origin/valid_time 仅选 `publishTime<=origin` 的版本，按该列降序取最新 |
| `/datasets/FUELHH` | 参数可按 `settlementDateFrom/To`、`fuelType` 或 publish range；`...DatasetRows.AugmentedOutturnData` 有 `publishTime,startTime,settlementDate,settlementPeriod,fuelType,generation` | 历史输入同时要求其自身 `publishTime<=origin`；目标后验 label 可用事先固定的 outturn snapshot，不允许其倒流进 context |

两个精确的 bounded probe 候选（本分支未下载这两条历史日；已交主线程独立核验）：

```text
https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR?publishDateTimeFrom=2026-01-01T00:00:00Z&publishDateTimeTo=2026-01-02T00:00:00Z&format=json
https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH?settlementDateFrom=2026-01-02&settlementDateTo=2026-01-02&fuelType=WIND&format=json
```

不能使用 `/forecast/generation/wind/latest?from=...&to=...` 直接回填历史 origin：它按目标区间给 latest view，须另用发行过滤证明可用。另有 `/forecast/generation/wind/history?publishTime=...` 接受单个准确发行时刻。官方 [Wind generation](https://bmrs.elexon.co.uk/wind-generation) 说明 forecast 对象为 ESO 可见且有 operational metering 的风场，图中 outturn 取 Generation by Fuel Type；[outturn 定义](https://bmrs.elexon.co.uk/generation-by-fuel-type)也排除 embedded/无 operational meters 风场。因此总体支持对应，但没有据此证明 fleet、curtailment 与 revision 口径完全一致；该部分要在冻结实验前说明。

这个候选的优势是直接 `publishTime`，不是把 initialization 或下载时间重命名为 issuance。其限制是只有单一 GB 聚合风电时，本质上首先是 forecast postprocessing；单任务胜利不能宣称发现通用时间序列状态机制。本分支本地公开 HTTP 可以读取官方规范，主线程报告服务器访问同一规范为 HTTP 403；不能把本地元数据访问写成服务器数据下载成功。本分支未证明查询可得到完整目标年/全部发行，也未缓存固定字节版本，故状态为 **PROTOCOL_CANDIDATE，未冻结数据版本**。主线程的一日原始 SHA 审计可升级 provenance，不会自动升级机制证据。数据访问与版本未冻结不是关闭方法的理由；方法主张当前没有成立，依据是 direct/joint/calibration 普通解释未被排除。

### B. 其他公开数据为何暂不能替代严格发行证据

| 数据 | 原生依据 | 本轮可用边界 |
|---|---|---|
| EUPPBench v1.0，较早数据基础 | [原论文](https://essd.copernicus.org/articles/15/2635/2023/index.html) §2 区分 2017—2018 operational forecasts 与旧年份 reforecasts；站点观测可配 ensemble。§4.4—4.5 给 EMOS、AR-EMOS；后者直接建模旧误差 AR。 | 很适合强校准控制；优先只用 operational forecast 分支。time 是 forecast reference，尚非原生公开时间；[作者文档](https://eupp-benchmark.github.io/EUPPBench-doc/files/EUPPBench_datasets.html) 写 noon，而论文写 00Z，需查真实 metadata 后定 origin。不能悄悄当无问题 vintage。 |
| OpenSTEF Liander2024 | [作者 card](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark)的 `weather_forecasts_versioned` 有 `timestamp,available_at`，但文字说模拟 availability；源链接是 Historical Weather API。 | 最合适的结构不等于已证明原始发行。当前[消费代码](https://github.com/OpenSTEF/openstef/blob/main/packages/openstef-beam/src/openstef_beam/benchmarking/benchmarks/liander2024.py) L95—100 只确认读取 versioned parquet，并未给出生成来源。未核清构造之前不得称 true historical weather vintage。 |
| Open-Meteo Single Runs | [官方文档 Data Sources](https://open-meteo.com/en/docs/single-runs-api)：`run` 是初始化而非公开时间；多数模型 archive 从 2026-04-02 起。2024 起的 IFS 档案特别写为 Cycle 49R1 hindcasts。 | 不把 2024 IFS 回算当当年实际发行。2026 档案仍要加提供方发行延迟并固定 run/model；不拼接后来更新。[Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) 是各 run 头几小时的 seamless series，不能作为某 origin 的完整未来预报。 |
| NREL PERFORM | [作者 README Dataset/Forecast/File format](https://github.com/PERFORM-Forecasts/documentation/blob/main/README.md) 有 `issue_time,forecast_time` 与 99 percentiles，但部分 actuals 来自 SAM/WRF，预测亦为研究生成。 | 可做自洽 reforecast/半合成协议，不能只因字段叫 issue_time 就声称历史真实公开 vintage。 |

### C. 供主线程决定是否冻结的最小实验，不是已接受 Idea

**要证伪的具体命题**：给定同一发行时可用的信息，显式跨步 uncertainty/error state 在有限样本预算内，对强 direct 条件模型和普通误差状态/校准控制提供可重复的概率预测增益；增益不只来自输出 bias 或额外输入。

先用 Elexon 小规模日审计决定能否建全期数据。只有 provenance 通过后，才把连续年月划成 train/val/test；按 forecast origin 切分，排除 horizon 重叠导致的 label 泄漏。所有输入版本均做 as-of join。仅以训练期已兑现误差构造 error history。预先固定 origin（如每日 12:00 UTC）和 48 个未来半小时；所有模型获得同一已发布版本。需要同时报告没有外部 forecast 的历史基线、raw issued forecast 和 strongest calibration，避免掩盖外部模型自身已带来的收益。

最小对照建议 6 arms：① raw forecast + 历史统计校准/AR error；② 参数匹配 TiDE/MLP direct，含相同 forecast、lead、历史误差；③ 原生 DAG 或 KITE 的完整作者实现（优先一个跑通且预算适配者）；④ 普通 joint probabilistic forecaster，联合目标和条件并显式接受 F；⑤ 候选 state/transition；⑥ ⑤的 matched 输出校准版本，将同量参数用于 output residual 而不改变 transition。②和④不得通过不提供误差历史、删掉 horizon 信息或弱化协变量融合来让路。

单 5090 的筛选预算建议限制总训练配置为 6 arms × 2 chronological tasks/splits × 2 seeds = 24 fits；小模型使用同一 context=168h、horizon=24h、最多 50 epochs、val early stopping，采用预先固定的一套配置，不另做超参搜索。这里是拟议预算上限，不是本轮运行记录，也不承诺某作者默认全设置都能在此预算跑完。若只有同一聚合风电的两个时间段，须称 two splits，不能冒充两个独立 domain。第二独立任务仍须另过发行审计；在其不存在时，不能发出“跨任务通用机制”结论。

主指标为预先固定 quantile loss/CRPS；path 模型另外用 energy/variogram score 检查完整路径，point MAE 作为辅助。以 origin-day 做 paired block bootstrap，不能把每个半小时当独立重复。候选须在两个独立任务各较 strongest 普通控制改善至少预先规定的实用阈值（筛选可用 2%），且概率校准无明显恶化；阈值是筛选决策，不是理论界限。若 direct 已相当、输出校准解释全部收益、效果只出现在挑选的 severe-error 子集，或新增 state 不胜同预算 joint 模型，就关闭此具体命题。不能用继续换 seed/小调 loss 维持。

## 对三问的最终回答

1. **区分已存在，但未统一**：WorldTS/DAG/KITE 等已有 forecast-conditioned 任务；Pythia 是历史 context，BeliefGraph 是 logged action/持久 exogenous，OutageDiT 当前是未来天气观测。没有理由把它们统一判作泄漏，也没有理由用其 benchmark 证明严格发行预报误差已被解决。
2. **普通方法覆盖了大块问题**：direct 条件分布已经允许预报不准，PBC/AR-EMOS 已展示有时间信息的误差校准，joint/ensemble/Bayesian 方案应作强控制。本轮没有发现足以排除这些解释、并要求新状态框架的有限预算残差。
3. **Elexon 是可进一步核验的真实发布协议候选**：其 `publishTime` 比 init-time/模拟 available_at 更直接；本分支没有冻结完整数据版本。给出可失败的 24-fit 筛选设计，但不因审计进展强行留下 Idea。当前合理产出是提高问题和证据的精度，而非立刻训练另一种 latent module。
