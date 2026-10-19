# C32 mainstream-model 核查：目标协议、既有修复与剩余判别

检索/源码审计日期：2026-10-17。视角：WorldTS、LatentTSF 及 2026 predictive latent targets 的主流路线。主检索窗口 2026-04-17 至 2026-10-17；必要回溯明确标出。只使用公开论文、作者项目与官方代码。没有训练、推理复现或 SSH；下列实验均为待执行方案。本视角只写本文件。

## Abstract

已核查证据不支持把“未来观测编码不是完整物理状态”作为新框架的缺陷。WorldTS 的公开实现并不是 future-only point encoder：它使用带历史前缀的局部 patch target、独立 covariate 分支、history statistics normalization，以及面向预测状态训练的 decoder。LatentTSF 则刻意固定 observation autoencoder 及其几何。两者并不是同一种 target-learning protocol。

2026 年已有的 multi-horizon EMA targets、factorized predictive targets、观测 grounding 和 joint tokenizer/generative forecasting，已覆盖许多宽泛修复。当前可保留一个具体问题：**同一 decoder 同时拟合 predicted states 和 encoded states，是否会在真实任务上产生预测损失的负迁移？** 这是可证伪的训练兼容性问题；最强第一修复已在 WorldTS 原生代码里。因此，本视角的**新研究方向结论为 NONE**，并保留一个小规模资格实验，不把它包装成新方法。

## 1. 问题与证据口径

本视角回答三个问题：实际模型到底预测什么；现有工作已修复什么；排除 C11/C21 后，哪一个最小实验仍值得做。

- **事实 F**：本轮逐条读到的论文内容、官方 pinned source 行为。
- **推断 I**：由上述实现推出的可能机制或数学条件，没有对应实际失败结果。
- **未知 U**：尚未在 native task/checkpoint 上测得，不能改写成已有发现。

C11/C21 作为 brief 提供的已知实验约束使用，本轮没有重跑。尤其是 C21：成熟 compressor 上同一类 projection 能成功，所以绝不能把先前失败写成 target 原理不足。没有把 physical identification、strict causality、zero-shot shared checkpoint 或无限 rollout 当作这些模型的必需性质。

## 2. 官方来源的版本与本地副本核对

| 来源 | 本轮验证的版本 | 可以声称的范围 |
|---|---|---|
| WorldTS | 官方 `decisionintelligence/WorldTS` main = `269426ea6217d7152a244da0e209d922acd46d4f`，commit 2026-10-09，message `Initial WorldTS release (snapshot 753972b)` | 官方 pinned DAG numerical route 的实际代码；不是根据本地目录名猜测 |
| LatentTSF | 官方 `Muyiiiii/LatentTSF` main = `7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2`，commit 2026-05-23 | 官方 `my_AE.py`、`my_train.py` 与 README 的默认 recipe |

版本通过 GitHub public API 查询。对应 [WorldTS commit](https://github.com/decisionintelligence/WorldTS/commit/269426ea6217d7152a244da0e209d922acd46d4f)、[LatentTSF commit](https://github.com/Muyiiiii/LatentTSF/commit/7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2)。

**F：**本地 `upstream/WorldTS` 无 Git provenance 的疑问不能只靠字符串判断。本轮从官方 pinned raw URLs 读入文本，统一 CRLF/LF 后，与本地以下四个文件逐字比较，四者均完全相同：

1. `DAG-bench/ts_benchmark/baselines/worldts/models/worldts_model.py`，23,613 characters。
2. `DAG-bench/ts_benchmark/baselines/worldts/worldts.py`，36,525 characters。
3. `DAG-bench/ts_benchmark/baselines/worldts/losses/latent_state_loss.py`，9,908 characters。
4. `DAG-bench/scripts/w_future/worldts.sh`，43,590 characters。

官方 pinned source 本身就包含 VISReg options 和 `v46B-lite` 字样。它们的存在不是本地改动的证据；脚本选中的 active route 才决定 native baseline。该检查不证明本地整个 tree、数据或 checkpoint 均未改动。

## 3. 两个目标协议不可混写

### 3.1 WorldTS：learned local-window target，非 latent distribution learner

**F：**[论文 Eq.18–21](https://arxiv.org/html/2609.31162v1#S4.SS5) 使用 stop-gradient future target 的 L1 matching、整条 future sequence 的 flattened cosine，以及对 predicted/encoded states 的 variance–covariance regularization。Stage 2 固定 encoder/predictor，同一个 decoder 接收 predicted 和 encoded future states，分别产生 forecast 与 reconstruction supervision。这不是 energy score、conditional likelihood 或完整未来条件分布识别目标。论文 §3 允许 future-known 和 future-unknown covariates。

**F，源码复核：**[pinned model](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/DAG-bench/ts_benchmark/baselines/worldts/models/worldts_model.py#L340) L340–352 在 future target 前拼接 `state_patch_len-1` 个历史 target；L478–500 使用 **history input** 的 RevIN statistics 同时 normalize future target/future covariates；L504–542 分别 encode named target、historical/future covariates 并传给 predictor。因此“单个未来观测不能反演完整状态”“未来 normalization 泄漏”“把 observed driver 强迫塞进 target”都不是对该 route 的准确描述。

**F，active recipe：**[released numerical scripts](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/DAG-bench/scripts/w_future/worldts.sh) 使用 `stage1_latent → stage2_decoder_mix`，latent L1/cosine weights 为 10/15，regularizer weight 0.1，stage1/stage2 各 20 epochs，mixed reconstruction weight 0.3。未显式改 regularizer 时，[wrapper defaults](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/DAG-bench/ts_benchmark/baselines/worldts/worldts.py#L49) 选择 `legacy_vc`；可选 VISReg 不能混称论文默认。

**I：**该 target 是为有限 horizon 条件点预测服务的 learned observation-window statistic；不是已证明的 minimal sufficient predictive state，也不需要先是这样的 state 才能改善 forecast。L1+cos+regularization 的联合最优点不能直接写成 conditional mean；仅纯 L1 部分对应 latent coordinate medians 的讨论也不能替代完整目标分析。

### 3.2 LatentTSF：frozen observation chart 与固定 latent metric

**F：**[LatentTSF §3.3–3.5](https://arxiv.org/html/2602.00297v2#S3.SS3) 先预训练 point-wise autoencoder，再冻结 encoder/decoder；预测器学习 encoded history → encoded future，损失为 latent squared error 加 flattened cosine。该论文首发 2026-01-30，属于窗口外的必要背景；官方 May release 在窗口内。其 §5.4 报告 decoded supervision 和 encoder/decoder fine-tuning 在所测配置中退化，不能无视该反证。

**F：**[pinned `my_AE.py`](https://github.com/Muyiiiii/LatentTSF/blob/7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2/my_AE.py#L17) 的基础 MLP AE 在最后一维逐点应用两层 encoder/decoder；[pinned `my_train.py`](https://github.com/Muyiiiii/LatentTSF/blob/7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2/my_train.py#L250) 默认 prediction/alignment weights 为 10/15，`perceptual_weight=0.0`，encoder/decoder 均冻结。可选 temporal AE 文件/flag 的存在，不意味着论文默认 target 已使用历史。

**I：**它学的是固定 chart 中的可预测统计量。固定多样 target 可以避免 target encoder 自己塌缩，却不能无条件保证 optimal forecast 有非零方差；这不等于已证明任何真实 LatentTSF failure。非线性 decoder 下 latent prototype 与 observation-optimal forecast 可能不同，但 WorldTS 已通过预测状态训练 decoder；不能把 LatentTSF 的潜在问题直接移植为 WorldTS 缺陷。

## 4. 2026 主流修复已覆盖到哪里

下表按解决机制组织，不把相近名称当独立新方向。事实均为论文层或项目层核查，未复现其数字。

| 机制与公开工作 | 已核实事实 | 对本轮方向的约束 |
|---|---|---|
| 多 horizon / encoder 使用协议：[CF-JEPA](https://arxiv.org/abs/2606.07031)，Lee、Sim，2026-06-05 | Random contiguous crops；short/mid/long future representation prediction；EMA 与 online encoders 对不同任务分别使用。这里 CF 指 Crop-based Forward，勿与 October controllability factorization 同名论文混淆 | “future target + EMA + 多尺度/多 horizon”已是直接路线。摘要级核实足以排除宽泛创新声称，不能据此断言其已解决全部 covariate-aware 问题 |
| 分解 predictive content：[Orthogonal JEPA](https://arxiv.org/html/2608.20065v1#S2)，Cui、Heng、Ouyang，2026-08-20；后续 [JEPA-Anything](https://arxiv.org/html/2609.20800v1#S2)，2026-09 | Learned analysis bases、分支 regression、orthogonality/activity/variance regularization 与完整 latent synthesis；跨领域 predictive-state interface | “单个 latent 中 dominant component 压制弱 component，因此加 factor heads”已有直接先例。这两个来源属于同一 OPF 路线，不能算两份独立证据；它们也不自动证明 semantic disentanglement |
| 以 numerical observation 约束 target：[Eidos](https://arxiv.org/html/2602.14024v1#S3)，Zhou 等，2026-02-15（窗口外必要背景） | Future aggregator target、stop-gradient latent alignment、grounding 与直接 quantile forecasting supervision；目标 anchoring 已实做 | “latent target 不够 informative，加入 observation grounding”不是新 gap。其 paper 主要是 univariate forecasting，不能直接当成已证实适配 WorldTS multimodal/covariate route 的结果 |
| Joint stochastic target/forecast 学习：[Latent-Foresight](https://arxiv.org/html/2610.01942v1#S3)，2026-10-01 | Joint tokenizer + flow predictor；future target/noisy input stop-grad、latent normalization、encoded/predicted-latent reconstruction；对 stop-grad/normalization 做 ablation | “用 stochastic latent objective、joint representation learning、predicted-latent reconstruction 消除冻结 tokenizer 不匹配”已被明确提出。证据来自 VFM scene features，不能外推为数值时间序列的既定胜利 |
| 显式 future-driver effect dynamics：[BeliefGraph-JEPA](https://arxiv.org/html/2610.05409v1#S3)，Li 等，2026-10-04 | Typed latent-effect state 的 future-driver rollout、graph routing 与 capacity-controlled residual；matched controls 比较这些模块 | “迟滞外因影响应进入 latent dynamics”也不是空白。其动作/机制结构是该任务的条件，不能反向要求 WorldTS 必须有 action 或 physical identification |
| 跨 backbone 的实际反例：[Does JEPA Pretraining Help TSF?](https://arxiv.org/html/2609.31680v1#S2)，Feng 等，2026-09-16 | 一个 shared-encoder LeJEPA/SIGReg instantiation，nine backbones、eleven benchmarks，之后监督 fine-tuning；报告明显 backbone-dependent gains/losses | 支持“组合兼容性需测”，不支持“JEPA 必然有益”或“全部 JEPA 无效”。这是单一 instantiation；其优化/统计解释不是对 WorldTS 的机制证明 |

综合推断：固定 AE、learned regularized target、EMA multi-horizon、factorized target、grounded/joint stochastic target 已经形成连续的设计空间。剩余研究价值不能靠重新命名这个空间取得；但已有组件不排除某个预算受限组合有价值，关键是证明一个 ordinary repair 仍解释不了的 native residual。目前没有该 residual。

## 5. 唯一保留的精确问题：mixed decoder supervision 的兼容条件

**RQ***：在 WorldTS 已有、输入合协议的 real forecasting task 上，固定同一个 stage1，`L_pred + 0.3 L_recon` 是否比 inference-only `L_pred` 系统性更差？若有差异，它是否只来自 shared decoder 的两种监督分布冲突，且由原生 prediction-only stage2 消除？

这里的 decoder input 必须包含它实际看到的**完整输入**：latent patches、history-state prefix 和 RevIN context；不能只比较一个 latent token。

**I，机制的必要条件：**令 Qp(S,Y) 是 predicted-state decoder inputs 的联合分布，Qr(S,Y) 是 encoded-state inputs 的联合分布。Mixed objective 在二者的加权混合上拟合 decoder；native L1 下其最优解是该混合的 conditional median。只有在两种 S 的 support 有足够重叠、而 conditional label medians 不同，或 finite decoder capacity 产生拟合竞争时，reconstruction supervision 才可能伤害 Qp 风险。若支持近乎分离、两个条件 median 一致，或额外监督改善有限样本估计，mix 可以更好。

**U：**本轮没有测得这些必要条件在任何 WorldTS real task 中成立。将“存在两个分布”直接写成“必然 train–test mismatch”不成立。且 WorldTS 已显式把 predicted states 送入 decoder，因此“decoder 只见过 true encodings”是错误指控。

**最强普通修复，F：**[pinned wrapper L821–861](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/DAG-bench/ts_benchmark/baselines/worldts/worldts.py#L821) 已实现 `stage2_decoder_pred`；mixed route 的 reconstruction contribution 就是 `decoder_warmup_weight * warm_loss`。因此 alpha=0 / prediction-only 是现成 control，不需要发明 target-selection、readout-aware complex loss 或新 decoder。进一步的普通竞争者是小学习率 supervised fine-tuning；跨任务证据提示它可能有效，也可能破坏 frozen geometry，须实测而不能默认胜出。

## 6. 一个单卡最小资格实验与停止规则

**只做 fixed-stage1 decoder fork；先不改 target。** 首个 native task 用 released Energy T=168、H=24、latent_dim=32、state/decoder patch_len=3 recipe。保留原 target/covariate roles 和原 chronological split，不人为压成二维 bottleneck。[官方 active script](https://github.com/decisionintelligence/WorldTS/blob/269426ea6217d7152a244da0e209d922acd46d4f/DAG-bench/scripts/w_future/worldts.sh) 提供这些设置。该规模适合单卡筛查是容量上的判断，实际 5090 显存/耗时未测，不能报告已运行。

1. 使用来源、config、split 可核实的 stage1 checkpoint；没有合格 checkpoint 时，未来实验才按 native recipe 生成，当前不启动。每个 seed 的同一个 stage1 分叉给两臂，freeze 所有 encoder/predictor。
2. A：原生 `stage2_decoder_mix`，alpha=0.3；B：原生 `stage2_decoder_pred`，alpha=0。相同 decoder initialization、batch order、optimizer、learning rate、early-stopping rule，3 paired seeds，共 6 decoder fits。只按 validation 选 checkpoint。
3. 主指标 test MAE，次指标 MSE；同时记录 predicted-input error 与 encoded-input reconstruction error，逐 horizon、按 validation 定义的 predictability bins 报告。无需搜集物理状态或 artificial latent probes。
4. 先比较相同 optimizer updates，此时 B compute 不超过 A。若 A 更强，再给 B 补足 A 的 decoder-training compute budget，才评价 mix 的净价值；记录 wall-clock/FLOPs proxy 与已用 updates。禁止把额外调参预算藏给新臂。
5. 若存在稳定且有实用幅度的差异，再在 NP H=24 做一次完全同协议复核；未通过首个资格实验就停止，不先扩大模型或重做 toy。

| 结果 | 能作出的判断 | 不能作出的判断 |
|---|---|---|
| B 稳定优于 A，且 encoded reconstruction 变差 | decoder auxiliary task 的负迁移成立；native control 已修复该任务 | 不能宣称未来 target 原理不足或得到新 world-model 框架 |
| A/B 相近或 A 更好 | 这条具体 mismatch 动机撤销 | 不能据此证明 target universally sufficient |
| 两臂都差，encoded-state reconstruction 也差 | decoder 容量/优化或编码可逆性优先排查 | 不能先归咎 predictor target |
| encoded-state decode 好、两臂 forecast 都差 | stage1 prediction/metric/信息问题仍未区分 | 不能从一次预测误差推断 hidden-state aliasing |

最后一行若出现，下一次判别才比较 latent objective weights/optimization、相同输入的普通 observation predictor、以及 capacity controls。**两臂实验不能一次识别这三者。** 同样，输入协议问题先通过 channel/timestamp/normalization 审计排除，不能靠“换目标后有效”倒推出原接口正确。

## 7. 对 RQ 的结论与决策

- **RQ1：**核实的是 learned/local observation-window target 和 frozen observation chart，不是已证明的 physical/predictive sufficient state；其 conditional forecasting utility 必须用 native task 评估。没有找到可归因于这些目标的已复现新失败。
- **RQ2：**多 horizon、EMA、factorization、grounding、joint stochastic target 已有明确先例；WorldTS 自己已有 predicted-state decoding 和 prediction-only control。现有证据不足以把 generic history-aware 或 informative target 当作未解决贡献。
- **RQ3：**最低成本下一判断是 native fixed-stage1 mix-vs-pred decoder fork。它能撤销或确认一个训练兼容性疑点，不能单独证实信息不足；若 pred-only 修好，即停止 target-method 叙事。

**方向决策：NONE。** 保留上述资格实验作为普通 baseline audit，不把它升级成研究主线。只有在输入协议、原生 decoder 修复、合理优化/权重与容量 controls 后，仍出现跨任务可重复 residual，才值得提出目标机制。本轮没有这样的证据。
