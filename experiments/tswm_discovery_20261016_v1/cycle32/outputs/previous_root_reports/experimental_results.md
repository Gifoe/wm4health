<!-- cycle31-current-start -->
# 最新 cycle31：按用户纠正研究范围

UTC 2026-10-16T19:51:23.870858+00:00。通用时间序列 world-model 框架，以 WorldTS 为研究层次参照；不强制零样本共享 checkpoint、因果/物理解释或某种结构。NormWear-2 原生预测执行、三处条件接口诊断和普通修复完成，独立复核通过；不是新方法或论文性能复现。接受方向0，整体 ACTIVE_UNACHIEVED。

见[范围纠正](cycle31/SCOPE_CORRECTION_20261017.md)、[实际证据](cycle31/experimental_results.md)与[当前研究判断](cycle31/final_research_direction.md)。以下各轮历史保留，其“当前”仅指当时。
<!-- cycle31-current-end -->

<!-- cycle29-index-start -->
# 最新证据cycle29

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/experimental_results.md](cycle29/experimental_results.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/experimental_results.md](cycle28/experimental_results.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

# 实验事实与适用范围

本轮在服务器 RTX 5090 32GB 完成。数据与脚本可生成；没有下载不必要的大型信号集。所有 discovery 与 confirmation 定义见各 plan JSON。结果没有进入超参数或 checkpoint 选择。

## E1. 已训练 WorldTS 的条件可达范围

`outputs/existing_future_support.json`：4 个旧官方 core checkpoint，各 8 个训练 context。NP 输入条件支路的最大支持长度为 5；PJM 为 9。所有位置探针在理论支持之外的数值变化为精确 0，独立 autodiff 的远距梯度也为 0。保存 data/config/checkpoint hash 与权重前后 hash。该结论是本版本结构限制；输入为 observational covariates，所以不能解释为电价的真实因果效应。

## E2. 新随机动作诊断迁移，3 seeds

系统 `x[t]=.94*x[t-1]+.06*u[t]+noise(.005)`；actions AR(1) rho=.97。独立 train2048 / validation256 / discovery256 / 尚未使用的原 confirmation512；L64/H48。脉冲幅度 .5、位置0/12/24，用相同噪声的精确配对真值。effect NMSE=sum(effect error²)/sum(true effect²)；late gain 为脉冲5步之后对真值的投影系数。

下表 mean ± sample SD over 3 independent training seeds；先对3个 pulse位置平均，再跨 seed 汇总。

| Model | Parameters | Factual MSE | Effect NMSE | Late-tail gain |
|---|---|---|---|---|
| gru | 14881 | 0.000264025 ± 3.56e-05 | 0.00779388 ± 0.00374 | 1.03789 ± 0.0256 |
| worldts_e2e | 39361 | 0.0351417 ± 0.000357 | 2.4679 ± 0.223 | 0 ± 0 |
| worldts_fir | 40129 | 0.000460829 ± 0.000162 | 0.0573689 ± 0.0229 | 0.973956 ± 0.0171 |
| worldts_two_stage | 39361 | 0.042474 ± 0.000476 | 2.86223 ± 0.196 | 0 ± 0 |

四组 whole-plan positive-shift sign 都是100%；这是我们自定义的粗诊断，**不是 TSWM 论文的 MC 指标**。完整 pulse 原始量、training curves、峰值显存与训练秒数见 `outputs/runs/*/`。普通 FIR 只增加768参数，GRU更小；因此“增加动作记忆”作为新机制被淘汰。ARX discovery MSE≈0.0001863，effect NMSE≈1e-9。白激励 ARX 是控制，白激励神经网络没有执行，不能声称完整 excitation factorial 已完成。

## E3. 已知 coupling 不可识别性的复核

`outputs/coupling_probe.json`：2D controlled affine Gaussian，3训练样本量×3训练seed×3H，共27个描述记录。全局 action-dependent SO(2) rotation 精确保留 conditional kernel；最大 NLL 数值差约2.7e-15。N16384/seed303/H64 的 mean family width=.7580，64.84%轨迹的效应符号在此家族内不确定；4个固定 gauge 的符号分歧50%。ordinary 3-fit CF range均值=.02748，不能代表结构 coupling range。16个额外 paired simulator transitions 用普通 Procrustes 可恢复角度 .7；这是额外信息，不是免费或通常真实可获得的监督。

这复核的是 CoMP 等已知机制，不是新理论。只有一个 constant-angle 家族，不是全部 compatible SCM 的 sharp bounds。

## E4. 学得的非线性 Gaussian kernel

`outputs/neural_coupling_probe.json`：3 seeds，train12288/val2048；固定非线性 drift 的均值网络，以 val MSE 选 ckpt。一步 validation MSE=.01034/.01035/.01049，噪声基准=.01；真实均值 RMSE=.0170/.0182/.0200。dev256仅选一个全局 witness angle；confirmation1024评估。选择结果都是 -pi。kernel NLL invariance≈8.9e-16；反事实符号分歧57.6%/47.6%/36.5%。它是 sensitivity witness，不是正确 CF estimator。

`outputs/neural_range_refinement.json` 使用另一组全新1024轨迹，冻结相同 ckpt；7→256角度后，真CF覆盖率分别为59.77%/74.02%/68.36%。额外加入已知真角度的 **oracle diagnostic** 也不改变结果，排除单纯 coarse grid 解释。oracle true-kernel family含真CF100%。仍不能把 neural 结果单独归因为研究缺口；有限样本/优化 bias 是竞争解释，所以 E5/E6 使用全局最优 OLS 控制。

## E5/E6. 参数置信集与结构 coupling，独立确认

初始置信 tube screen：3样本量×64独立训练数据，各32查询轨迹，H8/32/64；已知 isotropic Gaussian noise。随后冻结 shared-parameter prototype，用**全新训练 seeds 与全新 query cohorts**重做同等规模 confirmation。总计两组各192个OLS fits，不能把每个H/查询当作独立模型重复。

下面是第二组预先指定 primary cell：N16384/H64，64拟合×32轨迹。family coverage 指完整 true global-angle effect interval 被包含；与单个真角度点覆盖不同。wrong signed claim 是断言正/负却不对整个 true family 成立的比例。

| Method | True-point coverage | Whole-family coverage | Mean width | Sign certifiable | Wrong signed claim |
|---|---|---|---|---|---|
| gauge_only | 89.50% | 15.72% | 0.75775 | 37.84% | 1.37% |
| joint_paired_tube | 100.00% | 100.00% | 1.11645 | 4.39% | 0.00% |
| joint_separate_tube | 100.00% | 100.00% | 1.18499 | 1.07% | 0.00% |
| joint_shared_parameter_tube | 100.00% | 100.00% | 1.07979 | 13.67% | 0.00% |
| linearization_without_remainder_ABLATION | 100.00% | 100.00% | 1.04389 | 17.33% | 0.00% |
| oracle_known_kernel_family | 100.00% | 100.00% | 0.75878 | 37.40% | 0.00% |
| statistical_only_fixed_common_noise | 43.26% | 2.39% | 0.23386 | 72.95% | 39.84% |

shared-parameter tube 相对 paired norm tube：宽度减少 **3.28%**（按独立训练重复成对bootstrap95%CI 3.14%–3.43%）；sign certification 增加 **9.28个百分点**（CI 8.06–10.50）。全部64拟合的32查询完整家族覆盖为100%。**这不是泛化到所有SCM/所有模型的100%保证**；理论保证只在明示假设下以训练样本置信概率成立。

**整体方法筛选失败**：预设门槛 coverage≥95%、sign增加≥5pp、width减少≥15%；前两项通过，宽度项失败（3.28%）。不修改门槛、不按测试结果换primary cell。N256/H64 joint tube宽度约13–17，相对oracle .759，说明小样本下非常保守；N2048也几乎不能可靠判断正负。线性特例可计算，但当前原型不足以解决完整问题。

## 验证与资源

`outputs/verification.json`、`verification.log`：5项关键测试通过，覆盖独立精确脉冲真值、split独立/可复现、Gaussian rotation likelihood不变、解析family极值与独立密集rollout吻合、置信集边界128个模型的外包区间检验。不是用单元测试证明全部数学性质；推导另见 `confidence_certificate.md`。

本轮12个初筛神经训练job累计约140秒；额外3个小型Gaussian均值网络训练与审计约7秒。confidence screen约5秒，shared confirmation约35秒（含区间计算）。这是小模型诊断，不是大规模benchmark训练成本估计。大权重/NPZ留服务器，紧凑结果同步本地，哈希见provenance。


## E7. 通用可行对手：区间宽度有多少是不可省的？

另一套全新数据，8个独立OLS fits、各16条H64轨迹、N16384。普通projected Adam在同一联合系数置信球内，6 restarts×200步，分别搜索效应下/上端；每个候选对时间共享同一个W，耦合仍为同一个全局angle。优化只看factual path，不读取真实CF。对给定W的angle极值解析计算，不需要搜索angle网格。

找到的**可行内范围**平均宽度0.93463，basic sound outer范围1.11427，shared-parameter outer范围1.07706。因此在这组查询与**同一不确定性集合**下，任何sound算法相对basic平均宽度的改善最多为16.12%；实际最优可能更小。不能从这个内范围声称已经找到了sharp外界或global optimum。

这限定了线性特例中继续做“更紧算法”的收益空间：预设15%目标已接近可行对手给出的上限。它没有证明一般非线性场景没有缺口，但进一步削弱了仅凭当前线性特例立项新算法的理由。所有记录、优化曲线在 `outputs/feasible_extrema.json`；所有端点witness参数和factual路径保存在服务器 `outputs/feasible_witnesses/*.npz`，逐个通过独立NumPy recurrence重放与置信球可行性检查。


## E8. 强通用基线：未修改 auto_LiRPA / CROWN

固定源码 `e3930353a3589f7d10879f1e71223d1eefb57196`，独立Python3.11.15 / torch2.8.0+cpu / numpy2.2.6 / scipy1.15.3环境。将同一个regression ellipsoid写成单一L2输入球，保留跨时间/跨分支参数共享；全局角度以64个连续区间覆盖[-pi,pi]。CROWN直接包围整个factual-abduction→counterfactual计算图；没有自定义relaxation、没有更换uncertainty set。每个worker的forward用独立NumPy recurrence复核。

H8 smoke：CROWN width=.68925，alpha-CROWN=.68223，专门shared tube=.71085。alpha耗时59.0s，对比CROWN .85s，额外tightness约1%。因此保留原V1计划，另冻结V2只扩展CROWN，避免重复低信息量alpha计算。这不等于证明alpha-CROWN或GPU验证不足。

长时域下，每个H计划为2个拟合×4个查询；下面按已完成worker汇总，**不是8个独立训练重复**。时间为CPU bound computation，未将专用NumPy批量实现与通用graph实现作严格速度宣称。

| H | Completed cases | CROWN mean width | Basic tube width | Shared Taylor width | CROWN compute seconds/case |
|---|---|---|---|---|---|
| 32 | 8 | 0.82688 | 0.88469 | 0.84708 | 9.92 |
| 64 | 8 | 1.05531 | 1.12736 | 1.08986 | 37.84 |

原始结果与逐worker日志在 `outputs/lirpa/`；全局角度、共享系数球及完整factual path语义相同。CROWN在已观察的匹配case中已是重要竞争者，不能再把“共享参数能让区间变窄”当作创新。当前原型还同时失败于其先前预设宽度gate，所以**淘汰当前原型作为新方法路线**。更一般的连续非线性问题尚未由本轮证明已解决，但其潜在存在也不构成已确认的Research Gap。

环境错误单独处理：初始NumPy2.4.6继承环境的eigh触发Windows DLL exception，改用新venv内的稳定binary wheels；constant buffer的batch维警告按通用库接口修复；它们都没有被计入科学失败或算法性能结论。


协议记录：native scalar screen的stage1实际上按latent validation loss选checkpoint，stage2和end-to-end组按factual validation MSE；初始config的统一描述过于宽泛。原config未篡改，所有差异、代码版本恢复与资源型计划变更均见 `protocol_deviations.md`。这不改变结构支持零或当前NO-GO结论，但不得声称各阶段选择目标完全相同。
