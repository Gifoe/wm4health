<!-- cycle46-progress-index-start -->
# C46：普通组合对照完整执行，简单神经分离方案未获支持

UTC 2026-10-17T11:59:10.469077+00:00。研究单位仍是可复用时间序列 world-model 框架，以 WorldTS 的贡献范围为参照。整体 ACTIVE_UNACHIEVED，接受最终方向0。

本轮完成8个 fresh fits / 27250 updates，完整 held-out joint-law 评分192组 / 15360 origins，独立 NumPy/SciPy 复核。未见时间组合 ES：最佳神经普通对照约0.5491，廉价 aligned PCA/LGSSM 为0.4409；双侧分离的额外收益在两初始化之间变号。TRAIN/VAL-only known-state probes也显示这批神经状态的线性读出明显较弱。它们不支持新框架主张，也不构成所有神经世界模型的普遍失败证据。见[完整结果](cycle46/experimental_results.md)、[诊断](cycle46/DIAGNOSTIC_INTERPRETATION.md)和[优先级裁决](cycle46/PRIORITY_DECISION.md)。

普通 joint-path adapter 已完成线性解析协方差和默认常数扩散非线性步长资格核查；原作者核心未改。所有数据、选中/末次权重、日志、样本和 SHA 保留。O1 同时改变 recognition/decoder 条件，且不同模型目标和参数量不同，不能把本表解释为孤立架构因果对比。

不继续包装简单 factorization 或为了让普通方法失败而加难度。下一步先核查观测采集策略变化与物理动态分离这一新线索的强近邻及信息条件；HyperNSDE/TESAR-CDE 已覆盖许多直接组件，尚无新方法效果。见[近邻](cycle46/literature_review.md)和[未选定研究线索](cycle46/RESEARCH_BRANCH_NOTES.md)。C44 原查询协议仍明确 INCOMPLETE，不重启。

以下为历史。
<!-- cycle46-progress-index-end -->

<!-- cycle45-progress-index-start -->
# 最新C45：查询路线降优先级，完整基线合同已核查，框架问题继续

UTC 2026-10-17T11:08:54.020548+00:00。C44完整quality/VAL筛查完成，剩余query提前停止，原16-model查询协议明确INCOMPLETE；5完成组972rows只是部分证据。C45实际原生SSM资格与独立dense Gaussian核查完成，采样区别有ordinary控制；0新方法效果/0接受方向。整体ACTIVE_UNACHIEVED。见[C45本项](cycle45/experimental_results.md)、[C44停止依据](cycle44/PRIORITY_DECISION.md)和[C45当前状态](cycle45/CURRENT_STATE.md)。以下为历史。
<!-- cycle45-progress-index-end -->

<!-- cycle44-progress-index-start -->
# 最新C44进展：框架目标保留，16组训练完成，独立评估未完

UTC 2026-10-17T01:53:27.482782+00:00。可复用时间序列world model是研究单位，数据集只是证据载体；当前强普通控制筛查不是新方法。整体ACTIVE_UNACHIEVED、最终接受方向0。同拟合law references/16估计replicates/真实总成本尚缺，不提前宣称收益。见[C44本项](cycle44/experimental_results.md)、[范围](cycle44/FRAMEWORK_SCOPE.md)和[状态](cycle44/CURRENT_STATE.md)。以下为历史。
<!-- cycle44-progress-index-end -->

<!-- cycle43-complete-index-start -->
# 最新：C43新增SPF原生core资格和多尺度创新边界

UTC 2026-10-17T01:03:21.055878+00:00。29sourcefiles固定，100discardedpreflightupdates，9000schedule值独立核验，checkpoint/同尺度续接差0；正式比较fits0。多尺度粗采样已有完整近邻；内部crosslevel状态不同不等于质量失败。整体ACTIVE_UNACHIEVED、接受最终方向0。见[C43本项](cycle43/experimental_results.md)及[科学裁决](cycle43/SCIENTIFIC_DECISION.md)。以下为历史。
<!-- cycle43-complete-index-end -->

<!-- cycle42-complete-index-start -->
# 最新：C42未知参数ordinary HMM组合学习筛查

UTC 2026-10-17T00:39:19.858998+00:00。32fits/48000updates与全部raw scores/selected models已独立核验；11等于ordinary structured HMM，当前没有独立new method或被接受最终方向。人工heldout评分，旧真实TEST未评分。整体ACTIVE_UNACHIEVED。见[C42本项](cycle42/experimental_results.md)及[总状态](CURRENT_RESEARCH_STATUS.md)。以下是历史。
<!-- cycle42-complete-index-end -->

<!-- cycle41-complete-index-start -->
# 最新：C41联合状态模型资格与组合复用边界

UTC 2026-10-17T00:23:05.091583+00:00。LIFTS作者示例原生训练/接口通过；2304人工control rows/9216cells与12Gaussian未来law独立核验。旧真实TEST未评分，人工known-law controls有评分。当前候选是有锚的view–dynamics复用，方法效果与新颖性尚未建立；整体ACTIVE_UNACHIEVED，接受最终方向0。见[C41本项](cycle41/experimental_results.md)和[总状态](CURRENT_RESEARCH_STATUS.md)。以下为历史。
<!-- cycle41-complete-index-end -->

<!-- cycle40-complete-index-start -->
# 最新：C40框架级问题与真实观测普通对照

UTC 2026-10-17T00:01:46.384048+00:00。8普通fits/16000updates，2任务2seed，独立raw-source/score/information核验；TEST未评分。研究单位为通用时间序列world-model框架；当前观察修正路线尚未形成新机制，接受最终方向0，整体ACTIVE_UNACHIEVED。见[C40本项报告](cycle40/experimental_results.md)、[框架目标](cycle40/FRAMEWORK_DIRECTION.md)及[总状态](CURRENT_RESEARCH_STATUS.md)。下方为历史。
<!-- cycle40-complete-index-end -->

<!-- cycle39-complete-index-start -->
# 最新：C39跨查询概率律诊断与普通控制

UTC 2026-10-16T23:32:02.916653+00:00。864moment rows及256score cells独立复核；8contexts/2MC reps，SGD0、8AR解析fits，TEST未评分。当前仅保留固定tail真实reveal质量—成本问题，最终接受方向0，整体ACTIVE_UNACHIEVED。见[C39本项报告](cycle39/experimental_results.md)及[总状态](CURRENT_RESEARCH_STATUS.md)。下方为历史。
<!-- cycle39-complete-index-end -->

<!-- cycle38-complete-index-start -->
# 最新：C37完整研究裁决与C38框架查新

UTC 2026-10-16T23:06:41.197043+00:00。整体ACTIVE_UNACHIEVED，接受方向0；不把站点优化或普通修复当新框架。见[C37独立结果与归因](cycle37/SCIENTIFIC_DECISION.md)、[C38本项报告](cycle38/experimental_results.md)和[更新下一步](cycle38/POST_C37_DECISION.md)。下方为历史。
<!-- cycle38-complete-index-end -->

<!-- cycle37-complete-index-start -->
# C37完整学习证据已核验

UTC 2026-10-16T23:04:09.714426+00:00。12 neural/2 HGB fits、56 path arms，TEST未评分。整体ACTIVE_UNACHIEVED，最终接受方向0。见[本项报告](cycle37/experimental_results.md)。下面为历史。
<!-- cycle37-complete-index-end -->

<!-- cycle37-live-index-start -->
# 最新过程：C37训练正在运行

UTC 2026-10-16T22:46:42.121573+00:00。两真实预报任务×两个seed×三个neural arms固定协议执行中，另有普通rank/CAL/HGB控制；最终效果尚未独立复核，接受方向0，总目标ACTIVE_UNACHIEVED。见[本项C37报告](cycle37/experimental_results.md)和[总状态](CURRENT_RESEARCH_STATUS.md)。下面内容为各轮历史。
<!-- cycle37-live-index-end -->

<!-- cycle36-index-start -->
# 最新证据：C36，整体研究仍未完成

文献/源码和服务器一日发行数据资格审计已完成。当前observer-only提案与普通校准积分目标等价，关闭其新机制主张；没有新模型训练或方法效果证据，接受最终方向0。原生预报为小时粒度，24/48对齐不是缺半天。见[本项最新报告](cycle36/experimental_results.md)与[总体状态](CURRENT_RESEARCH_STATUS.md)。下面内容均为各轮历史，旧成果保留。
<!-- cycle36-index-end -->

<!-- cycle35-complete-start -->
# C35机制pilot已完成且独立复核；最终方向仍为0

UTC 2026-10-16T21:51:04.613280+00:00。两真实公开任务solar/Jena、6arms、单seed、common native WorldTS E/D、paired P与batch schedule，12fits+4teacher阶段共28000updates。未通过事先定义的双任务screen；停止当前具体分支保护组合的扩训，不改loss/字段/阈值挽救。 全部是TRAIN/VAL探索，TEST未评分，不能当独立泛化或论文复现。研究层次仍是可复用时间序列world-model框架，不是这两任务提分。

- solar：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 2.366% / -1.627% / 9.549%；预列screen passes=False。
- jena：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 0.000% / 0.000% / 0.000%；预列screen passes=False。

V2独立raw-window、checkpoint/selection/schedule/hash与native-adapter重放通过：max预测差0，指标差1.44416e-08。受保护N/PZ/PY/PZ_NC的P在全部检查step完全相同。V1因默认CUDA微小梯度漂移在solar6fits后停止，旧源码/权重/日志保留；V2只修正进程内确定性，科学配置/阈值不变。0/12fits仍有cap改善，单seed/VAL复用/固定readout/辅助loss捆绑均限制结论。

见[实际结果](cycle35/experimental_results.md)、[研究判断](cycle35/final_research_direction.md)、[冻结V2方案](cycle35/MECHANISM_PILOT_PLAN_V2.md)。整体ACTIVE_UNACHIEVED，接受最终方向0。下面旧live/当前均为历史。
<!-- cycle35-complete-end -->

<!-- cycle34-qualified-start -->
# C34 已完成原生生成器资格与候选淘汰，最终方向仍未确定

UTC 2026-10-16T21:31:05.588651+00:00。Sundial固定公开128M权重已在服务器执行，3个fixture的same-noise原生重放差0，独立12项检查和梯度差分通过；全部model state不变。RTX5090缓存history后的1024paths/H96/50steps采样约0.08354秒。两份小型FEV真实序列与513MB权重正常TLS/SHA验证，依赖仅task-local。

first-hit-label proposal 的目标函数与既有conditional channel fitting逐项对应，撤回为新方法；没有启动该长prototype。资格/成本不等于新框架或动力学失败证据。上下文与数值分支的学习责任仅为另一个待证伪线索，未运行、未接受。研究层次仍是可复用time-series world-model框架，不以单数据集提分或冻结预测器query wrapper替代。

见[本轮实际结果](cycle34/experimental_results.md)、[方向判断](cycle34/final_research_direction.md)及三份framework视角补充。整体ACTIVE_UNACHIEVED，最终接受方向0；本轮短作业已退出，旧成果保留。下面旧“当前”均为历史。
<!-- cycle34-qualified-end -->

<!-- cycle33-complete-start -->
# C33已闭合：普通读出反证，当前不推荐新框架

UTC 2026-10-16T20:53:41.527173+00:00。C33全部24 fits、56组TEST输出、独立数值复核与来源闭合已完成；整体目标ACTIVE_UNACHIEVED，接受最终方向0。

本轮结论：**关闭当前target/readout线索作为新world-model框架的动机；不继续在NP/PJM上调这条路线。** 固定相同stage1后，普通global MLP已降低部分预测误差，因此此前误差不能直接证明latent信息不足。此为普通机制归因，不是新模型。

prediction-only相对mix的primary MAE下降：NP 1.557%，PJM 0.219%；PJM三个配对seeds并非全正。改用预列原生训练objective选模后，下降只有NP 0.513%、PJM 0.491%，不能把primary差异全归为辅助重构的系统性负迁移。

global相对native prediction-only的MAE下降：NP 5.600%，PJM 1.402%，各自三个decoder seeds同向。PJM的168-origin区块敏感性范围跨0，而且MSE上升3.809%；不声称跨系统稳定或全面改善。

加宽local的MAE变化很小且seed不一致。global同时改变非局部聚合、参数量及对已有RevIN统计的显式条件使用，因此不能单独归因为某一种新动力学机制。它没有增加观测，但与local的函数类/参数和计算不同。

全部训练1384.295秒，float32特征缓存15.962秒；20epochs、每epoch1141updates、每fit22820updates，24fits共547680updates。13/24 fits触发末5epoch仍改善>0.5%的cap标记，不能称充分收敛。

独立核验112个MAE/MSE、120条epoch顺序计划、24个初始化/参数/选模及48个选中checkpoint关联；指标复算最大差0.0。TEST特征/选中head独立重放最大绝对差1.43e-06/3.73e-07。TRAIN/VAL额外20项特征fixture和全量target、data/config/旧权重/来源SHA一致。

完整材料见[原生归因结果](cycle33/experimental_results.md)、[最终判断](cycle33/final_research_direction.md)、[冻结方案](cycle33/NATIVE_READOUT_PLAN.md)。C33训练及全部postprocess均已退出；以下旧live状态/旧“当前”仅为历史。
<!-- cycle33-complete-end -->

<!-- cycle32-latest-start -->
# 当前 cycle32：潜态目标与任务答案兼容性

UTC 2026-10-16T20:12:23.034221+00:00。已执行固定可逆target的原生loss组件数学诊断与独立复核；不等于实际WorldTS训练失败。普通修复已解本例，仍需原生残差证据。最终方向0，目标ACTIVE_UNACHIEVED。

见[实测](cycle32/experimental_results.md)、[方向判断](cycle32/final_research_direction.md)及[范围约束](cycle31/SCOPE_CORRECTION_20261017.md)。以下历史中的“当前”仅指当时，各轮原产物保留。
<!-- cycle32-latest-end -->

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
