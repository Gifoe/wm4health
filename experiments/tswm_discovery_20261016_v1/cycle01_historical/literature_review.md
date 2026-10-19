<!-- cycle49-final-index-start -->
# C49已完成：条件随机世界模型评分与固定时间预算比较

UTC 2026-10-17T14:53:33.295967+00:00。8组训练/18800正式updates；主评分16组原始轨迹/12288模型—历史—揭示分数，512条独立合成物理轨迹。独立NumPy重放全部预测及指标通过，同预算60/120/180秒规则在EVAL前固定并全部报告。模型、优化器、原始轨迹、日志和版本修复凭据保留。见[完整结果](cycle49/experimental_results.md)、[科学裁决](cycle49/SCIENTIFIC_DECISION.md)、[同预算结果](cycle49/outputs/cost_frontier_v1/RESULTS.md)。

本轮是原创受控实验，非原论文headline复现；0真实域。已有convMMD、条件路径评分和观测感知模型约束其创新边界。当前接受最终方向仍为0，整体ACTIVE_UNACHIEVED。TSWM原生缺失插值的构造边界测试发现输入契约风险，发生频率和实际效果未测，不能作为新模型贡献。以下为历史。
<!-- cycle49-final-index-end -->

<!-- cycle50-progress-index-start -->
# C49收尾：8组训练和主评分完成，独立复核与同预算比较继续

UTC 2026-10-17T14:26:16.045904+00:00。18800正式updates；主评分16组原始轨迹/12288模型—观测历史—揭示分数，对应512条独立合成物理轨迹、1536个配对观测历史。独立复核仍在执行，不作方法收益主张。验证器的大型压缩数组重复读取已按版本修复，原权重和主评分未重跑。

同预算60/120/180秒规则已在EVAL打开前留档。恢复队列自动继续独立NumPy复核、固定预算比较、分析与图。最新状态见[当前快照](cycle49/CURRENT_STATE_V2.md)及实际进程；旧训练快照作为历史保存。已有convMMD/观测感知生成推断使创新边界更紧，见[近邻挑战](cycle49/NOVELTY_CHALLENGE_20261017.md)。尚未建立新框架贡献，接受最终方向0，整体ACTIVE_UNACHIEVED。

以下为历史。
<!-- cycle50-progress-index-end -->

<!-- cycle49-progress-index-start -->
# C49：新的条件随机世界模型学习比较正在执行

UTC 2026-10-17T14:01:48.608749+00:00。整体ACTIVE_UNACHIEVED，接受最终方向0。研究目标仍是WorldTS贡献范围的可复用时间序列world-model框架。

C48已经完成16组训练/40750正式updates、44组原始预测/1694条模型—起点—揭示评分，独立CSV/模型/指标重算PASS。显式延迟没有稳定超越GRU；普通GRU/ridge也能利用新观测。该具体路线降优先级，结果见[完整结果](cycle48/experimental_results.md)和[裁决](cycle48/SCIENTIFIC_DECISION.md)。勿重启已完成队列。

C49实现“在条件随机世界模型训练中解析积分观测过程”与完整计数似然变分学习的shared-core比较。两组预测核心同为9458参数，变分组另有13944训练期识别参数。独立数学/梯度/似然/NumPy模型核查通过，12discarded updates不计正式训练。冻结8组新训练，当前完成3组；无EVAL收益主张。详见[协议](cycle49/PROTOCOL.md)、[当前状态](cycle49/CURRENT_STATE.md)、[文献边界](cycle49/RESEARCH_INTAKE.md)。

条件路径评分、观测模型积分等组件已有先例，具体框架收益与新颖性仍待证明。当前为两类独立合成过程的可证伪筛查，尚无真实域验证。历史索引如下。
<!-- cycle49-progress-index-end -->

<!-- cycle48-progress-index-start -->
# C48：真实过程的记忆机制比较已启动，最终方向仍未确定

UTC 2026-10-17T13:19:45.026138+00:00。上一轮C47为PROGRESS，本轮新增论文对应源码/求解器、独立数值资格和正式16-fit比较。研究单位仍是WorldTS层次的可复用时间序列world-model框架，数据只作机制证据。整体ACTIVE_UNACHIEVED，接受最终方向0。

两份Zenodo作者存档与98个Git blobs已逐项匹配。原生NDDE/求解器通过独立Heun递推和延迟梯度核查；CPU实际更快，24discarded updates不计正式训练。新的96h→48h协议使用ETTh1/Jena真实过程、相同允许信息；强对照含GRU及full-history ridge，NODE只作弱消融。此为跨任务适配，非论文headline复现或新方法。

此时16组队列已启动，完成4组；完整结果和EVAL独立重放仍待完成。不得把VAL、资格或局部进度写成最终收益。评估程序等待全部选择固定后才打开TEST。见[冻结协议](cycle48/PROTOCOL.md)、[真实执行状态](cycle48/CURRENT_STATE.md)、[机会图及创新边界](cycle48/RESEARCH_INTAKE.md)。精确当前进度以服务器supervisor/实际进程为准，勿因观察超时重启。

已有可学习延迟、state-dependent DDE、observer和自适应world model均纳入边界；不把增加memory本身当创新。C47当前Laplace方案与C46简单factorization继续降优先级，C44原查询协议保持INCOMPLETE、不重启。

以下为历史。
<!-- cycle48-progress-index-end -->

<!-- cycle47-progress-index-start -->
# C47：状态—观测时钟新目标完成反证，当前方案降优先级

UTC 2026-10-17T12:35:21.069093+00:00。研究目标仍是可复用的时间序列 world-model 框架，以 WorldTS 的贡献范围为参照。数据用于检验能力；不把具体数据集涨点当作最终贡献。整体 ACTIVE_UNACHIEVED，接受最终方向0。

本轮完成16个 fresh fits、21200正式updates；两种有限网格动力学、两个初始化、四种同参数学习目标。封闭评估54组raw paths /13824模型—起点—策略分数，对应512条独立物理轨迹及1536个配对观测历史。NumPy/SciPy独立重算全部评分、log-domain似然与状态—计数协方差通过，ES最大差4.09e-7。见[完整结果](cycle47/experimental_results.md)与[计数说明](cycle47/COUNT_GUIDE.md)。

完整随机时钟似然已恢复受限均值时钟丢失的效用。新Laplace/hybrid目标在OU无可靠未来质量收益，在双稳态两个种子中均更差：跨策略平均ES差分别+0.009927/+0.006875。降低当前固定feature-bank方案优先级；单步更便宜尚不能证明匹配总成本或高维扩展优势。见[科学裁决](cycle47/SCIENTIFIC_DECISION.md)。

真实执行的是固定1D grid、已知emission/初始先验下的受控筛查，非HyperNSDE完整复现或真实域框架验证。HyperNSDE已有unsplit随机时钟分支；PCF-GAN/SigTPP已有全局路径分布匹配。新颖性不能来自重命名普通组件。原始数据、权重、优化器、日志、图及SHA保留，未调EVAL。C44原查询协议仍明确INCOMPLETE，不重启。

以下为历史。
<!-- cycle47-progress-index-end -->

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

UTC 2026-10-17T11:08:54.020548+00:00。C44完整quality/VAL筛查完成，剩余query提前停止，原16-model查询协议明确INCOMPLETE；5完成组972rows只是部分证据。C45实际原生SSM资格与独立dense Gaussian核查完成，采样区别有ordinary控制；0新方法效果/0接受方向。整体ACTIVE_UNACHIEVED。见[C45本项](cycle45/literature_review.md)、[C44停止依据](cycle44/PRIORITY_DECISION.md)和[C45当前状态](cycle45/CURRENT_STATE.md)。以下为历史。
<!-- cycle45-progress-index-end -->

<!-- cycle44-progress-index-start -->
# 最新C44进展：框架目标保留，16组训练完成，独立评估未完

UTC 2026-10-17T01:53:27.482782+00:00。可复用时间序列world model是研究单位，数据集只是证据载体；当前强普通控制筛查不是新方法。整体ACTIVE_UNACHIEVED、最终接受方向0。同拟合law references/16估计replicates/真实总成本尚缺，不提前宣称收益。见[C44本项](cycle44/literature_review.md)、[范围](cycle44/FRAMEWORK_SCOPE.md)和[状态](cycle44/CURRENT_STATE.md)。以下为历史。
<!-- cycle44-progress-index-end -->

<!-- cycle43-complete-index-start -->
# 最新：C43新增SPF原生core资格和多尺度创新边界

UTC 2026-10-17T01:03:21.055878+00:00。29sourcefiles固定，100discardedpreflightupdates，9000schedule值独立核验，checkpoint/同尺度续接差0；正式比较fits0。多尺度粗采样已有完整近邻；内部crosslevel状态不同不等于质量失败。整体ACTIVE_UNACHIEVED、接受最终方向0。见[C43本项](cycle43/literature_review.md)及[科学裁决](cycle43/SCIENTIFIC_DECISION.md)。以下为历史。
<!-- cycle43-complete-index-end -->

<!-- cycle42-complete-index-start -->
# 最新：C42未知参数ordinary HMM组合学习筛查

UTC 2026-10-17T00:39:19.858998+00:00。32fits/48000updates与全部raw scores/selected models已独立核验；11等于ordinary structured HMM，当前没有独立new method或被接受最终方向。人工heldout评分，旧真实TEST未评分。整体ACTIVE_UNACHIEVED。见[C42本项](cycle42/literature_review.md)及[总状态](CURRENT_RESEARCH_STATUS.md)。以下是历史。
<!-- cycle42-complete-index-end -->

<!-- cycle41-complete-index-start -->
# 最新：C41联合状态模型资格与组合复用边界

UTC 2026-10-17T00:23:05.091583+00:00。LIFTS作者示例原生训练/接口通过；2304人工control rows/9216cells与12Gaussian未来law独立核验。旧真实TEST未评分，人工known-law controls有评分。当前候选是有锚的view–dynamics复用，方法效果与新颖性尚未建立；整体ACTIVE_UNACHIEVED，接受最终方向0。见[C41本项](cycle41/literature_review.md)和[总状态](CURRENT_RESEARCH_STATUS.md)。以下为历史。
<!-- cycle41-complete-index-end -->

<!-- cycle40-complete-index-start -->
# 最新：C40框架级问题与真实观测普通对照

UTC 2026-10-17T00:01:46.384048+00:00。8普通fits/16000updates，2任务2seed，独立raw-source/score/information核验；TEST未评分。研究单位为通用时间序列world-model框架；当前观察修正路线尚未形成新机制，接受最终方向0，整体ACTIVE_UNACHIEVED。见[C40本项报告](cycle40/literature_review.md)、[框架目标](cycle40/FRAMEWORK_DIRECTION.md)及[总状态](CURRENT_RESEARCH_STATUS.md)。下方为历史。
<!-- cycle40-complete-index-end -->

<!-- cycle39-complete-index-start -->
# 最新：C39跨查询概率律诊断与普通控制

UTC 2026-10-16T23:32:02.916653+00:00。864moment rows及256score cells独立复核；8contexts/2MC reps，SGD0、8AR解析fits，TEST未评分。当前仅保留固定tail真实reveal质量—成本问题，最终接受方向0，整体ACTIVE_UNACHIEVED。见[C39本项报告](cycle39/literature_review.md)及[总状态](CURRENT_RESEARCH_STATUS.md)。下方为历史。
<!-- cycle39-complete-index-end -->

<!-- cycle38-complete-index-start -->
# 最新：C37完整研究裁决与C38框架查新

UTC 2026-10-16T23:06:41.197043+00:00。整体ACTIVE_UNACHIEVED，接受方向0；不把站点优化或普通修复当新框架。见[C37独立结果与归因](cycle37/SCIENTIFIC_DECISION.md)、[C38本项报告](cycle38/literature_review.md)和[更新下一步](cycle38/POST_C37_DECISION.md)。下方为历史。
<!-- cycle38-complete-index-end -->

<!-- cycle37-complete-index-start -->
# C37完整学习证据已核验

UTC 2026-10-16T23:04:09.714426+00:00。12 neural/2 HGB fits、56 path arms，TEST未评分。整体ACTIVE_UNACHIEVED，最终接受方向0。见[本项报告](cycle37/literature_review.md)。下面为历史。
<!-- cycle37-complete-index-end -->

<!-- cycle37-live-index-start -->
# 最新过程：C37训练正在运行

UTC 2026-10-16T22:46:42.121573+00:00。两真实预报任务×两个seed×三个neural arms固定协议执行中，另有普通rank/CAL/HGB控制；最终效果尚未独立复核，接受方向0，总目标ACTIVE_UNACHIEVED。见[本项C37报告](cycle37/literature_review.md)和[总状态](CURRENT_RESEARCH_STATUS.md)。下面内容为各轮历史。
<!-- cycle37-live-index-end -->

<!-- cycle36-index-start -->
# 最新证据：C36，整体研究仍未完成

文献/源码和服务器一日发行数据资格审计已完成。当前observer-only提案与普通校准积分目标等价，关闭其新机制主张；没有新模型训练或方法效果证据，接受最终方向0。原生预报为小时粒度，24/48对齐不是缺半天。见[本项最新报告](cycle36/literature_review.md)与[总体状态](CURRENT_RESEARCH_STATUS.md)。下面内容均为各轮历史，旧成果保留。
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

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/literature_review.md](cycle29/literature_review.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/literature_review.md](cycle28/literature_review.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

# 文献、代码与创新边界（2026-10-16）

检索重点为 2026-04-16 至 2026-10-16；保留更早的不可识别性、系统辨识、可达性基础工作。以下是定向检索，不是系统综述或“没有找到即不存在”的证明。公开代码快照见 `outputs/source_registry.json`。论文证据、代码证据与本轮实验分别标明。

## 起点与排除范围

| 工作 | 已核查的内容 | 对当前判断的约束 |
|---|---|---|
| [LatentTSF](https://arxiv.org/abs/2602.00297) / [代码](https://github.com/Muyiiiii/LatentTSF) | 论文、README、AE/latent predictor 代码；旧复现报告 | AE 重建/潜空间预测本身不是新研究问题；本轮未重训 |
| [WorldTS](https://arxiv.org/abs/2609.31162) / [代码](https://github.com/decisionintelligence/WorldTS) | 数值 core、训练阶段、真实旧 checkpoint；本轮执行 | 本版本未来条件只经局部 encoder 与 decoder；该事实不能推广到所有 world model |
| [Accuracy and Mechanism Consistency](https://arxiv.org/abs/2610.01842) / [代码](https://github.com/HaochenZhang717/TSWM-Official-Repo) | 论文全文/实现；标题现为 On the Divergence of Accuracy and Mechanism Consistency in Time Series World Models | 已研究准确性与机制方向脱钩、方向监督、不同 plan encoder；不能把这一大命题重新命名为创新。代码默认 rho 与论文设置须显式对齐 |
| [RP-GSSM](https://arxiv.org/abs/2505.23569) / [代码](https://github.com/gatsby-sahani/rp-gssm) | 论文/README/JAX 实现 | 已研究不依赖重建的 latent dynamics；本轮未执行 JAX 训练 |
| [PRO-DYN](https://arxiv.org/abs/2507.15774) / [代码](https://github.com/ARBrachet/PRO-DYN) | 论文与代码结构 | 动态结构本身已有路线；未执行新复现 |
| [DoTime](https://arxiv.org/abs/2607.27263) / [代码](https://github.com/thummd/dotime) | prior、pairing、evaluation、PFN masking、suite 版本 | 默认 `generate_pair` 为 independent-noise interventional；显式 counterfactual 才共享噪声。新版 suite 已区分 level/effect sign，不能声称其未区分 |
| [EpiCF-Bench](https://github.com/complex-ai-lab/epi-cf-benchmark) | README/tree/数据协议 | calibrated ABM 的模拟反事实有明确结构真值；本轮未下载大型数据，未作排行榜比较 |
| [TIME](https://arxiv.org/abs/2602.12147) / [代码](https://github.com/zqiao11/TIME) | 论文/README/tree | zero-shot forecasting 不是干预或个体反事实真值 |

## 已直接淘汰的宽泛 Idea

| 先前可能的命题 | 直接近邻 | 为什么不能作创新主张 |
|---|---|---|
| 给条件预测器增加动作记忆 | [BeliefGraph-JEPA](https://arxiv.org/abs/2610.05409)；[Attenuated in-context identification](https://arxiv.org/abs/2610.08118) | 前者有持续的 latent action effect；后者已做动态响应诊断、激励对照、forced-system 微调及 ARX/N4SID/NARX 对照。本轮 FIR/GRU/ARX 也已修复 |
| 动作激励不足导致识别失败 | [On Identifiability of Controlled World Models](https://arxiv.org/abs/2607.22430) | 已有 excitation/identifiability 理论。白噪声神经网络 factorial 本轮未跑，不能把它写成验证结果 |
| 因果未来动作 mask | [Parallel Predictive World Models](https://arxiv.org/abs/2610.08627) | action-prefix parallel prediction 已有 |
| belief 而不是确定性 latent | [UWM-JEPA](https://arxiv.org/abs/2605.25313)、RP-GSSM、[MDN-VJEPA](https://arxiv.org/abs/2610.13956) | hidden-state uncertainty、multimodal/coherent rollouts 已有；后者提交日期应以原页面为准，本轮不依赖搜索摘要的异常日期 |
| 分离观测校正与自主传播 | [OSSM / Correct then Forecast](https://arxiv.org/abs/2609.33566) | observer / simulator 解耦已有 |
| 物理加残差潜状态 | [Phys-JEPA](https://arxiv.org/abs/2606.16076) | 不应重新包装为新结构 |
| 非平稳 Koopman / coarse-fine 潜动态 | [KUMA](https://proceedings.mlr.press/v306/xiong26j.html)、[COFE](https://www.ijcai.org/proceedings/2026/602) | 必须区分已有非平稳与多尺度建模 |
| 预测分布随新观测形成 martingale | [Calibrated Probability Forecast Sequences](https://arxiv.org/abs/2606.31621)、[MPNN](https://arxiv.org/abs/2506.11898) | forecast-revision consistency 已有，不是新原则 |

[Pythia](https://arxiv.org/abs/2610.05240) 与 [Meta-iLaD](https://github.com/GoForIt-YYB/Meta-iLaD) 也已筛查；后者所查公开仓库只有论文/海报而无可运行训练代码，不能写作已复现。综述条目深度不同：只有参与决策的近邻作了机制级阅读；不是所有论文都逐式复推。

## 当前反事实区间问题的最近竞争者

1. [CoMP, ICML 2022](https://proceedings.mlr.press/v162/foster22a.html) 已明确用条件相关 Gaussian latent rotation 展示相同拟合分布、不同反事实。**本轮旋转反例不是新发现**。代码固定快照已保存。
2. [Counterfactual Analysis in Dynamic Latent State Models, ICML 2023](https://proceedings.mlr.press/v202/haugh23a.html) 已作动态隐藏状态部分识别与优化界；不能宣称“首次动态反事实界”。
3. [CSM/APID, NeurIPS 2023](https://arxiv.org/abs/2306.01424) / [代码](https://github.com/Valentyn1997/CSM-APID) 已对连续结果部分识别，利用 level-set curvature 与生成模型拟合。本文主要处理二元 treatment/连续 outcome 的静态查询；代码 curvature 实现断言 dim_u=2。零曲率也不能随意解释成唯一点识别。它不是现成的冻结序列模型置信区间接口。
4. [Robust Counterfactual Inference in MDPs](https://arxiv.org/abs/2502.13731) 已对 compatible SCM 的离散动态反事实给界。
5. [Lally et al., 2026-10-14](https://arxiv.org/abs/2610.20210) / [代码](https://github.com/jessicalally/robust-cf-inference-beyond-mdps) 已联合处理 transition 统计不确定性与 causal ambiguity，构造 IMDP 并作悲观 value iteration。**联合处理两类不确定性不能作为我们的首创**。论文定义离散 state space；本轮源码矩阵/Julia IMDP 路线也吻合。未执行其完整实验，未声称超越其结果。
6. [Twin Rollouts](https://arxiv.org/abs/2608.08982) 已有共享噪声的视频分支；共享 seed 本身不是创新。
7. [Flow*](https://github.com/chenxin415/flowstar) 已有 Taylor-model flowpipe、symbolic remainder 与参数依赖的可达性计算；本轮 first-order + remainder 是这些标准思想的特例实现，**不是新验证理论**。尚未完成与 Flow*/CORA 之类成熟工具的预算匹配比较。
8. [Non-rectangular robust MDPs](https://arxiv.org/abs/2305.19004)、[其 Lp 对偶方法](https://arxiv.org/abs/2502.09432) 已研究共享模型不确定性与 rectangular approximation 的困难；参数跨时共享本身也不是创新。

## 剩余问题与创新置信度

尚值得检查的具体交集是：**对冻结的连续、非线性动态模型，在明确的噪声耦合敏感性类中，同时保留 factual abduction、跨分支/跨时间共享参数，给出既有覆盖保证又能支持非平凡效应判断的区间。**

相对所核查公开实现，尚未找到能直接完成该接口的现成方法。但“没有找到”不等于证明不存在；通用可达性工具可能已经足够。当前问题证据强于方法创新证据。不得宣称首次、不应据此直接立项写顶会论文。下一轮必须把通用可达性/静态 APID 的动态组合当作竞争解释，而不是只和松的 Lipschitz 界比较。


新增强近邻：[auto_LiRPA / CROWN](https://github.com/Verified-Intelligence/auto_LiRPA) 已支持一般计算图、双变量乘法与sin/cos。代码固定后执行了同一共享L2置信球和连续角度集合的计算图验证；它不是只供文字讨论的近邻。当前专用Taylor原型不能凭参数共享或微小区间改善宣称新颖。Flow*/CORA仍未执行，但当前原型不需要等待更多对照才能停止推进。


附加创新排除：同一Gaussian kernel的旋转改变pathwise-gradient方差并非新原理。[ICML2018](https://proceedings.mlr.press/v80/jankowiak18a.html)和[AISTATS2019](https://proceedings.mlr.press/v89/jankowiak19a.html)已研究transport-equation的旋转自由度、控制变量与方差优化。本轮只作文献筛选，没有虚报planning-gradient新实验。
