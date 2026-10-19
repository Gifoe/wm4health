# 时间序列 World Model：尝试汇总与证据边界

更新：2026-10-19。项目目标是学习可复用的状态、演化规律和条件模拟能力，贡献层次以 WorldTS 为参照。数据集用于检验机制；某个数据集上更低的 MAE/MSE 不单独构成研究方向。

**当前结论：已经积累基线复现、真实模型执行、机制审计与负面实验，但尚未验证一个足以立项的新研究方向。接受的最终方向为 0，研究目标仍未完成。** 本次交付是证据整理与归档，不是宣布科研目标完成。

导航：[逐阶段索引](RESEARCH_ATTEMPTS_INDEX.md) · [机器可读目录](experiment_index_20261019.json) · [归档范围与校验](ARCHIVE_SCOPE_AND_VERIFICATION.md) · [当前状态](../experiments/tswm_discovery_20261016_v1/CURRENT_RESEARCH_STATUS.md)。历史报告中的“继续”“待运行”和旧时间状态，以本页和当前状态为准。

## 1. 早期复现与 WorldTS 系列

| 尝试 | 实际执行和主要证据 | 判断 | 原始报告 |
| --- | --- | --- | --- |
| LatentTSF / DAG / WorldTS 三基线复现 | 24/24 配置完成并复核；21 次新训练、3 次核验复用。按该报告的复现容差，22 MATCHED、2 CLOSE。不是所有配置逐位复现 | 建立可用基线；不含新方法 | [复现报告](../experiments/baseline_reproduction_seed2021_v1/FINAL_REPORT.md) |
| Conditional Latent Loop | 30 次预测运行，另有 2 个验证集选定 AE；K=2 主决策与 K=4 敏感性分开。不同任务方向不一致 | **NO-GO**；不通过测试集选择深度挽救 | [报告](../experiments/conditional_latent_loop_seed2021_v1/FINAL_REPORT.md) |
| WorldTS Future-response + Loop | 与验证集选定强非循环对照相比，两任务平均相对 MSE 增益约 1.03%；并非所有配对种子同向，区块区间也未全部支持收益 | **INCONCLUSIVE**；没有稳定 Loop 优势 | [决策](../experiments/worldts_future_response_loop/NEXT_RESEARCH_DECISION.md) · [完整报告](../experiments/worldts_future_response_loop/FINAL_RESEARCH_REPORT.md) |
| Readout–latent alignment | 梯度、读出、普通 decoded supervision 的实际比较。NP 和 PJM 的收益方向不同；低余弦本身不等于优化冲突 | 对复杂 readout-aware loss **NO-GO**；普通监督 **INCONCLUSIVE** | [报告](../experiments/worldts_readout_alignment_feasibility_seed2021_v1/outputs/FINAL_RESEARCH_REPORT.md) |
| Stage-1 为什么 E1 最优 | NP/PJM 的轮末验证目标均 E1 最低，主要上升项为轨迹余弦；12 个 Stage-1 快照、126 个逐轮 Decoder 快照留有哈希。选模切片和时间隔离诊断切片的排序不稳定 | **NO_GO_NEW_STATE_SELECTION_METHOD**；不能由目标上升推出信息丢失或新损失必要性 | [报告](../experiments/worldts_stage1_state_selection_audit_seed2021_v1/outputs/FINAL_RESEARCH_REPORT.md) |
| NESO 在线 WorldTS | 真实 forecast vintages、模型训练、锁定评估均执行。6 系数线性修正解释 84.29% 的验证 MAE 收益；原生重算 p95 约 4.872 ms | 当前 V1 updater **NO-GO**。只有 303 个合格 Test 窗口，未完成原要求的完整 4–8 月覆盖，8 月无合格窗口 | [报告](../experiments/neso_worldts_online_v1/FINAL_REPORT.md) |

这里的三基线是原协议复现工作；Loop、读出、选模和 NESO 则分别属于新实验、机制审计或跨任务应用。不能把它们合并成“所有论文都已复现”。

## 2. 自主研究阶段：按问题整理

阶段编号用于追踪记录，**不等于独立实验数量**。目录覆盖 C1–C90 的 88 个阶段记录；C50/C51 没有独立目录，另保留两个 addendum。部分阶段只有文献、源码、数学或数据审计；部分存在多个训练与补充控制。所有细项见[逐阶段索引](RESEARCH_ATTEMPTS_INDEX.md)。

| 研究问题 | 阶段与代表模型 | 直接证据与限制 | 当前处置 |
| --- | --- | --- | --- |
| 概率状态是否丢失重要信息 | C2、C4–6、C11；RP-GSSM、KUMA、COFE、CHiLD、JEPA 组件 | 找到近似、实现或目标函数边界；普通似然、数值推断、矩控制或健康训练解释了多个案例。数学坏解不等于实际训练失败 | 未建立新架构必要性；[C2](../experiments/tswm_discovery_20261016_v1/cycle2/final_research_direction.md)、[C11](../experiments/tswm_discovery_20261016_v1/cycle11/final_research_direction.md) |
| Benchmark 是否真的测行动与反事实能力 | C3、C9–10、C72；EpiCF、DoTime、受控因果系统 | 审计输入信息、重复 recording、行动屏蔽、识别假设和真实 checkpoint。低反事实误差不能自动解释成干预机制正确 | 保留协议诊断；不把不可辨识目标或输入问题包装成新模型；[C3](../experiments/tswm_discovery_20261016_v1/cycle3/final_research_direction.md)、[C72](../experiments/tswm_discovery_20261016_v1/cycle72/final_research_direction.md) |
| 持久机制不确定性与联合未来 | C15–17、C23–25；MTS3、预训练 meta-ID、Explore-the-Context | MTS3 原生算子在指定高斯反例上出现严重欠覆盖；但完整学习模型未展示同样的校准失败，普通联合协方差修复也未改善该学习模型 NLL。belief 诊断的真实预测影响有正有负，部分来自先验误差 | 不能从算子反例推出普遍失败；不接受协方差或 belief 修复新架构；[C16](../experiments/tswm_discovery_20261016_v1/cycle16/final_research_direction.md)、[C25](../experiments/tswm_discovery_20261016_v1/cycle25/final_research_direction.md) |
| 自主长期模拟是否只拟合表面指标 | C18–19；DynaMix、普通向量场辨识 | 原生模型确实能在发布示例生成长轨迹；部分事件偏差存在，但普通历史条件辨识有较强解释力，网格敏感性也影响结论 | 未确立独立、普遍的动力学保真缺口；[C19](../experiments/tswm_discovery_20261016_v1/cycle19/final_research_direction.md) |
| 该保留哪些输入、该预测哪些状态 | C20–22、C32–35；WorldTS、条件 GRU、主动子空间、普通读出 | 瓶颈下会漏掉影响响应的 driver；增大状态、普通降维或直接预测能修复部分现象。真实风洞与 NP/PJM 读出控制未支持通用新框架；分支保护 pilot 未过双任务预设门槛 | 当前角色/target/readout 具体路线停止扩训；[C22](../experiments/tswm_discovery_20261016_v1/cycle22/final_research_direction.md)、[C33](../experiments/tswm_discovery_20261016_v1/cycle33/final_research_direction.md)、[C35](../experiments/tswm_discovery_20261016_v1/cycle35/final_research_direction.md) |
| 上下文适配是否有不可替代残差 | C26–31、C66–69；PCFM、NormWear-2、概率 ICL/系统辨识 | 原生接口、glue、训练或普通系统辨识解释多个初始异常；真实迁移结果有成功也有未复现部分。不能把修接口重命名为世界模型创新 | 尚无通过的通用机制；[C28](../experiments/tswm_discovery_20261016_v1/cycle28/final_research_direction.md)、[C29](../experiments/tswm_discovery_20261016_v1/cycle29/final_research_direction.md)、[C69](../experiments/tswm_discovery_20261016_v1/cycle69/final_research_direction.md) |
| 新观测能否真正改善同一未来的分布 | C36–40；KITE、Sundial、FlowNP、GRU mixture | Sundial 请求间 moment 不兼容是有限上下文诊断，尚不等于实际质量损害。C40 的 8 次学习比较显示两类普通模型在两任务获得 32 个真实新观测后均改善固定未来目标的 energy score | “现有模型普遍无法修正状态”不成立；尚未完成所有强基线比较；[C39](../experiments/tswm_discovery_20261016_v1/cycle39/final_research_direction.md)、[C40](../experiments/tswm_discovery_20261016_v1/cycle40/experimental_results.md) |
| 观测过程、记忆与查询预算是否需要新框架 | C41–49；HMM、NCDSSM、SPF、GRU、时钟/观测似然模型 | 因子化在人工 held-out 组合上有收益，但也有 seen 代价；查询层级、显式 delay、观测积分未得到稳定的新机制依据。C44 完整查询协议提前停止，不能写成全量完成 | 保留有界比较和负证据，不推荐采样 wrapper 或普通分解新命名；[C44](../experiments/tswm_discovery_20261016_v1/cycle44/final_research_direction.md)、[C48](../experiments/tswm_discovery_20261016_v1/cycle48/final_research_direction.md)、[C49](../experiments/tswm_discovery_20261016_v1/cycle49/final_research_direction.md) |
| 有限参考分布/真实记忆机制 | C54–61；RP-GSSM、普通 Gaussian likelihood、HystRNN/辨识控制、HSMM | 数学 population 控制、有限参考梯度和实际训练分开。已有 running/global normalization 控制修复指定非线性传感例；普通记忆原语与 HSMM 在各自限定任务中有效 | 泛化到新世界模型贡献的证据不足；[C57](../experiments/tswm_discovery_20261016_v1/cycle57/final_research_direction.md)、[C59](../experiments/tswm_discovery_20261016_v1/cycle59/final_research_direction.md)、[C61](../experiments/tswm_discovery_20261016_v1/cycle61/final_research_direction.md) |
| 块条件生成能否直接充当干预模拟 | C62–65；PRO-DYN、COFE、CSDI、causal MDN | 区分条件分布、干预分布和近似 sampler；普通 causal MDN 已接近受控真值。PRO-DYN 只完成 8/9 计划 fit；COFE C63 是未训练组件审计 | 不接受从默认实现/错误查询合同推导新机制；[C62](../experiments/tswm_discovery_20261016_v1/cycle62/final_research_direction.md)、[C65](../experiments/tswm_discovery_20261016_v1/cycle65/final_research_direction.md) |
| 观测接口和动力学能否组合复用 | C70–79、C84–85；ModularNRI、状态空间组件、EMPS/F16、校准与适配 | 多个单变量传感变化案例被普通非线性校准、历史对齐或直接适配解释。40 个既有 TSWM 模型的时序审计未见所寻 anticipation；有锚与无锚归因必须区分 | 当前受控修复路线不足以立项；C71 正式评估仍未完成；[C78](../experiments/tswm_discovery_20261016_v1/cycle78/final_research_direction.md)、[C79](../experiments/tswm_discovery_20261016_v1/cycle79/final_research_direction.md)、[C85](../experiments/tswm_discovery_20261016_v1/cycle85/final_research_direction.md) |
| 预训练动力学、局部机制与合成数据选择 | C80–83、C86–87；FIM-ODE、普通弱形式辨识、系数先验 | 实际运行公开约 13M 参数模型与多个控制；冻结 feature/readout 和局部组合的优势未建立。C87 原生适配与普通强先验比较存在数值参考资格未过的限制，不能作完全认证的连续时间结论 | 不继续用教程或精度探针挽救旧候选；[C81](../experiments/tswm_discovery_20261016_v1/cycle81/final_research_direction.md)、[C83](../experiments/tswm_discovery_20261016_v1/cycle83/final_research_direction.md)、[C87](../experiments/tswm_discovery_20261016_v1/cycle87) |
| 动作时序与多行动共享物理世界 | C88–90；MTS3、CSDI、ordinary transition mixtures | 普通滞后 timing 修复消除 MTS3 anticipation；更强响应指标未全通过。CSDI 未达到双系统 competence 门槛；普通 learned mixtures 的独立确认接近参考 | 停止当前 timing/coupling 新架构路线；不声称所有高维、部分观测和多行动世界模型问题已解决；[C88](../experiments/tswm_discovery_20261016_v1/cycle88/final_research_direction.md)、[C90](../experiments/tswm_discovery_20261016_v1/cycle90/final_research_direction.md) |

## 3. 最近的决定性证据

**C88：真实 timing 缺陷存在，普通修复足以否定当前新方法动机。** 原生 MTS3 的未来动作后缀编辑会改变较早预测，最大约 0.2813 个标准化单位；普通 lagged-manager 控制将该量降为 0，且两系统 factual MSE 的退化均未超过预设 5% 门槛。但是补充的全部八项 paired-response 条件没有全过，不能宣称干预保真已全面解决。[结果与裁决](../experiments/tswm_discovery_20261016_v1/cycle88/DECISION_RECONCILIATION88_V1.md)

**C90：没有资格把原生生成器失败解释成 coupling 缺陷。** 原生 CSDI 首颗 seed 完成固定 20,000 更新。Duffing 通过 CAL support 检查，reservoir 比率约 0.6627，未过 ≤0.5 门槛。按预先协议停止，后两颗训练 seed 与封存的原生比较 EVAL 没有执行；这不是三种子确认。[实验结果](../experiments/tswm_discovery_20261016_v1/cycle90/experimental_results.md)

普通控制另行冻结，在独立 baseline-only 确认中得到：

| 系统 | 精确参考 energy | 无监督 learned mixture energy | 有组监督的普通 library energy |
| --- | ---: | ---: | ---: |
| Duffing | 0.205339 | 0.207521 | 0.205355 |
| Reservoir | 0.027229 | 0.027415 | 0.027363 |

这些控制使用低维、完全可观测、Markov、有限机制/多项式结构的有利条件；有组监督 library 的信息也更强，不能列成与原生 CSDI 公平的容量/信息排行榜。这里的意义是：**当前任务尚不支持必须开发新 coupling 架构。** 同一组件跨计划采样按构造保留单计划边际；nearest-bin agreement 也不等于精确的共享物理证书。确认使用单独生成的 baseline 数据，不是失败后开放原生比较 Test 的替代说法。[独立确认协议](../experiments/tswm_discovery_20261016_v1/cycle90/ORDINARY_CONFIRMATION90_V1.md)

## 4. 尚未闭合的执行项

**C71：训练结束，正式评估未闭合。** 本次从服务器核查到训练 campaign 的退出码为 0，后续 free-simulation 评估队列退出码为 1；已不存在相应活跃 Python 进程。旧 progress 仍写“运行中”，不能采用它作为当前状态。三颗 seed 的最终训练凭据与各自有效更新数见[机器目录](experiment_index_20261019.json)。

这项评估故障不证明模型成功或失败。本次只归档状态；后续应先定位评估入口错误并沿用已冻结的比较协议，不重训、不改变原预算，也不据此创建新研究故事。C44 的未完成查询、C62 的未完成 fit、NESO 的覆盖不足和 C87 的数值资格限制同样保留。

## 5. 早期健康时序与非时序支线

仓库原有 8 个 VitalDB 诊断全部保留：[原始系列导航](VITALDB_HISTORICAL_INDEX.md)。它们涉及行动信息、支持几何、支持删减、长 horizon、transition grounding 和多时间尺度。它们属于具体临床观测数据的机制诊断，不能单独升级为通用时序 world model 的创新结论；事实轨迹分数也不等于因果治疗效应。

[Long-Horizon Loop Dependency Audit](../experiments/longhorizon_loop_dependency_audit_seed42_v1/COMPLETION_AUDIT.md) 也予以归档，但其 TaH/NESTFUL 任务属于工具调用/序列依赖审计，**不是时间序列世界模型 Benchmark**。官方 serving 的 Windows/Linux 依赖与 parity 限制保留；不计入上述时序模型的复现证据。

## 6. 如何使用这份归档

先读本页的判断，再按问题进入原始报告；需要核对某轮实际运行了什么时，查看该轮 `experimental_results.md`、协议、baseline registry、负证据及 closure/verification。历史候选标题表示当时的探索假设，不表示当前推荐。

当前不推荐继续扩大旧 Loop、readout、covariance、scalar sensor repair 或本受控 coupling 配置。下一项尚待执行的决定性工作是闭合 C71 已承诺的正式评估，再依据实际残差决定是否形成新的研究问题。若普通方法已经解释现象，应停止相应新方法主张；若证据不足，应写明不足。

原始数据、模型权重、预测数组和完整运行日志留在服务器。本次从服务器读取紧凑文本并直接发布 GitHub，未将这些 payload 同步到本地。代码/报告是归档，不声称整个历史项目均已从零重新运行；每个文件的内容哈希与原始/发布差异见发布清单。
