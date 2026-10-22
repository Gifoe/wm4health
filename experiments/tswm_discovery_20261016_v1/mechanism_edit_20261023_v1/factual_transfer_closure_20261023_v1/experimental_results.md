# DALI原生核心实际训练闭合：时钟通过，factual门槛未过

本轮是固定源码的WorldModel核心跨任务transfer，不是CARL强化学习或整篇论文复现。使用17,052,480个非optimizer注册数组元素、原生context Transformer/RSSM/decoder/目标/优化器；没有小网络替代。一个训练轨迹完成2000更新，第一次保存失败前的500次重算另计，实际optimizer调用2500。原始失败、空文件和恢复记录均保留。另有先前4次已返回、全部丢弃的核心资格调用；原生abort内部完成情况未知，不冒称额外成功调用。

## 四个固定验证点

| Step | family0 MSE | family1 MSE | Aggregate MSE |
| --- | ---: | ---: | ---: |
| 0 | 4.122723427 | 2.514941760 | 3.318832594 |
| 500 | 0.693910936 | 0.210347805 | 0.452129371 |
| 1000 | 0.594761599 | 0.327539924 | 0.461150762 |
| 2000 | 0.511075073 | 0.158095442 | 0.334585258 |

以开发VAL aggregate mean-forecast MSE选中step2000。两个family分别为0.511075073和0.158095442，普通因果lagged MTS3参考为0.219622196和0.082494544，相对比2.3271和1.9164，均超出预注册1.25倍门槛。双方容量、函数类和训练样本预算不同，此表不是公平效率排名。原生4-draw平均与每draw风险、重构误差均保留，未按干预或Test调参。

## 已排除的实现解释

四个验证点的52个指标独立float64复算，最大差1.78e-15；训练原前缀12个日志点重放差0。完整PyTree保存/读回恢复已通过，数据窗口/目标/源文件SHA重验。时钟诊断中65步filter和32步imagination与同RNG原生逐步调用的所有状态、解码输出完全一致；改变未来动作后前24步输出不变。11项保存数组比较又经独立Torch/NumPy复核，差均为0。这只资格核验我们的时序合约，不证明物理预测正确。

原生跨模态正向投影kernel/bias在实训2000步后仍完全不变；反向投影、decoder及其他98组参数数组有变化。原生梯度与独立解析检查已确认该stop-gradient路径，但它尚未被唯一识别为factual误差的原因。论文与当前代码的目标状态/梯度差异见[源目标边界](../PAPER_SOURCE_OBJECTIVE_BOUNDARY_V1.md)。

## 明确决定

这份checkpoint未达到机制归因的factual资格。停止用它支持全局机制编辑失败或新架构动机，停止追加当前配置训练；不运行物理参数编辑或打开独立EVAL。训练预算、目标差异、跨任务迁移与函数类仍是普通解释，当前未分离它们。不是一项已验证的DALI论文能力缺陷。

Research Question：同一context编辑能否对应跨物理root/action复用的固定机制变化？本轮尚未进入这个检验。

Evidence：[训练终态](../factual_fit_recovery_v1/RESULTS_V1.json)、[终态核验](VERIFICATION_V1.json)、[时钟](../factual_clock_step2000_diagnostic_v1/RESULTS_V1.json)、[实际投影变化](../crossmodal_checkpoint_comparisons_v1/STEP2000_V1.json)。

Existing Methods：普通配方、状态重新过滤、显式参数辨识、context投影和干预abstraction仍须先排除。最新可执行的Meta-iLaD来源和相关辨识理论应作为直接先行研究，而不是只比较失败的迁移配置。

Proposed Idea：无接受的新机制。Falsification：若现有已辨识模型或普通控制闭合问题，淘汰动机；未通过factual门槛也不构成机制失败。

Feasibility：当前原生核心CPU执行可行、来源和状态可复核；这个预算下的factual质量不足。全部模型/数据/二进制状态留G。

Next Experiment：先在有真实参数、可控初始状态且原生factual能力合格的模型上资格验证固定机制语义，优先核查现成Meta-iLaD/IDF辨识方法是否已满足。其原生物理数据生成与执行依赖须先确认。当前总体目标ACTIVE/UNACHIEVED，接受最终方向0。
