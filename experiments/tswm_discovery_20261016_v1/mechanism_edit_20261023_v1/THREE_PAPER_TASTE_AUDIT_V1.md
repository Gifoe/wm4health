# 根据已有真实实验，对四步研究流程的审计

结论：已有源码执行和负面证据足以淘汰多条动机，但尚未证明一个需要新世界模型机制的失败。不能把“跑过三篇论文”写成“同任务完整复现三种SOTA”，也不能把实现错误直接升级成创新。当前接受最终新颖方向0。

## 1. 三篇的选择与执行范围

| 论文/底座 | 固定源码 | 实际执行 | 尚不能声称什么 |
| --- | --- | --- | --- |
| WorldTS |269426ea6217d7152a244da0e209d922acd46d4f| 原NP/PJM四个baseline单元，另有Loop、readout、Stage1、NESO审计 | 全论文复现、全部benchmark领先、循环推理有稳定必要性 |
| LatentTSF |7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2| 原ETTh1/Electricity十六单元；失败归因campaign另有30个新fit、24个冻结模型审计 | AE误差已被唯一归为信息丢失、需要新架构、dense origins普遍造成恶化 |
| TSWM recipe |2e5e48540ff0da48e8192163969a6c58df1d5a3e| C53四十个源码子集fit、C79八十次冻结重放 | 第三种独立网络、完整论文扫参、未来动作泄漏已发生、方向一致性是新问题 |

这是两个模型工作加一个benchmark/recipe。任务输入、方向监督和数据资格不同，不能合成统一SOTA榜。旧权重校验与重放不计新训练。第三个同能力、同信息协议的独立强模型，尚未被这份主三篇比较充分建立。DALI目前是额外原生核心transfer资格实验，不自动填补完整第三种SOTA复现。

各源码、配置、seed、运行和错误由[注册表](../three_baseline_failures_20261023_v1/baseline_registry.md)与[闭合证据](../three_baseline_failures_20261023_v1/outputs/campaign_synthesis_v4/experimental_results.md)定位。

## 2. 哪些失败已经不值得立题

| 真实现象 | 排除/对照结果 | 研究决定 |
| --- | --- | --- |
| LatentTSF的Electricity部分horizon偏离论文配方结果 | 官方decoded-loss控制相对raw的Val均值差H336为+1.35%、H720为-1.03%，均进入预注册5%范围；Test残差仍保留 | 停止以该Val差距为理由增加架构；不能通过Test重开分支 |
| WorldTS Loop/状态修正收益不稳 | 既有低维普通校正已解释部分收益，E1最佳或gradient cosine不足以证明状态信息丢失 | 不把多跑几层或latent对齐重新命名为必要机制 |
| TSWM准确率与机制方向可分离 | 原文已定义方向一致性及相应监督；源码重放未发现设想的future-action leakage | 两种解释都不能当新发现 |
| C71自由rollout较差 | 增加原生搜索预算显著改善；history-VAR2的H50平均MSE0.000232233，低于每颗神经seed及特权图诊断的均值 | 当前充分观测子集没有证明新架构必要性，停止追加该子集训练 |
| MTS3未来行动提前影响预测 | 普通时间索引控制使early-prefix改变量为0，且满足原注册factual门槛；额外响应误差仍未全闭合 | 普通因果对齐已推翻原方法动机；剩余误差不能直接归为新时间抽象需求 |
| LatentTSF潜态TTC很高 | 保持物理预测完全不变的原点平移可抬高TTC，打乱时间仍可接近1 | 指标绝对值不足以认证动力学；平移敏感性及中心化不是新world-model方法 |

见[C71终态](../cycle71/outputs/free_simulation_closure_20261023_v1/experimental_results.md)、[C88决定](../cycle88/final_research_direction.md)和[指标核验](../three_baseline_failures_20261023_v1/metric_validity_v1/outputs/experimental_results.md)。这些负面决定均保留，不为了产出idea重新调同一失败case。

## 3. 当前新增实验究竟验证到哪一步

DALI的五个原生记录入口已执行：部分入口的名义编辑为0、一个索引改了时间槽、一个编辑随时间变号；同时存在正确固定轴编辑的反例。它们是查询协议问题，有普通修复，不构成模型机制失败。原生跨模态loss还有目标状态/梯度路径差异；原生JAX与独立解析检查确认一个投影梯度为0，尚未说明它导致预测问题。

真实原生核心factual transfer按独立协议执行。第一个运行完成500次更新后因我们的嵌套Optax保存错误退出；恢复从已验证step0开始。原运行全部记录点1/2/50/.../500的标量在恢复前缀中差值均为0。第500步恢复checkpoint及完整状态已保存，独立float64复算13个指标最大差4.44e-16。

| 第500步开发VAL，32步自由预测 | 原生核心MSE | 普通因果lagged MTS3参考MSE | 原生/参考 | 预设资格门槛 |
| --- | ---: | ---: | ---: | --- |
| family0 |0.6939109363|0.2196221964|3.1596|未通过|
| family1 |0.2103478048|0.0824945441|2.5498|未通过|

训练尚未完成固定2000次更新。不同架构/预算不构成公平效率排名；这里的参考用于阻止在factual能力不足时解释机制误差。时钟核验也尚未通过，训练结束后才用正确环境执行一次诊断。未访问干预真值或独立EVAL。详细门槛见[冻结协议](FACTUAL_FIT_PROTOCOL_V1.md)，当前运行状态以服务器终态和后续有日期快照为准。

## 4. 唯一保留的窄候选与竞争解释

候选问题：一个in-support的latent context编辑，能否由一个跨物理初始状态与行动查询复用的固定真实参数变化解释？当前尚无合格的训练后失败证据，更未接受新颖性。

| 竞争解释 | 必须做的反证 | 什么结果应停止方法路线 |
| --- | --- | --- |
| 查询构造错误或root/action时钟错误 | 显式轴、固定编辑、原生逐步时钟与无提前响应资格 | 普通修复消除现象 |
| factual学习不足 | 冻结预算/选择规则，先过两个family的普通预测门槛 | 未达门槛则不能归因机制表示 |
| context编辑离开可实现支持 | 使用真实推断的donor context并比较普通投影 | 普通in-support/投影控制充分 |
| 更换context同时改变了物理root | 检查root-state保留，加入普通重新过滤/状态transport控制 | 普通状态恢复解释差距 |
| 参数本来不可辨识 | 用显式系统辨识和可观测等价类检查信息条件 | 证据只能支持多个等价机制时，不能要求唯一参数语义 |
| 存在独立的全局机制缺陷 | 校准只拟合一个固定参数变化，冻结后换root和行动查询；加入同信息显式参数强基线 | 原生模型通过，或普通强基线/控制已闭合，就淘汰新机制动机 |

[Intervention Gap](https://arxiv.org/abs/2608.29998v1)和[Interventionally Consistent Surrogates](https://papers.nips.cc/paper_files/paper/2024/hash/26b8e3dc3a21fcd660d80c63b767f324-Abstract-Conference.html)已直接覆盖一般干预保真度与因果abstraction。新候选必须进一步区分机制参数编辑、局部状态编辑、监督条件和跨查询要求，不能只换评测名称。见[当前文献边界](MECHANISM_NOVELTY_SCREEN_V2.md)。

最有价值的下一步是完成已冻结factual资格及其终态复核。若门槛未过，现有证据不能支持一个新机制idea；若通过，再执行上述竞争解释的最小区分实验。当前不推荐新架构，不以增加实验数量替代这个判断。
