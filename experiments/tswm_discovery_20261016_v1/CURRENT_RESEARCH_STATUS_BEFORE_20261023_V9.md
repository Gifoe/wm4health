# 当前研究状态 — 2026-10-23 V8

总体目标ACTIVE/UNACHIEVED，接受新颖方向0。

## 四步流程的严格判断

[三篇真实运行、失败归因与立题门槛审计](mechanism_edit_20261023_v1/THREE_PAPER_TASTE_AUDIT_V1.md)明确：目前主线是两个模型工作加一个benchmark/recipe，不能称为同任务完整复现三种SOTA。已有普通loss、推断、时间对齐和表示指标控制已推翻多条新架构动机；这些负面结果继续保留。

## 新增原生核心训练证据

DALI的实际source-based factual transfer第500步恢复checkpoint及完整状态成功保存，原运行全部12个记录点的标量与重放一致。两个family的32步MSE分别为0.6939109363和0.2103478048；普通因果lagged MTS3参考为0.2196221964和0.0824945441，尚未通过预设factual门槛。这个参考不是公平预算/效率排名。训练继续原2000-update协议，未使用干预或独立EVAL。

[独立float64指标复算](mechanism_edit_20261023_v1/factual_metric_verification_recovery_v1/VERIFIED_STEP0500_V1.json)13个指标最大差4.44e-16；[有日期的机器快照](mechanism_edit_20261023_v1/CHECKPOINT500_INTERIM_SNAPSHOT_V1.json)记录发布范围和完成步数下界，500步不是最终结果。

已启动耐久post-fit校验队列，观察实际训练PID；终态后复算全部保存指标、比较投影变化、执行一次已登记的时钟诊断。它不自动重训，也不把失败解释为物理机制缺陷。时钟与root-state尚未资格完成，不能进入机制失败结论。

候选仍是跨root/action的固定机制编辑语义，未建立新颖性或方法收益。先行工作与普通控制边界见审计表。所有数据、完整模型、优化器PyTree与环境留服务器G，未同步本地。
