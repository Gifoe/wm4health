# C35 执行记账

1. 原始12项梯度/相同初始forecast/future-target不变性preflight通过。
2. Windows CIM首次启动在创建ShowWindow属性前失败，未生成任何训练；改为New-CimInstance -Property后启动成功。
3. V1 supervisor42564、launcher41000：solar teacher两阶段及6fits完成，保护P轨迹hash断言失败而停止；未跑Jena，未生成最终summary；旧pilot_v1和全部脚本保留。
4. 实现诊断明确默认CUDA首步梯度约4e-9差异会积累；独立12步deterministic控制恢复逐步相等。详见outputs/diagnostics三个JSON。
5. V2科学设计不变，仅独立脚本进程确定性；必须先通过双数据12步P轨迹preflight，再完整重跑。
