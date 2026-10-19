# VitalDB 历史诊断入口

以下 8 个实验已在仓库中，保持原文件不变。它们是历史观测性健康时序研究，不代表已经完成通用时间序列世界模型选题，也不识别因果治疗效应。

| 项目 | 报告 | 阅读边界 |
| --- | --- | --- |
| Intervention grounding | [完整报告](../experiments/vitaldb_intervention_grounding_seed0_v1/outputs/FINAL_REPORT.md) | 原提出缺口未获充分支持；设备 CE 不作预测输入/训练损失 |
| Prospective future actions | [完整报告](../experiments/vitaldb_prospective_action_seed0_v1/outputs/FINAL_REPORT.md) | future-action 信息确有用；不支持通用 controllability gap，收益受高给药速率片段影响 |
| Support geometry | [完整报告](../experiments/vitaldb_support_geometry_seed0_v1/outputs/FINAL_REPORT.md) | 状态距离与误差相关，但未超越 ensemble 风险指标形成稳定增益 |
| Controlled support removal | [完整报告](../experiments/vitaldb_intervention_support_shift_seed0_v1/outputs/FINAL_REPORT.md) | 预设 support-specific reliability gap 未获支持 |
| Multi-horizon headroom | [完整报告](../experiments/vitaldb_multihorizon_headroom_seed0_v1/outputs/FINAL_REPORT.md) | 有限 direct-prediction headroom；future factual anchor 不能隔离 rollout drift |
| Physiological grounding audit | [完整报告](../experiments/vitaldb_physiological_grounding_audit_seed0_v1/outputs/FINAL_REPORT.md) | current-state 与 predicted-transition probes 有差异，但 factual endpoint 可见性限制归因 |
| Transition-grounding viability | [完整报告](../experiments/vitaldb_transition_grounding_viability_seed0_v1/outputs/FINAL_REPORT.md) | 匹配辅助头和验证集 λ 选择；具体数值/判断以该报告为准，不升级为通用机制证据 |
| Multi-timescale feasibility | [完整报告](../experiments/vitaldb_multitimescale_dynamics_seed0_v1/outputs/FINAL_REPORT.md) | fast/slow 与容量对照；具体结论以报告为准，不把结构命名当作创新证明 |

[返回时间序列世界模型总览](TIME_SERIES_WORLD_MODEL_RESEARCH_SUMMARY.md)
