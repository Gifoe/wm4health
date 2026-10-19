# Negative evidence and unresolved alternatives

- 更大状态或普通完整观测模型可修复cycle20现象；没有普遍架构不可能性。
- 成熟敏感性压缩器配合动作输出目标已成功，新的神经压缩器不是必要结论。
- 同一绝对投影目标在敏感性压缩中可保留gain，在神经辅助目标中未稳定保留；不能宣称它在信息论上丢掉了因果驱动。尚未排除目标权重/优化与结构偏置解释。
- 这里把动作无关AR坐标视为相对于控制物理结果的干扰，并非声称通用预测应忽略它们。若任务要求预测完整观测，它们就是合法预测信号；只报告特权物理读出不足以证明通用world-model能力更强。
- WorldTS的命名条件接口已区分目标/条件；当前混合输入任务不能直接作为它失败的证据。
- 主候选projected_bank与response_only不同。不能用后者的正面测试结果替代前者的失败，也不能把普通重建、随机或规范MoP的成功删掉。
- projected_bank在coupled_linear/rotating未保留gain，在saturating有seed变异；原始“绝对未来投影比差分更好”的候选机制未获支持。response_only在前三族成功，但queueing由普通obs_budget明显超过；无跨机制统一胜者。
- 物理标签仅用于诊断，但诊断decoder/探针依然是特权评估，不是无需标签的完整可部署读出系统。
- 线性正交混合、有限二值动作、已知当前gain构成受控问题，可能偏向global输出span。实际模态、非线性传感几何和未知未来条件尚缺证据。
- 队列机制有已知action-equivalence限制，不能归因到一个新机制。teacher若未通过预设readiness，则该族不能支撑方法收益。
- 3000更新与teacher+student各1500只控制更新数量，teacher训练/查询与辅助头都有成本；参数和耗时必须一起解读。
- 最终新颖方向仍为0；不把一次合成prototype当成通用时间序列世界模型论文。
