# 训练前处理修订 v2：2026-10-17（测试性能查看前）

v1 的物理生成与分叉检查有效，但额外的 context std 归一化改变了跨系统风险权重。独立 train-only 审计按 seed1 的完整262144窗口日程计算：相对 context mean 的归一化目标能量均值34.094；最高0.1%窗口贡献95.7408%，最高0.01%贡献56.5998%。最小u-context std为0，最小y-context std约0.00877。这是目标/尺度问题的证据，不是新架构失败或训练后测试结论。

已在确认 owned worker pid41204 后停止 v1；保留全部数据、权重、优化器、日志、脚本及训练选择记录。v1 不进入正式方法比较，也不作为修订后方法优于修订前的性能证据。尚未运行任何模型的独立测试评估。

v2 使用原始生成器定义的 independent fixed-reference 单位，不再按短 context 做第二次中心化/标准化。训练、验证、测试对所有本轮训练模型均使用相同恒等变换；MSE 等于该固定单位下的预测 MSE。数据和划分不变、没有删除窗口、没有根据测试性能选择尺度阈值。公开预训练 ICL 作为外部参考仍使用其已验证的 context-only 归一化，记录它与新训练模型的预处理/预算差异。

v2 模型定义不变，文件复用并记录 v1 world_models.py 的 SHA。为避免把首轮仍明显下降的验证曲线当作能力上限，所有模型统一最多32768 steps、最少4096 steps；每512验证，连续6次未达到0.2%相对改善时停止。始终保存实际最小验证误差的 checkpoint；若到cap仍明显下降则继续标记 underconverged，不作架构必要性结论。batch、seed、optimizer、lr、损失类型、数据以及测试计划不变；三种子和四模型使用相同 v2 批次日程。

新输出为 outputs/world_training_v2 和 outputs/world_evaluation_v2；不会覆盖 v1。主执行协议 MODEL_COMPARISON_PLAN.md 的其他内容及 EVAL_MECHANISM_ABLATION_PLAN.md 继续适用。这里是训练协议修订，不是研究 idea；接受方向仍0。
