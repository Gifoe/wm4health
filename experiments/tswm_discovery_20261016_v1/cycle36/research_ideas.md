# C36 候选账本

更新 UTC 2026-10-16T22:15:52.350331+00:00。C36 是已完成的文献/源码及数据资格审计，整体目标 ACTIVE_UNACHIEVED，接受最终方向 0。本轮没有模型训练、预测误差或方法效果实验。

目标为可复用时间序列 world-model 的状态学习、条件转移与观测生成框架。数据集只是证据载体；允许分别训练，不强制共享零样本 checkpoint、action、因果/物理识别、Loop 或 LLM。

**已关闭候选：** 外部预报observer适应、冻结目标动力学的世界模型。它在当前分解下就是普通multilead calibration+frozen conditional generator+integration，没有可辨认的新目标；不改名为新框架。

**唯一保留的问题，未选为Idea：** 高维相关预报误差下，目标相关的有限情景表示与共享多步转移是否有样本/计算效率交互。不是声称首次不确定输入、状态、联合分布、可学习sigma points或情景压缩。最接近完整先例为DSSMF、KITE、UKF-L及任务相关scenario reduction；目标相关低秩map本身也不构成差异。

继续之前应先在TRAIN/VAL检验真实joint-error压力、strongdirect/普通calibration残差、K曲线和实际latency。没有这些压力证据，不启动notes中建议的24fits。Elexon一个GB聚合风电任务可核版本合同，单独胜利不能上升为通用world-model机制。GEFCom风/光可按原组织者合同作第二任务方向，不需要为每行捏造网络接收日志。

若压力不存在，立即转向新模型能力；若存在，先冻结最小公平比较再训练。不得通过增加模块、挑error子集、换日期/seed/rank或复用TEST来保持候选存活。这里保留一个可失败的问题，而不是推荐一个有顶会潜力已被证实的方法。
