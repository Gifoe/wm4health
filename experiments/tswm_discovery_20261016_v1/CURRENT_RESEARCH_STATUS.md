# 当前研究状态 — 2026-10-23 V5

总体研究目标ACTIVE/UNACHIEVED，最终接受新颖方向0。三个主paper仍为WorldTS、LatentTSF与TSWM recipe；此前30个新训练、24个dense-model审计和C71完整评估的结论保持。

## 新的反证实验已完成

从既有C88响应残差提出“时间聚合是否看不见行动顺序”的假设。四个冻结模型、64条原开发轨迹、三个固定块顺序编辑加identity，0训练。实际原生MTS3抽象包含时间戳：重排行动可改变表示约0.275–0.702，而一起重排行动/时间戳只产生约1.79e−7数值差。四个基础预测均与旧数组逐位相同。全部24个非identity格都好于无响应参照，否定了朴素的顺序盲假设。

[完整结果与归因](action_order_probe_20261023_v1/outputs_recovery_v1/experimental_results.md) · [256个独立指标检查](action_order_probe_20261023_v1/outputs_recovery_v1/VERIFICATION_V1.json) · [冻结协议](action_order_probe_20261023_v1/PROTOCOL.md)。第一次模拟前缀校验失败和数值接续修复独立保存，没有重训、换动作、调门槛或打开reserved EVAL。

剩余行动响应误差不等于一个新的信息瓶颈；一颗seed、1000原始更新和开发数据不足以唯一归因。普通时间对齐修复已解决的anticipation问题不重新立项。原Electricity、spring和timing方法路线保持关闭。

## 最近文献与下一候选

[15项检索、10项重点来源](action_order_probe_20261023_v1/literature_search/papers.md)确认：条件行动激励、干预保真、支持相关分歧、order-aware压缩已有很近的研究。泛称“world model不确定性”或添加signature没有直接新颖性。

保留一个待资格检查的问题：**同一latent context编辑，能否在不同初始状态和后续行动下，由一个固定的真实动力学参数变化解释？** 它来自之前未执行的DALI线索，不是本轮已证明的失败。需要先有准确的原生模型、初始状态保真和跨query固定机制测试，并排除普通manifold/projection/deep-conditioning控制。

DALI官方pin5ca393468defffe79e4e3537ba9766f6d31b510b已核查，当前tree/release未找到checkpoint文件或发布资产；既有服务器runtime没有JAX/JAXlib/MuJoCo/DM-Control，也没有WSL可执行文件。原生训练尚未运行，这些是可行性准备事实，不是模型失败或最终阻塞。[机器可读资格记录](action_order_probe_20261023_v1/DALI_FEASIBILITY_CURATED_V1.json)。

只发布紧凑源码、配置、错误凭据、指标及研究报告；原数据、状态、预测与权重留服务器G数据盘，未同步本地。当前没有得到满足新颖性和最小方法验证要求的最终方向。
