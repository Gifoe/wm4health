# 当前研究状态 — 2026-10-23 V4

总体目标ACTIVE/UNACHIEVED，最终接受新颖方向0。当前聚焦的三篇为WorldTS、LatentTSF和TSWM recipe；它们不能无条件称为三种SOTA或制作同输入统一排名。

## 本轮归因已闭合

30个新训练、24个dense-model审计和官方decoded/frozen-composition归因保持原计数。预注册Val门槛要求停止当前Electricity新增架构路线，18个条件扩模分支实际执行0。Test残差保留，不将其谎称完整解释或据此继续调参。

[三篇失败与研究判断](outputs/campaign_synthesis_v4/experimental_results.md) · [七项研究交付](outputs/campaign_synthesis_v4) · [冻结组合归因](outputs/transport_evidence_v1/INTERPRETATION_V1.md)。

## C71旧评估已完成

三颗原预算seed的12个神经评估臂与3个经典控制全部完成，独立复核通过。1000步图搜索显著好于100步，但history-VAR2的H50平均MSE仍低于每颗native/真实图特权seed。reserved EVAL未打开，不支持新增架构。旧RUNNING快照只是其当时状态。

[C71最终报告](../cycle71/outputs/free_simulation_closure_20261023_v1/experimental_results.md) · [全部独立复核](../cycle71/outputs/free_simulation_closure_20261023_v1/VERIFICATION71.json)。

## 指标诊断已完成

两份原生H96模型，0更新。平移潜态原点后物理预测最大差0，TTC和打乱时间TTC却可接近1。1160个scalar独立复核最大差1.56e−15。它说明单个几何分数不足以认证动力学，属于已知数学问题，未建立新方法、完整论文geometry表复现或创新方向。

[公式、采样和完整结果](metric_validity_v1/outputs/experimental_results.md)。

## 保存与研究门槛

保留全部旧证据，纠正当前入口；只发布紧凑代码/config/曲线/指标/报告。原始数据、权重、状态库、预测及长日志留服务器G数据盘，未同步本地。停止当前Electricity/spring案例追加训练，后续候选必须有独立可观测能力残差、现有方法边界与held-out最小方法实证。本轮未找到满足这些要求的最终方向。
