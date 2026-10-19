<!-- root-post-result-decision40 -->
主要发现：两种普通模型在两任务揭示32个真实观测后，固定tail的平均energy均改善；ETTh1上GRU强于本配置FlowNP32，Jena上FlowNP8已呈现可用的质量/成本折中。结果不支持一个普遍的“无法修正”瓶颈。完整解释见[科学裁决](SCIENTIFIC_DECISION.md)。

# C40 固定未来目标、真实观测揭示：普通基线实测

UTC 2026-10-16T23:59:42.542540+00:00。研究对象是可复用时间序列世界模型框架；数据集只作机制验证载体。本轮是普通方法筛查，不是提出了一个新框架。当前工作问题：同一学得状态能否同时支持未来推演和真实新观测下的修正。最终新机制仍未成立，接受最终方向0，整体ACTIVE_UNACHIEVED。

192历史/96future，所有queries固定B=future65:96，r=0/8/32真实观测，剩余gap采样，train-only normalization，32 iid draws。公平U-statistic energy主指标；下表均为探索性VAL/两seed均值。query计时为single-origin、32draws、GPU同步；AR包含本次cache建立或真实观测增量更新，FlowNP包含全部原生conditioning。

| data | ordinary model | flow steps | r=0 | r=8 | r=32 | query s |
|---|---|---:|---:|---:|---:|---:|
| ett | ar8mix | 0 | 0.22475 | 0.21951 | 0.20446 | 0.060441 |
| ett | flownp | 1 | 0.57229 | 0.55570 | 0.54583 | 0.004653 |
| ett | flownp | 2 | 0.40090 | 0.39231 | 0.36963 | 0.008766 |
| ett | flownp | 8 | 0.36773 | 0.36221 | 0.32634 | 0.036215 |
| ett | flownp | 32 | 0.37585 | 0.37307 | 0.34633 | 0.138104 |
| jena | ar8mix | 0 | 0.33166 | 0.28937 | 0.27446 | 0.062147 |
| jena | flownp | 1 | 0.32453 | 0.31324 | 0.30605 | 0.004667 |
| jena | flownp | 2 | 0.32171 | 0.30661 | 0.28806 | 0.009856 |
| jena | flownp | 8 | 0.31274 | 0.29497 | 0.27330 | 0.037120 |
| jena | flownp | 32 | 0.30465 | 0.28854 | 0.26935 | 0.148366 |


![Fixed-tail quality](outputs/analysis_v1/fixed_tail_quality.png)

所有origin/seed/reveal/step的5个指标见[CSV](outputs/analysis_v1/all_origin_scores.csv)。所有曲线、cap flags、实际训练/评测成本和cache增量成本见outputs/screen_v1/results.json。逐时阻断不确定性见[analysis.json](outputs/analysis_v1/analysis.json)，只作描述，不将重叠windows或两seed当独立重复。

两数据、两seed、固定有限预算、VAL反复用于选择；TEST未评分。VAL origins={'ett': 73, 'jena': 23}；达到训练上限时选中 2/8，最后一段VAL仍改善 6/8。不能将训练不足、普通AR误差累积或原生采样器开销称为新机制缺口。SFM、适当state-space和ordinary consistency等强普通控制尚未完成。

独立核验：20 raw NPZ、300 score cells，max差1.11e-16。独立pdist实现，校对原数据/时间戳/target坐标/split/selection/checkpointSHA；未独立重新训练GPU。Native API及training loss同随机源重放见qualification.json。
