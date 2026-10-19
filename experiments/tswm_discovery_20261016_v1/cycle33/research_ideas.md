UTC 2026-10-16T20:53:41.527173+00:00。C33全部24 fits、56组TEST输出、独立数值复核与来源闭合已完成；整体目标ACTIVE_UNACHIEVED，接受最终方向0。

本轮结论：**关闭当前target/readout线索作为新world-model框架的动机；不继续在NP/PJM上调这条路线。** 固定相同stage1后，普通global MLP已降低部分预测误差，因此此前误差不能直接证明latent信息不足。此为普通机制归因，不是新模型。

A（future-target/report compatibility）作为当前最终方向关闭：数学问题有意义，实证不足且普通修复提供反证。B（mixed decoder监督）归因已完成，不能把alpha=0取名新方法。C（多尺度/随机target预算）仍仅是已有拥挤设计空间，尚无具体新交互，不推荐为替代标题。

prediction-only相对mix的primary MAE下降：NP 1.557%，PJM 0.219%；PJM三个配对seeds并非全正。改用预列原生训练objective选模后，下降只有NP 0.513%、PJM 0.491%，不能把primary差异全归为辅助重构的系统性负迁移。

global相对native prediction-only的MAE下降：NP 5.600%，PJM 1.402%，各自三个decoder seeds同向。PJM的168-origin区块敏感性范围跨0，而且MSE上升3.809%；不声称跨系统稳定或全面改善。

加宽local的MAE变化很小且seed不一致。global同时改变非局部聚合、参数量及对已有RevIN统计的显式条件使用，因此不能单独归因为某一种新动力学机制。它没有增加观测，但与local的函数类/参数和计算不同。

UTC 2026-10-16T20:49:43.432475+00:00。整体通用时间序列 world-model 发现任务 ACTIVE_UNACHIEVED；接受最终方向0。

# 候选判断

本轮没有新增可接受研究方向。通用world-model目标不等于把某个电力数据集的读出器换大。若普通控制解释观察，应终止该机制的创新主张。若仍有残差，先做同允许信息的observation predictor及收敛控制，再决定是否值得换独立领域；不能直接设计复杂架构。

| 数据 | 候选 / 基准 | MAE下降%逐seed | 汇总下降% | 168-origin区块+seed敏感性95%范围 |
|---|---|---|---:|---|
| NP | native_pred / native_mix | 1.88, 2.22, 0.55 | 1.56 | [0.5013039034507717, 2.8140605167796506] |
| NP | wide_local_pred / native_pred | 0.17, -0.68, -0.67 | -0.39 | [-1.0337380160630172, 0.12992407169213402] |
| NP | global_pred / native_pred | 5.83, 5.22, 5.75 | 5.60 | [3.524836027795887, 7.705446641436046] |
| NP | global_pred / wide_local_pred | 5.67, 5.86, 6.38 | 5.97 | [3.766127222106794, 8.174577105851185] |
| NP | native_mix / original_stage2_reference | -0.45, -0.72, 0.82 | -0.12 | [-0.9562533544973566, 0.742868075512079] |
| NP | native_pred / original_stage2_reference | 1.44, 1.52, 1.37 | 1.44 | [0.6248465177362619, 2.210080687877827] |
| NP | global_pred / seasonal_24 | 46.07, 45.77, 45.98 | 45.94 | [43.16678053174721, 48.84282386567495] |
| PJM | native_pred / native_mix | 0.72, 0.18, -0.24 | 0.22 | [-0.43271728580974955, 0.9374197922748021] |
| PJM | wide_local_pred / native_pred | 0.20, -0.58, 0.67 | 0.10 | [-0.6292250505580608, 0.8471966314878516] |
| PJM | global_pred / native_pred | 0.89, 1.47, 1.84 | 1.40 | [-1.2470976011829606, 4.013900590047899] |
| PJM | global_pred / wide_local_pred | 0.69, 2.04, 1.17 | 1.30 | [-1.3801657039942823, 3.854220101964211] |
| PJM | native_mix / original_stage2_reference | -0.09, 0.54, -0.07 | 0.12 | [-0.2448927108015501, 0.597143199264894] |
| PJM | native_pred / original_stage2_reference | 0.62, 0.71, -0.31 | 0.34 | [-0.44196783472938955, 1.1145194192688543] |
| PJM | global_pred / seasonal_24 | 37.26, 37.69, 37.28 | 37.41 | [32.398583652965144, 41.622354512788576] |
