UTC 2026-10-16T20:53:41.527173+00:00。C33全部24 fits、56组TEST输出、独立数值复核与来源闭合已完成；整体目标ACTIVE_UNACHIEVED，接受最终方向0。

本轮结论：**关闭当前target/readout线索作为新world-model框架的动机；不继续在NP/PJM上调这条路线。** 固定相同stage1后，普通global MLP已降低部分预测误差，因此此前误差不能直接证明latent信息不足。此为普通机制归因，不是新模型。

prediction-only相对mix的primary MAE下降：NP 1.557%，PJM 0.219%；PJM三个配对seeds并非全正。改用预列原生训练objective选模后，下降只有NP 0.513%、PJM 0.491%，不能把primary差异全归为辅助重构的系统性负迁移。

global相对native prediction-only的MAE下降：NP 5.600%，PJM 1.402%，各自三个decoder seeds同向。PJM的168-origin区块敏感性范围跨0，而且MSE上升3.809%；不声称跨系统稳定或全面改善。

加宽local的MAE变化很小且seed不一致。global同时改变非局部聚合、参数量及对已有RevIN统计的显式条件使用，因此不能单独归因为某一种新动力学机制。它没有增加观测，但与local的函数类/参数和计算不同。

全部训练1384.295秒，float32特征缓存15.962秒；20epochs、每epoch1141updates、每fit22820updates，24fits共547680updates。13/24 fits触发末5epoch仍改善>0.5%的cap标记，不能称充分收敛。

独立核验112个MAE/MSE、120条epoch顺序计划、24个初始化/参数/选模及48个选中checkpoint关联；指标复算最大差0.0。TEST特征/选中head独立重放最大绝对差1.43e-06/3.73e-07。TRAIN/VAL额外20项特征fixture和全量target、data/config/旧权重/来源SHA一致。

以下为原始完整自动汇总，数值未改。

UTC 2026-10-16T20:49:43.432475+00:00。整体通用时间序列 world-model 发现任务 ACTIVE_UNACHIEVED；接受最终方向0。

# 原生冻结读出归因结果

| 数据 | 普通读出 | MAE均值±seed标准差 | MSE均值 | 参数 | 选中epoch | 仍改善/cap |
|---|---|---:|---:|---:|---|---:|
| NP | native_mix | 0.213596 ± 0.001757 | 0.166540 | 115329 | [4, 3, 17] | 0/3 |
| NP | native_pred | 0.210271 ± 0.000168 | 0.164397 | 115329 | [20, 19, 18] | 2/3 |
| NP | wide_local_pred | 0.211097 ± 0.001039 | 0.165129 | 312705 | [20, 8, 7] | 2/3 |
| NP | global_pred | 0.198495 ± 0.000574 | 0.154074 | 924696 | [20, 19, 18] | 1/3 |
| PJM | native_mix | 0.142611 ± 0.000513 | 0.057258 | 352129 | [20, 15, 20] | 2/3 |
| PJM | native_pred | 0.142299 ± 0.000811 | 0.056741 | 352129 | [16, 15, 19] | 2/3 |
| PJM | wide_local_pred | 0.142156 ± 0.000497 | 0.056634 | 653857 | [16, 20, 20] | 2/3 |
| PJM | global_pred | 0.140304 ± 0.000544 | 0.058902 | 1678360 | [19, 16, 15] | 2/3 |

| 数据 | 固定reference | MAE | MSE |
|---|---|---:|---:|
| NP | original_stage2_reference | 0.213351 | 0.167257 |
| NP | last_value | 0.504966 | 0.700873 |
| NP | seasonal_24 | 0.367176 | 0.440462 |
| NP | seasonal_168 | 0.531223 | 0.774893 |
| PJM | original_stage2_reference | 0.142788 | 0.057516 |
| PJM | last_value | 0.425094 | 0.356139 |
| PJM | seasonal_24 | 0.224167 | 0.127280 |
| PJM | seasonal_168 | 0.321437 | 0.284492 |

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

正数表示候选MAE下降。范围是旧数据、固定checkpoint条件下的描述敏感性，不是跨系统显著性。oracle true-target-code reconstruction单列于机器结果，含未来真实编码，不能作为forecast或信息充足性的证明。所有24 fits先完成选模，随后统一打开TEST。
