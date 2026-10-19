UTC 2026-10-16T20:53:41.527173+00:00。C33全部24 fits、56组TEST输出、独立数值复核与来源闭合已完成；整体目标ACTIVE_UNACHIEVED，接受最终方向0。

本轮结论：**关闭当前target/readout线索作为新world-model框架的动机；不继续在NP/PJM上调这条路线。** 固定相同stage1后，普通global MLP已降低部分预测误差，因此此前误差不能直接证明latent信息不足。此为普通机制归因，不是新模型。

本轮四个head只构成普通诊断envelope，原stage2是预算/初始化不同的reference。global参数为NP924696、PJM1678360，分别约为原local的8.02倍、4.77倍；相同updates不等于相同compute。

全部训练1384.295秒，float32特征缓存15.962秒；20epochs、每epoch1141updates、每fit22820updates，24fits共547680updates。13/24 fits触发末5epoch仍改善>0.5%的cap标记，不能称充分收敛。

独立核验112个MAE/MSE、120条epoch顺序计划、24个初始化/参数/选模及48个选中checkpoint关联；指标复算最大差0.0。TEST特征/选中head独立重放最大绝对差1.43e-06/3.73e-07。TRAIN/VAL额外20项特征fixture和全量target、data/config/旧权重/来源SHA一致。

UTC 2026-10-16T20:49:43.432475+00:00。整体通用时间序列 world-model 发现任务 ACTIVE_UNACHIEVED；接受最终方向0。

# 原生与诊断协议

原始完整stage2仅作reference；新24 fits冻结原stage1，4种普通decoder×3 decoder seeds×2数据。native_mix与native_pred逐seed同初始化；20完整epochs，batch32，每fit22820更新，共547680更新。两臂之外的wide/global参数和计算不同。所有选模用validation forecast MAE；另完整报告训练objective选模控制。详见[NATIVE_READOUT_PLAN.md](NATIVE_READOUT_PLAN.md)。

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

原模型是source-native旧复现；新decoder矩阵是自主机制诊断，不是新增模型排行榜或完整官方训练复现。共享float32冻结特征cache成本另见native_cache_v1/complete.json，优化时间与显存见每fit result.json。
