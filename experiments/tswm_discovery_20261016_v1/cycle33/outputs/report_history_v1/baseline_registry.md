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
