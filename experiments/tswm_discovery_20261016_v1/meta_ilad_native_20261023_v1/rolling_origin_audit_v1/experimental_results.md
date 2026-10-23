# Complete rolling-origin execution

| 预测起点（秒） | 正确时钟，重编码 | 正确时钟，普通初态拟合 | 从0时刻持续携带状态 | 错误时钟负对照 | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.402757 | 0.263514 | 0.402757 | 0.402757 | 29.060561 |
| 0.5 | 0.403590 | 0.179727 | 0.276528 | 32.272216 | 28.295429 |
| 1.0 | 0.231418 | 0.185729 | 0.240858 | 33.664062 | 29.478633 |
| 1.5 | 0.338780 | 0.195311 | 0.241026 | 22.139109 | 33.633679 |
| 2.0 | 0.241771 | 0.202518 | 0.240186 | 0.241772 | 32.181220 |

All7500 query risks, native-bank/query RNG, exclusions, prefix argmins and within-training-range times independently verified. Carry/restart maxpixel3.55e-6; five native replays maxpixel3.10e-6. Full selection/log receipts retained.

| 起点（秒） | Prefix改善而future变差 / 300 | Future改善 / 300 | 最大单例future MSE增加 |
| --- | ---: | ---: | ---: |
| 0.0 | 117 | 183 | 0.343473 |
| 0.5 | 53 | 247 | 0.158344 |
| 1.0 | 105 | 195 | 0.518379 |
| 1.5 | 52 | 248 | 0.128515 |
| 2.0 | 116 | 184 | 0.260331 |

[All1500 rows](ALL_QUERY_RESULTS_V1.csv) and [full interpretation](../ROLLING_ORIGIN_FINDINGS_V1.md).
