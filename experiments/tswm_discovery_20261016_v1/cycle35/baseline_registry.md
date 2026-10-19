# C35 来源与执行登记

| 对象 | 来源 | 状态 |
|---|---|---|
| Native WorldTS components | decisionintelligence/WorldTS revision269426ea6217d7152a244da0e209d922acd46d4f；18个源码SHA见source_manifest.json | 原模块执行、改任务/小配置；非论文性能复现 |
| Solar real series | autogluon/fev_datasets，revisionf71c0fff4cf81283a2c43e7f3a73aa4f9826aef8，solar_with_weather/15T | target + 9历史covariates；无future covariates |
| Jena real series | 同revision，jena_weather/10T | target_0 + 20历史字段；通道物理含义/单位不猜 |
| N/JZ/JZ_N/PZ/PY/PZ_NC | 冻结MECHANISM_PILOT_PLAN_V2.md | 两任务×6arms，单seed，TRAIN/VAL |
| 完整historical-only WorldTS / TimeXer / stack-gate | 潜在正式确认的必要controls | 本pilot未运行，不报告不存在的分数 |

solar TRAIN 137655、VAL 206；jena TRAIN 36357、VAL 54。原始数据复用C34已核验小型parquet，不重下大型数据或旧权重。所有source/data/config/schedule/checkpoint/array hashes在pilot_v2/summary.json、verification.json及OUTPUT_MIRROR_MANIFEST.json；pt checkpoints与raw parquet在服务器，紧凑结果与脚本双端保存。

分阶段teacher原生loss=10MAE+15cos+.1legacyVC；Stage2 prediction-MAE-only。本轮arm冻结common E/D，未实现完整WorldTS recipe或Pythia，不能将pilot名当已发表模型复现。PY比latent arms多32个projection参数，报告真实容量；未声称完全相等FLOPs。
