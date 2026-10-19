# C37 训练前读取器修正

原附件未改变。V1读取器假设小时1..24，原文件实际是0..23，因此自动停止。V2修正为原时间解析，按组织者声明的daily-midnight+24h合同，将valid_time-1h所属日作为1..24小时target block。没有凭空移动时间戳或发明UTC。该协议是原benchmark合同的迁移，不声称有逐行真实发行日志。

solar predictors15含有POWER列；只从独立train15取标签，future/history covariate严格白名单VAR数字列，POWER绝不入特征。没有读取solution文件做训练/调参。wind仅固定zone1和U10/V10/U100/V100。

V2的全量finite检查发现wind标签11个NaN，天气变量无缺失；solar选择范围标签/天气无缺失。V3保留NaN，后续按冻结规则排除涉及NaN的整个预测窗口，不插值。每分区记录候选/保留/排除数，不据误差重新挑窗口。V1/V2源码、失败输出目录均保留。

这些是训练前数据读取修正，不是模型改进或研究缺口。C36目标函数等价提案仍关闭。C37学习协议另行冻结后才训练。
