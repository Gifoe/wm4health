# C44 Baselines and reproducibility

UTC 2026-10-17T11:05:22.791241+00:00。整体ACTIVE_UNACHIEVED，最终接受方向0。正式16 fits/55500updates、16模型held-out full-law评分及1024origins独立重算完成，truth reference为REFERENCE_READY。240个VAL配置中MLMC/antitheticMLMC选择均为0。因当前路线缺少新增框架机制与必要性证据，停止剩余重复查询采样；5/16完整model groups的972行查询数据已独立核查，原完整查询协议明确INCOMPLETE。停止依据在选择子集的query effects评分前记录，全部权重、原始文件和未完成组产物保留。

用户明确目标为可复用的时间序列world-model框架，WorldTS作为贡献范围参照。单数据集提分、ordinary采样器包装或implementation repair不足作为最终贡献。具体idea由机制证据决定，未要求新数学算子、Loop或固定某一数据集。

两种独立生成的noisy known-process carriers，各两初始化，四个完整模型：原生SPF、仅条件packing启用梯度的ordinary控制、TRAIN-PCA上的两层GRU256/4-component full-covariance MDN、联合估计的LGSSM。所有模型只有8步历史观测；真值状态不提供给learner。来源commit、数据、信息、优化和selection规则见PROTOCOL.md。不同family目标/预算/容量不同，不宣称隔离架构优劣。

|case|family|seed|selected update|stopped update|nominal params|last-gradient params|train+VAL seconds|
|---|---|---:|---:|---:|---:|---:|---:|
|gaussian|ar_mixture|2026441|1000|4000|611388|611388|26.82|
|gaussian|ar_mixture|2026442|2000|4000|611388|611388|23.90|
|gaussian|lgssm|2026441|500|1000|49|49|76.24|
|gaussian|lgssm|2026442|0|1000|49|49|72.55|
|gaussian|spf_condition_trainable|2026441|1000|4000|9440132|1159940|142.25|
|gaussian|spf_condition_trainable|2026442|1000|4000|9440132|1159940|139.91|
|gaussian|spf_native|2026441|1000|4000|9440132|1155716|149.47|
|gaussian|spf_native|2026442|1000|4000|9440132|1155716|136.97|
|nonlinear|ar_mixture|2026441|4000|4000|611388|611388|23.97|
|nonlinear|ar_mixture|2026442|3000|5000|611388|611388|34.40|
|nonlinear|lgssm|2026441|0|1000|49|49|72.72|
|nonlinear|lgssm|2026442|500|1500|49|49|117.50|
|nonlinear|spf_condition_trainable|2026441|3000|5000|9440132|1159940|193.12|
|nonlinear|spf_condition_trainable|2026442|2000|4000|9440132|1159940|156.40|
|nonlinear|spf_native|2026441|3000|5000|9440132|1155716|171.73|
|nonlinear|spf_native|2026442|2000|4000|9440132|1155716|151.84|

Selected checkpoints与last states、固定minibatch indices、曲线与SHA都在outputs/fits_v1。原生SPF继承的unused parent参数保留并计入nominal；不从inner width推断参数数。原论文气候headline未复现。
