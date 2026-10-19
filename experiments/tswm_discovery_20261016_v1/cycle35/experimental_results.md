# C35 实际机制pilot，V2为权威结果

未通过事先定义的双任务screen；停止当前具体分支保护组合的扩训，不改loss/字段/阈值挽救。 接受最终方向0，总体ACTIVE_UNACHIEVED。不是某两数据集优化任务；两任务仅用来筛掉不值得扩展的框架机制。

服务器RTX5090、float32、TF32关闭、确定性CUDA、Adam foreachFalse，seed2026351。WorldTS原生encoder/transition/decoder/loss的独立小配置component transfer；不是原论文表格或Pythia复现。固定E/D，复制P，共同batch计划、2000updates/arm，12arms fits加4个1000update teacher阶段，共28000updates。主运行385.572秒。

solar TRAIN 137655、VAL 206；jena TRAIN 36357、VAL 54。history168，H96，TRAIN/VAL/TEST=70/10/20，TRAIN-only scaling。缺失窗口排除但不删时间行；只有历史covariates，无未来covariate输入。所有分数均为VAL，TEST未评分。VAL既用于选模又用于探索判别，不能当独立确认或发表的TEST性能。

| Task | Arm | VAL MAE | VAL MSE at MAE-selected step | Step | Trainable params | Seconds |
|---|---|---:|---:|---:|---:|---:|
|solar|N|0.30010329|0.41777144|400|96288|23.93|
|solar|JZ|0.27031299|0.33607157|700|2075936|29.18|
|solar|JZ_N|0.26792143|0.34211557|1400|2075936|30.59|
|solar|PZ|0.27228040|0.33810981|1600|2075936|30.27|
|solar|PY|0.27887793|0.35513997|1300|2075968|27.43|
|solar|PZ_NC|0.30102482|0.41862406|400|2075936|29.90|
|jena|N|0.13854755|0.04361054|0|96288|23.23|
|jena|JZ|0.13854755|0.04361054|0|2549024|28.39|
|jena|JZ_N|0.13854755|0.04361054|0|2549024|29.41|
|jena|PZ|0.13854755|0.04361054|0|2549024|29.22|
|jena|PY|0.13854755|0.04361054|0|2549056|27.26|
|jena|PZ_NC|0.13854755|0.04361054|0|2549024|29.35|

- solar：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 2.366% / -1.627% / 9.549%；预列screen passes=False。
- jena：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 0.000% / 0.000% / 0.000%；预列screen passes=False。

- jena选择step0的arms：N, JZ, JZ_N, PZ, PY, PZ_NC；训练未改善所选VAL指标，零差异来自共同teacher初始预测，不能解释为不同方法收敛到相同解。

JZ_N与PZ的forward/loss在相同权重时相同，差异是fusion/latent loss是否更新P；PY同时改变输出位置并去掉latent辅助loss，因此它是强普通控制，但不能单独隔离latent几何。N/PZ/PY/PZ_NC在相同update的P逐点hash及numerical分数完全一致；不同arm所选best step不同，其best numerical score无需相同。共同冻结E/D是为隔离优化通路；它不覆盖WorldTS同时学习encoder与conditions的全部机制，失败不能证明covariate-conditioned world modeling无价值。

- solar：在JZ_N自己的所选step，与N同step比较的数值支路MAE变化为 -0.933%（正值更差）；固定TRAIN batch的五个梯度余弦为 0.4610, 0.3804, 0.0508, 0.1691, 0.4645。不是总体梯度分布或因果证明。
- solar：统一step2000的numerical MAE为N 0.30649836、JZ 0.30151334、JZ_N 0.30162983；JZ相对N变化-1.626%，JZ_N相对N变化-1.588%。所选step0不会掩去这些完整训练轨迹，但它们也不能取代预列VAL选模。
- jena：在JZ_N自己的所选step，与N同step比较的数值支路MAE变化为 +0.000%（正值更差）；固定TRAIN batch的五个梯度余弦为 0.6546, 0.0643, 0.7542, 0.3420, -0.1256。不是总体梯度分布或因果证明。
- jena：统一step2000的numerical MAE为N 0.14888102、JZ 0.21013599、JZ_N 0.14850502；JZ相对N变化+41.144%，JZ_N相对N变化-0.253%。所选step0不会掩去这些完整训练轨迹，但它们也不能取代预列VAL选模。

0/12 arms在最后500updates仍改善历史最佳VAL MAE超过0.5%。不能宣称充分收敛或把预算内失败扩展为模型类不可能。

独立复核：12所选checkpoint从raw窗口重建输入、独立native/adapter forward重放，最大预测差0、最大MAE/MSE复算差1.44416429e-08；数据/scaler、teacher/arm样本序列、选择step、初始state、冻结E/D与protected-P轨迹通过。独立复核耗时2.989秒。

下列未训练参考是完成拟合后的teacher胜任性诊断，未用于原screen设计、选checkpoint或改门槛：

- solar / last_value：VAL MAE 0.54257238，MSE 1.25048077。
- solar / daily_seasonal：VAL MAE 0.33624637，MSE 0.59321159。
- jena / last_value：VAL MAE 0.17634977，MSE 0.06539032。
- jena / daily_seasonal：VAL MAE 0.47599238，MSE 0.35321599。

曲线见outputs/figures_v2/learning_curves.png，原始VAL预测、完整曲线、schedule与provenance均保留。V1因为默认CUDA微小梯度漂移导致严格P轨迹断言失败，在solar6fits后自动停止；不纳入这些效果结论。V1源码/plan/权重/日志与实现诊断原样保留。V2不是盲于V1日志的独立科学确认，仅修正可复现执行。
