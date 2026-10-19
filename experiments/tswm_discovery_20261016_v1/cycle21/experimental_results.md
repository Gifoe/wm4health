# Independent-cohort results

主候选在执行前固定为projected absolute future-bank；response_only是预先列出的竞争机制。即使后者表现更好，也只能形成下一轮机制假设，不能把主候选改名后称为事先确认。所有配置均完整报告。

**实质判断：** 主候选未取得稳定的预期结果。在前三类独立cohort中，projected_bank的gain准确率均值分别为49.95%/50.12%/67.88%，前两族响应NRMSE接近1。response_only三个系统的9个初始化全部gain准确率100%，物理MSE均值0.00232/0.00122/0.00219，响应NRMSE0.06594/0.08962/0.06567；这是差分目标有价值的开发证据。队列族却是普通obs_budget更好：物理MSE0.00514、响应0.07618，而response_only分别0.03420/0.20134。因此不能把差分目标宣称为统一获胜方法。更强随机控制有部分收益，不能用旧的简化双头失败来概括所有随机方法。

- coupled_linear: teacher readiness=True, validation MSE=[0.001616, 0.001698, 0.001316]
- rotating: teacher readiness=True, validation MSE=[0.000986, 0.001024, 0.001037]
- saturating: teacher readiness=True, validation MSE=[0.002637, 0.002565, 0.002726]
- queueing: teacher readiness=True, validation MSE=[0.005275, 0.007198, 0.006087]

| Family | Method | Gain acc. | Inverse acc. | Physical MSE | Modulation NRMSE | Frozen-state next MSE |
|---|---|---:|---:|---:|---:|---:|
|coupled_linear|joint_budget|0.495361|1.000000|0.067831|0.999870|0.067275|
|coupled_linear|obs_budget|0.498739|1.000000|0.074549|0.999086|0.073424|
|coupled_linear|reconstruction|0.492350|1.000000|0.068522|0.999644|0.067818|
|coupled_linear|forecast_bank|0.503947|0.999674|0.072345|0.999741|0.072225|
|coupled_linear|response_only|1.000000|1.000000|0.002321|0.065935|0.001375|
|coupled_linear|projected_bank|0.499471|1.000000|0.067846|0.998466|0.067617|
|coupled_linear|random_projection|0.500407|0.999919|0.069830|1.000000|0.069371|
|coupled_linear|wide_budget|0.999919|1.000000|0.011019|0.169865|0.010109|
|coupled_linear|mop_budget|0.595215|0.986206|0.096609|0.955776|0.081880|
|coupled_linear|projected_mop|1.000000|0.996460|0.018181|0.306181|0.006570|
|rotating|joint_budget|0.511637|0.996948|0.020816|1.018321|0.020785|
|rotating|obs_budget|0.699626|0.997965|0.019722|0.883885|0.018655|
|rotating|reconstruction|0.511475|0.996582|0.021634|1.018151|0.021559|
|rotating|forecast_bank|0.503866|0.999593|0.021233|1.002941|0.021205|
|rotating|response_only|1.000000|1.000000|0.001219|0.089623|0.000865|
|rotating|projected_bank|0.501180|0.999797|0.020302|1.002015|0.020205|
|rotating|random_projection|0.500041|0.999390|0.020666|1.002054|0.020472|
|rotating|wide_budget|1.000000|1.000000|0.002451|0.142433|0.001378|
|rotating|mop_budget|0.687663|0.997640|0.018088|0.752886|0.016764|
|rotating|projected_mop|0.675700|0.994507|0.026502|0.754063|0.019260|
|saturating|joint_budget|0.501831|1.000000|0.086374|0.999422|0.085631|
|saturating|obs_budget|0.505290|0.999959|0.094790|0.998822|0.094029|
|saturating|reconstruction|0.509521|1.000000|0.086589|0.999214|0.086011|
|saturating|forecast_bank|0.506144|1.000000|0.087829|0.999909|0.087211|
|saturating|response_only|1.000000|1.000000|0.002189|0.065667|0.001362|
|saturating|projected_bank|0.678792|1.000000|0.058529|0.690547|0.057904|
|saturating|random_projection|0.503540|1.000000|0.087112|0.999869|0.086723|
|saturating|wide_budget|0.999878|1.000000|0.007016|0.101456|0.005147|
|saturating|mop_budget|0.692301|0.995158|0.086319|0.739884|0.070870|
|saturating|projected_mop|1.000000|0.999552|0.016323|0.305771|0.006983|
|queueing|joint_budget|0.883382|0.831136|0.052379|0.329951|0.072585|
|queueing|obs_budget|1.000000|0.840373|0.005139|0.076175|0.007125|
|queueing|reconstruction|0.885091|0.828776|0.054827|0.349128|0.074422|
|queueing|forecast_bank|0.773885|0.765544|0.138225|0.509425|0.114826|
|queueing|response_only|0.960042|0.822266|0.034199|0.201337|0.013224|
|queueing|projected_bank|0.943888|0.818970|0.024027|0.186264|0.012123|
|queueing|random_projection|0.792603|0.769735|0.110578|0.527790|0.104959|
|queueing|wide_budget|1.000000|0.836182|0.006772|0.084007|0.004797|
|queueing|mop_budget|0.928711|0.809814|0.078070|0.325408|0.071783|
|queueing|projected_mop|1.000000|0.820475|0.039230|0.270292|0.011261|

三个seed是相同数据cohort上的初始化变异，不是三个独立真实系统。gain为当前已提供、未来未知的二值随机驱动。Modulation NRMSE比较相同其余条件下两种支持内gain的物理后果；0为精确，1约为忽略差异。真值来自生成方程，不代表观测数据因果识别。queueing还有动作等价/裁剪，不能将全部误差解释为gain丢失。

## Mature reduction control

| Family | Output choice | Gain acc. | Physical MSE | Modulation NRMSE |
|---|---|---:|---:|---:|
|coupled_linear|all_outputs|0.499105|1.006682|0.999997|
|coupled_linear|action_outputs|1.000000|0.000361|0.001795|
|rotating|all_outputs|0.501872|0.899147|1.000001|
|rotating|action_outputs|1.000000|0.001102|0.003558|
|saturating|all_outputs|0.500081|1.001680|1.000000|
|saturating|action_outputs|1.000000|0.001667|0.004228|
|queueing|all_outputs|0.813477|0.445119|0.551855|
|queueing|action_outputs|0.999430|0.038702|0.170821|

动作输出版本共享候选目标选择成分；与全部输出版本的差异用于定位目标的作用。成功只证明旧压缩器足够，不证明目标选择新颖。保留失败、种子变异及不满足teacher readiness的系统，禁止测试后调rank、loss权重或decoder。

新cohort上，动作目标+成熟压缩器在前三族全部gain准确率100%，响应NRMSE均值0.00180/0.00356/0.00423；queueing物理MSE0.03870，仍不及普通观测预测0.00514。绝对投影future中的信息并非原则上不足：同一投影用于成熟敏感性压缩器可成功，而joint neural辅助目标失败，说明不能把现象单纯归因为目标没有信息。优化、目标权重与压缩器归纳偏置仍是竞争解释。

补充旧方法第一次启动在拟合前因模块别名被局部变量遮蔽而退出1。原始源码快照与失败日志保留；只修正命名，另起attempt1输出目录完成24次拟合和独立monomial重放。见CLASSICAL_CONTROL_REPAIR.md。主120-fit job和验证均一次exit0。

主job总耗时2478.98s，GPU峰值分配112.9MiB。数据哈希12、模型/预测各120、teacher12经复核；960个指标独立重算最大差1.49e-08；完整batch权重重放最大差5.96e-07。原始方程、观测逆变换、validation选择和测试时序另有检查。receipt范围是可重复性，不是新颖性证明。

![Independent cohort](outputs/analysis/response_target_comparison.png)

完整每seed数值、参数/时间、验证选择与逐项比较见原始results/fit_records和outputs/analysis/analysis.json。这里尚无真实数据、非线性观测、跨系统共享参数或闭环多步概率模拟结果。
