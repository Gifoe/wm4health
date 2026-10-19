# 已执行模型与协议

同一真实风洞，16随机游走记录+10波形记录，251016行；训练记录1–10、验证11–12、测试13–16。另8份相同采集配置波形为协议变化，2份osr_mic变化波形单列采集变化。64历史/16未来、stride16，共15584个窗口全部由原CSV独立重建，误差0。未来只有5个预排定控制路径；11传感器真值、未来dt、flag/counter不作输入。

| Model | Identity |
|---|---|
| worldts32 | 原生因果patch编码器、PatchTST预测器、patch读出；两阶段；32D/token；RevIN |
| worldts4 | 同上4D/token；RevIN |
| worldts4_fixed | 同上4D/token；关窗口RevIN，保留全局训练集scaler |
| worldts32_endtoend | 同一原生架构，直接联合训练预测MSE的目标控制 |
| gru64 | 普通条件GRU状态空间，64隐状态，末次观测残差读出 |
| mlp128 | 普通128宽MLP，完整历史/计划，末次观测残差读出 |
| ridge_arx | 8历史lag+16步计划；7惩罚由验证集选择 |
| persistence | 无拟合 |

前三种两阶段模型各3seed，另三种监督模型各3seed，共18神经拟合。全部世界模型/验证选择先于测试。源代码18文件固定，训练配置见EXPERIMENT_PLAN.md；新数据配置不是原论文各数据集调优设置。报告实际参数、更新和时间，不主张精确FLOP匹配。

4D/token不是4个数的世界状态：还保留64历史tokens、条件tokens、已知未来计划及RevIN的32个统计量。不能直接与前轮4D单时刻toy作容量等价比较。

主job耗时786.74s，峰值CUDA分配437.1MiB。原始数据/权重/预测/日志保存在服务器cycle22/outputs；本地为紧凑结果/报告镜像。
