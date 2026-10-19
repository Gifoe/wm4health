# 非线性多机制世界模型：实测与反证

**结论：不能把cycle24的标量失败推广为通用有害更新；本轮不支持开发新架构的必要性。** 原生更新确实改变了不该改变的真实物理预测律，但预测质量不一定恶化。必须区分这两件事。

六个原作者 validation-best final-budget 检查点（16维系统latent，100000训练步，各变体3个seed）；64个新物理系统；历史完整前缀0/8/32；两种probe；三次独立评估MC；918行预测、306行MC聚合。没有本轮训练。同一cohort被所有检查点共享，不能算成384个独立系统。源版本`f0baf6262523d9c5abaf8c0381b4db97818f7c34`。

科学指标是角速度CRPS、四个分支的联合energy/variogram、MSE和边际PIT覆盖率。四个查询共享同一个系统latent抽样。确定性cos/sin流形数据不使用full3D NLL作为科学评判；旧推断对照仍使用原decoder的full3D Gaussian似然，因此可以分开检验模型似然失配与推断失配。

## 冷启动，无信息probe

ΔCRPS=更新后减不更新；负值更好。区间先平均三次MC，再以64物理系统做4000次配对bootstrap；未做多重检验调整，单个checkpoint的区间不能当作6个独立数据集的确认。

| checkpoint | ΔCRPS | 95% bootstrap CI | prior coverage | native coverage | IS4096 full coverage |
| --- | --- | --- | --- | --- | --- |
| positive_s1 | +0.010783 | [+0.005478, +0.015888] | 0.8906 | 0.7214 | 0.8893 |
| positive_s2 | +0.004073 | [-0.000488, +0.008573] | 0.8971 | 0.7760 | 0.8724 |
| positive_s3 | +0.001870 | [-0.000517, +0.004270] | 0.9336 | 0.9193 | 0.9258 |
| unconstrained_s1 | -0.008695 | [-0.014444, -0.002823] | 0.8581 | 0.9596 | 0.8581 |
| unconstrained_s2 | -0.006720 | [-0.010073, -0.003273] | 0.8971 | 0.9701 | 0.8815 |
| unconstrained_s3 | -0.006037 | [-0.013109, +0.001177] | 0.8867 | 0.9805 | 0.8971 |

三个unconstrained检查点CRPS点估计全部改善，其中两个区间完全低于0；三个positive版本点估计退化，其中只有seed1区间完全高于0。positive与unconstrained同时还改变stddev rescaling，不能把差异完全归因于正权约束。no-update自身覆盖率也并非恰为95%。

## 只探测一个机制后的其他机制

冷启动时，已证明物理probe只依赖quadrant1增益；其余三个增益在源prior中独立。各查询均处于原quadrant、没有速度clipping，reset时保持同一系统参数。

| checkpoint | untouched ΔCRPS | 95% bootstrap CI | probed ΔCRPS |
| --- | --- | --- | --- |
| positive_s1 | +0.000098 | [-0.005476, +0.005743] | -0.399472 |
| positive_s2 | -0.007746 | [-0.018638, +0.003318] | -0.407429 |
| positive_s3 | +0.004043 | [+0.000476, +0.007588] | -0.392164 |
| unconstrained_s1 | -0.008124 | [-0.014614, -0.001534] | -0.403061 |
| unconstrained_s2 | -0.003423 | [-0.007792, +0.000892] | -0.398437 |
| unconstrained_s3 | -0.006930 | [-0.013416, -0.000370] | -0.400471 |

仅positive seed3的未探测机制退化区间完全高于0；并非所有模型均退化。所有六个模型在被探测机制上有明显CRPS收益。这不是“模型完全无法利用新信息”。

## 旧方法对照与计算

固定运行SLR完整协方差，以及1024/4096 defensive importance的加权粒子、对角Gaussian与完整协方差Gaussian矩投影；没有按测试指标挑预算。完整结果见[306行聚合CSV](outputs/analysis/aggregate_results.csv)、[所有原始分数](outputs/analysis/raw_scores.csv)、[配对差异](outputs/analysis/paired_contrasts.csv)、[ESS](outputs/analysis/importance_ess.csv)。三次MC不是三次新训练。

4096粒子在空历史无信息probe的ESS最小仍超过2500，信息probe下各checkpoint的ESS中位数约456–1463；并非整体粒子退化。长历史有少数低ESS，unconstrained seed3的n32敏感probe最低约1.59，不能据此断言高维Bayes失败。SLR/importance通常保持无信息probe预测接近不更新，并在敏感probe利用信息；原生摊销更新反而可能补偿prior误差，不能宣称“精确推断必定更低真实CRPS”。

主比较耗时328.28s；峰值GPU已分配内存475.6MiB。此耗时包括该批预测、保存和打分，不是训练耗时或单次在线延迟。服务器D盘本轮完成后约48GB可用。全部原始大数组保留在服务器；本地保存紧凑镜像。

## 独立验证

NumPy独立物理核4480比较，最大误差1.78e-15。918预测行的15606数组/聚合复算最大误差7.60e-14；54编码器检查、144原生预测重放、72proposal/权重核查、36SLR独立NumPy求解通过。不同batch的float32网络重放最大1.79e-6；同batch严格权重误差6.11e-16，ESS2.19e-11，矩2.36e-14，SLR3.56e-15。

初版推断复核将batch改成8192，曾触发过严的零误差和重要性权重门限；保留失败日志。第二版仅将数值代数重放恢复原16384 batch；不同编码路径仍独立检查，允许3e-6语义等价误差。权重/矩/SLR代数门限反而收紧，未修改模型、数据、预测或分数。

raw results SHA256：`4bd5c4743e6bdb1670b09046b231dc3a5686439e395c72d253e80ce8013c1d1f`。统一分析JSON含两份通过的独立验证receipt及其哈希。

![固定比较](outputs/analysis/pendulum_information_update.png)
