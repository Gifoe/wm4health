UTC 2026-10-16T19:21:51.067123+00:00。C29完整证据整合；原始12/12训练和全部主/真实/次级/实现复核通过。等更新对照已复核=True。接受方向0，总体目标ACTIVE_UNACHIEVED。

# 完整实验与指标

研究单位是可跨系统复用的时间序列 world model。WorldTS是问题层次参照；单数据集涨点、常规工程修复和已有模块改名均不足以立项。允许有限数据/计算下有价值的组合创新，不要求经典方法在无限资源下仍无解。

## 独立世界的多计划模拟

| 家族 | 模型 | factual RMSE | changed-plan effect RMSE |
| --- | --- | --- | --- |
| lti | conditional96 | 0.518685 | 1.184998 |
| lti | conditional128 | 0.478670 | 1.161889 |
| lti | compiled64 | 0.432509 | 0.927933 |
| lti | native_attention64 | 0.934855 | 2.213042 |
| lti | native_attention64_chunk32 | 0.773906 | 1.978284 |
| lti | arx_context_selected | 0.006396 | 0.014222 |
| lti | pretrained_native_icl | 0.837783 | 2.379410 |
| wh | conditional96 | 0.742050 | 1.552423 |
| wh | conditional128 | 0.734700 | 1.524709 |
| wh | compiled64 | 0.708764 | 1.557241 |
| wh | native_attention64 | 0.993575 | 1.802091 |
| wh | native_attention64_chunk32 | 0.878406 | 1.688486 |
| wh | arx_context_selected | 0.676759 | 1.953751 |
| wh | pretrained_native_icl | 0.985830 | 1.936135 |

每系统RMSE再平均；effect排除factual零差分。外部预训练权重预算不同，不能做公平训练效率结论。原生attention迁移的低分不等于其原论文能力上限。

| 家族 | compiled−wide | ratio | 描述性跨种子/系统区间 | 3个种子差值 |
| --- | --- | --- | --- | --- |
| lti | -0.233956 | 0.7986 | [-0.422689, -0.072438] | -0.0628, -0.1973, -0.4418 |
| wh | +0.032533 | 1.0213 | [-0.020724, +0.078972] | +0.0471, +0.0558, -0.0053 |

2000次跨世界/训练种子bootstrap只作描述；3种子不足以证明广泛稳定性。LTI优势不能扩展到WH，且ARX在LTI更强。

## 等更新次数归因控制

等更新次数对照全部6次训练、18432条指标、验证最小checkpoint与神经重放复核通过。每组每种子32768次更新；不声称FLOPs/耗时完全相同。仍使用原测试世界，属于归因控制，不是新独立复现。

| 家族 | 模型 | factual RMSE | effect RMSE |
| --- | --- | --- | --- |
| lti | compiled64 | 0.387575 | 0.829930 |
| lti | conditional128 | 0.408864 | 0.988970 |
| wh | compiled64 | 0.676747 | 1.516626 |
| wh | conditional128 | 0.694655 | 1.480681 |

| 家族 | compiled−wide effect | ratio | 描述性跨世界/种子区间 | 3种子差值 |
| --- | --- | --- | --- | --- |
| lti | -0.159040 | 0.8392 | [-0.241784, -0.069341] | -0.0628, -0.1795, -0.2348 |
| wh | +0.035945 | 1.0243 | [-0.006982, +0.077869] | +0.0471, +0.0135, +0.0472 |

等更新后LTI效应误差仍低约16.1%，WH反而高约2.4%，且WH三个种子差值均正。故原LTI优势并非只由更新次数不等解释，但没有跨家族通用优势。完整36对比、24汇总、各seed与局部cap标记见[等更新归因报告](FIXED_UPDATE_CONTROL_REPORT.md)。普通早停模型续训仍显著改善；cap局部下降标记false不能当作收敛认证。

当前固定cap下仍触发原局部下降规则的run为compiled64/2026292。因此此处只讨论32768更新内的有限预算实测，不将其解释为架构收敛后的能力上限。

## 独立实物记录迁移

| 物理对象 | 模型 | H128 NRMSE | 训练种子标准差 |
| --- | --- | --- | --- |
| silverbox | conditional96 | 0.809321 | 0.032586 |
| silverbox | compiled64 | 0.782454 | 0.032528 |
| silverbox | conditional128 | 0.808300 | 0.032766 |
| silverbox | native_attention64 | 0.942804 | 0.020000 |
| silverbox | native_attention64_chunk32 | 0.905391 | 0.026455 |
| silverbox | arx_context_selected | 0.121519 | - |
| silverbox | persistence | 1.272273 | - |
| silverbox | train_mean | 0.898431 | - |
| silverbox | pretrained_native_icl | 0.866632 | - |
| silverbox | pretrained_pcfm_cd_glue0 | 0.850399 | - |
| cascaded_tanks | conditional96 | 1.096481 | 0.035965 |
| cascaded_tanks | compiled64 | 1.055239 | 0.082355 |
| cascaded_tanks | conditional128 | 1.143663 | 0.146388 |
| cascaded_tanks | native_attention64 | 1.081088 | 0.162506 |
| cascaded_tanks | native_attention64_chunk32 | 1.106687 | 0.080170 |
| cascaded_tanks | arx_context_selected | 6.657338 | - |
| cascaded_tanks | persistence | 1.234223 | - |
| cascaded_tanks | train_mean | 0.968691 | - |
| cascaded_tanks | pretrained_native_icl | 1.078687 | - |
| cascaded_tanks | pretrained_pcfm_cd_glue0 | 1.180374 | - |

Tanks的train-mean优于全部神经模型；Silverbox的上下文ARX明显更强。两台对象不足以证明跨领域模型优势。

## 有限计划选择诊断

| 家族 | 决策器 | 平均 regret | 4计划内 oracle 选择率 |
| --- | --- | --- | --- |
| lti | compiled64 | 0.2327147 | 0.9141 |
| lti | conditional128 | 0.7954726 | 0.8646 |
| lti | native_attention64_chunk32 | 4.4681094 | 0.5807 |
| lti | arx_context_selected | 0.0000008 | 0.9974 |
| lti | pretrained_native_icl | 6.0255111 | 0.5417 |
| lti | always_zero | 0.0610344 | 0.8880 |
| lti | always_factual | 1.0475014 | 0.0312 |
| wh | compiled64 | 2.3565282 | 0.5625 |
| wh | conditional128 | 2.6195426 | 0.5408 |
| wh | native_attention64_chunk32 | 3.5614242 | 0.4705 |
| wh | arx_context_selected | 3.4562574 | 0.5547 |
| wh | pretrained_native_icl | 2.7803204 | 0.4245 |
| wh | always_zero | 1.4636206 | 0.5443 |
| wh | always_factual | 0.9381071 | 0.2031 |

候选固定为factual/zero/step/pulse，目标-1/0/+1，cost=mean((y-target)^2)+.01mean(u²)。只用模型点预测选择，再由clean物理分支评分；oracle仅四计划内最优。compiled可比wide更好却仍输固定策略，说明不能把轨迹平均分改善自动解释为实际控制收益。

这些是点决策器，不是各概率模型的原生期望成本评估。平方损失必须计入边际方差，未做此控制前不得宣布“原生不确定性/规划失败”或将其立为新Gap；也未做闭环MPC或真实未执行动作。

## 普通等价实现控制

| 模型 | 实现 | cached ms | 一次含准备 ms | 64次摊销 ms |
| --- | --- | --- | --- | --- |
| conditional96 | source_eager | 20.3804 | 20.7890 | 20.3893 |
| conditional96 | code_bias_fold_fused_gru | 0.3665 | 2.0093 | 0.3938 |
| compiled64 | source_eager | 42.2780 | 42.8916 | 42.2903 |
| compiled64 | merged_weight_fused_gru | 0.3478 | 1.9746 | 0.3741 |
| conditional128 | source_eager | 20.7680 | 21.2519 | 20.7761 |
| conditional128 | code_bias_fold_fused_gru | 0.3788 | 2.1916 | 0.4071 |
| native_attention64 | source_eager | 2.7185 | 4.2481 | 2.7419 |
| native_attention64 | cached_memory_sdpa_no_weight_output | 2.4190 | 3.7183 | 2.4405 |

H128/16计划，12个验证case的中位数；含context/preparation、cached以及64次摊销。576时间行、288全序列等价检查通过，maxabs=1.8894672e-05。折叠code bias/合并Δ权重后普通GRU可用cuDNN；attention缓存memory并允许SDPA。原eager耗时不能作为新编译机制动机。这些是具体实现耗时，不是所有架构的下界，未穷尽torch.compile/KV等实现。

主验证79872指标、4096独立ARX拟合；真实验证4260指标、1136ARX重拟合；次级验证120汇总/72对比/64512决策/504汇总/6utility对比全部重算通过，最大float64相对误差3.134e-15。证明路径列于outputs/integrated_evidence_v1/summary.json。数学一致性不认证科学新颖性。
