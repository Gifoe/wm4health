# 真实系统迁移：固定模型、固定协议

本轮只有两个独立物理系统、三个非重叠记录角色；71个窗口不是71个独立装置。这里是 recorded-input rolling-origin simulation，允许每个origin之前128步真实观测，再无输出反馈模拟128步。它不同于官方50步初始化后整段open-loop评测。未来u是历史实际执行输入，不提供未执行计划真值；不据此宣称实际在线控制收益或因果忠实性。

checkpoint只按独立合成validation选取，无真实记录上的神经训练或测试调参。单位变换只来自官方train段；外部ICL/PCFM预训练预算不同，PCFM采用原生中位数、其他神经模型采用均值点预测，不虚称完全匹配训练预算。Silverbox的重叠no-extrapolation子集未重复汇总。

| 物理系统 | 模型 | H128 train-std NRMSE | seed std | 训练seed数 |
|---|---|---:|---:|---:|
| silverbox | conditional96 | 0.809321 | 0.032586133967111595 | 3 |
| silverbox | compiled64 | 0.782454 | 0.03252809335638977 | 3 |
| silverbox | conditional128 | 0.808300 | 0.032766062368594764 | 3 |
| silverbox | native_attention64 | 0.942804 | 0.020000085944272248 | 3 |
| silverbox | native_attention64_chunk32 | 0.905391 | 0.0264552298233568 | 3 |
| silverbox | arx_context_selected | 0.121519 | 无 | 1 |
| silverbox | persistence | 1.272273 | 无 | 1 |
| silverbox | train_mean | 0.898431 | 无 | 1 |
| silverbox | pretrained_native_icl | 0.866632 | 无 | 1 |
| silverbox | pretrained_pcfm_cd_glue0 | 0.850399 | 无 | 1 |
| cascaded_tanks | conditional96 | 1.096481 | 0.035964771125237834 | 3 |
| cascaded_tanks | compiled64 | 1.055239 | 0.08235453464426527 | 3 |
| cascaded_tanks | conditional128 | 1.143663 | 0.14638830027515526 | 3 |
| cascaded_tanks | native_attention64 | 1.081088 | 0.16250643514806687 | 3 |
| cascaded_tanks | native_attention64_chunk32 | 1.106687 | 0.08017004022343098 | 3 |
| cascaded_tanks | arx_context_selected | 6.657338 | 无 | 1 |
| cascaded_tanks | persistence | 1.234223 | 无 | 1 |
| cascaded_tanks | train_mean | 0.968691 | 无 | 1 |
| cascaded_tanks | pretrained_native_icl | 1.078687 | 无 | 1 |
| cascaded_tanks | pretrained_pcfm_cd_glue0 | 1.180374 | 无 | 1 |

所有4260行分数、原始预测、1136个ARX拟合与抽样神经重放已独立复核。实际校验数量以verification.json为准：4260行，ARX 1136次，神经 51组。表中seed std只表示3个训练随机种子的差异，不是物理装置总体置信区间。

这是一组模型组织方式的迁移证据，不自动确认新Research Gap。时标、系统先验覆盖、普通非线性识别与训练预算仍需根据具体失败排查。当前接受最终方向0，总体目标ACTIVE、UNACHIEVED。不得将该试验单独包装为一个通用新模型的充分验证。

源协议 REAL_TRANSFER_PLAN.md；官方分割及原始文件SHA见 outputs/real_splits/manifest.json；全部窗口分数见 outputs/real_transfer/per_window_metrics.csv。
