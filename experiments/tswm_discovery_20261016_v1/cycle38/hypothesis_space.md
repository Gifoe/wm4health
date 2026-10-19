# C38 竞争解释与反证

UTC 2026-10-16T23:03:29.601710+00:00。C38为三个独立clean-context视角的定向文献补查及root代码/原文复核；新训练0、新方法0、最终接受方向0，整体ACTIVE_UNACHIEVED。C37原作业另行处理，未使用其partial指标构思本轮故事。

| 尚未排除解释 | 决定性普通对照 | 何时关闭新机制归因 |
|---|---|---|
| 文本信息先被接口丢掉 | 同冻结语言模型contextual表示，同时给WorldTS与direct | 两者差距由前端修复解释 |
| 只是监督或训练分配 | 同信息direct+numerical auxiliary、对称有限1×/2×预算 | 普通方法追平，不能靠latent图挽救 |
| 只是readout不兼容 | 同接口兼容decoder及普通观测loss | 效用错位被普通适配消除 |
| 内容没有独立预测信息 | cross-fit residual probe；dates/entity/length/missingness/time-shift placebo | 正确文本不胜placebo，停止该任务作为证据 |
| 条件传播确有额外交互 | 普通conditional GRU/selective SSM；去交互保参数消融 | 同信息ordinary recurrence解释，关闭所提机制 |

本表是可失败实验逻辑，尚非新方法与训练合同；没有凭它启动fits。A效用错位优先，B稀少语义信息次之，两项都未立项。
