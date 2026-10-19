# C32：预测状态目标与可复用时间序列 world model

2026-10-17。用户刚纠正范围：研究应达到 WorldTS 的通用框架层次，不是单数据集优化。按数据集训练允许；不强制共享 checkpoint 零样本、动作、严格因果、物理解释、Loop、LLM 或在线变化识别。目标是从真实代码、最新文献与实验选一个有价值的具体方向；当前接受最终方向0。

本轮 owner 使用 ccf-idea-optimizer exploratory；deep-research 按三个 clean-context perspectives 定向核查。不是让每个视角仅找理由否定。若朴素动机已知，须给一个可证伪的精确改写或最小救援路线，再决定继续/撤销。不能用“component known”自动否定有限预算有效组合；也不能更名已有方法。

RQ1：WorldTS/LatentTSF/JEPA 的未来观测编码目标，究竟代表哪一种预测状态？在作者真正声称的条件预测任务中，有无具体、重要、可复现的优化或信息不匹配，而非强加因果/无穷长模拟要求？

RQ2：较新的 predictive-state、history-aware target、multiscale/self-predictive representation、stochastic latent objectives 等方法，哪些已直接解决该问题？在相同输入/总预算下，最强普通修复是什么？剩余增量须具体到机制/兼容条件，不能只说“预测充分”。

RQ3：哪一项单5090可运行实验能区分“目标本身不足、目标权重/优化、有限容量、条件输入不合协议、解码器问题”？优先现有 native task/code/data；若有合成诊断，须连到实际模型/真实任务，不把人为二维瓶颈当普遍失败。

## 已有证据和禁止重复的宽泛主张

- C11：学得目标分布的 latent Energy 有精确 folding 风险，但12个实际神经 fits 均保留状态，没有展示原生模型 alias failure。信息/logloss/decoded proper scores、EMA等均为既有控制。不要重跑同一个Gaussian fold toy。
- C20/C21：未来不可预测的 observed driver 在有限 bottleneck 下可能丢失；72-fit pilot+120-fit independent-cohort完成。Projected absolute future-bank主候选不稳定；response-only三族9/9成功，queueing上普通obs predictor更强；成熟 sensitivity compressor+适当目标可修复。WorldTS本来就区分 named target/covariates；CDL/Denoised MDPs、derivative/active subspaces/平衡约简已有。自动role-splitting不等于创新。不要把这条直接改名。
- C29：低秩条件编译已有先例；等更新LTI好16.1%但WH差2.4%，跨家族收益不统一。不要只增加新的forecast head/Loop。
- C30：naive causal FM速度投影在独立输入也破坏joint，普通causal likelihood和MAF/IAF蒸馏已有。不要再提朴素causal mask或新名teacher distillation。
- C31：NormWear-2公开1.87亿参数权重实际执行、三个条件接口错误及普通修复/独立验证完成。仅baseline qualification，不是新研究gap。Streaming状态/参数/变点机制已有HiP-RSSM/MTS3/APE/MOCA等强先例，没有测得新残差。

需要查看历史时，只读本brief相关科学MD/脚本。**禁止读取/执行 cycle6 秘密/W&B launcher。** 不碰外部无关项目，不公开发布、不发送消息、不启动新训练。每个视角只写自己指定MD，使用已核查的一手链接，标明本轮检索/源码审计/实际执行的区别。

检索重点2026-04-17至2026-10-17，必要时回溯经典。最终报告要支持最小实证决策，不追求文献数量。不可核实参考不用。可给 NONE，但必须说明最具体可执行的下一判断，不能只列prior-art catalog。
