# 原生模型实际训练：失败归因与恢复快照

当前研究目标ACTIVE/UNACHIEVED，接受新颖方向0。原三个主baseline的已有证据继续保留；DALI是额外候选的source-based transfer，不是第四篇完整论文复现。

## 本轮实际执行与边界

原生完整WorldModel核心的三次丢弃资格更新通过，未将17M变量核心替换成小网络。随后启动固定2000-update factual训练，使用512条TRAIN轨迹、64条既有开发VAL、四个固定随机draw，验证点0/500/1000/2000。按原生目标训练；reward0/cont1是明确的辅助迁移标签。没有干预或独立EVAL。

第一个实际运行完成500次更新，随后在保存Optax嵌套状态时被我们的NPZ序列化错误中断。500-step空文件、完整错误和退出回执保留；没有有效的500-step模型或指标回执，因此不报告一个丢失的预测结果。它不是模型发散，也不是一个研究缺口。

恢复从已校验的原step0 checkpoint及原NumPy状态开始，完整状态改为保留PyTree的主机生成pickle，模型数组另存NPZ。第1步完成实际保存和逐叶精确读回检查，之后每250步额外保存恢复状态，验证点不变。所有重算前缀都单列，若恢复完成则实际optimizer调用500+2000=2500次，不能称为仅计算2000次。原记录的loss标量逐点对照重放，不匹配就停止并报告。

发布快照恢复状态：`RUNNING_WITHOUT_TERMINAL_RECEIPT`，已记录完成更新下界50。下界与发布时刻不等于运行最终完成步数。最新完成和checkpoint状态以[机器快照](FACTUAL_TRANSFER_PROGRESS_SNAPSHOT_V1.json)为准。

## 当前已确认的源代码差异

原生跨模态loss在受控fixture中对recon_wm的kernel/bias给出严格0梯度，而反向投影梯度非零。独立Torch解析核验3072个状态梯度分量，最大误差4.44e-10。这个差异与训练预算不足无关，但尚未说明它导致预测失败，也没有证明论文实验使用此源码版本。

论文描述的跨模态目标与当前代码的目标状态/梯度路由有差别，见[版本边界](PAPER_SOURCE_OBJECTIVE_BOUNDARY_V1.md)、[实际原生梯度](crossmodal_gradient_contract_v1/RESULTS_V1.json)和[独立复核](crossmodal_gradient_contract_v1/VERIFICATION_V1.json)。当前运行保持源码不变；普通目标修正如有需要，必须作为单独等预算控制。

## 不允许据此提出的方法

编辑器的零尺度、错误索引和随时间变号已有普通修复。JAX静态参数遗漏和checkpoint序列化错误属于我们的harness。时钟核验第二次还发生了原因未定的原生process abort，资格保持未通过；训练进程与它独立。它们都不能作为新world-model机制的动机。

[当前文献反证](MECHANISM_NOVELTY_SCREEN_V2.md)还排除了把一般干预保真度、低秩patch或动态/静态信息分离重新包装成创新。候选只剩一个尚未验证的窄问题：在factual能力与物理root合约通过后，同一context编辑是否能被一个跨root/action复用的固定参数变化解释。

## 下一项决定性检查

先完成当前冻结训练并独立复算保存的预测；若两个动力学族不能达到预设普通基线门槛，就不进入机制归因。时钟核验、root-state保留、原生目标差异、普通显式参数/投影/重新初始化控制仍需排除。没有新方法、稳定提升或最终研究方向的接受结论。

[原冻结协议](FACTUAL_FIT_PROTOCOL_V1.md) · [保存恢复补充](FACTUAL_CHECKPOINT_RECOVERY_V1.md) · [实际训练入口](scripts/train_factual_native_v2.py) · [训练前源核心结果](native_core_profile_diagnostic_v1/RESULTS_V1.json)。数据、完整权重和环境留G，未同步本地。
