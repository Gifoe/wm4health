# Time-series world-model discovery — Cycle 2

这一轮是独立的新受控审计，不修改上一轮结果。研究总目标尚未完成，当前没有已验证的新颖方向。

先读 `final_research_direction.md`、`experimental_results.md` 和 `negative_evidence.md`。本轮26个训练/评测cell全部结束，已有decoder-free RPM对照修复了当前诊断；不据此提出新架构。

配置位于 `outputs/*plan.json`，机器状态在 `outputs/task_status.json`。原生模型源版本固定为 `a75225d5edceb972bb872dc9a335608733567dd4`，依赖父目录 `upstream/rp-gssm`。所有数值实验在服务器运行，模型/NPZ数组留在本目录的outputs子目录。

执行顺序：

1. `scripts/audit_direction_information.py`：调用DoTime真实metric/batching/VAR的新受控任务。
2. `scripts/probe_rpm_tail_bound.py`、`scripts/probe_native_rp_precision.py`、`scripts/probe_native_compatible_rpm.py`、`scripts/verify_marginal_divergences.py`：固定模型机制检查。
3. `scripts/train_native_alarm.py`：默认1000步计划；加 `--plan native_learning_extended_plan.json` 执行预先冻结的10000步计划。
4. `scripts/evaluate_native_alarm.py`：同理指定计划，进行训练reference下的冻结ratio推断与负对照。
5. `scripts/generative_alarm_baseline.py train` / `evaluate`：普通已有非线性观测SSM。
6. `scripts/rpm_exact_baseline.py train` / `evaluate`：已有RPM数值likelihood对照，分别保留正site与允许full Gaussian recognition。
7. `scripts/run_cycle2_checks.py`、`scripts/analyze_alarm_comparisons.py`、`scripts/plot_cycle2.py`、`scripts/plot_rpm_attribution.py`：数值验证、配对区间和可复现图。
8. `scripts/write_cycle2_reports.py`、`scripts/checkpoint_cycle2.py`：由实际结果生成研究记录及hash清单。

原生JAX组件使用本目录 `.venv_rpm`（CPU）；PyTorch训练/积分/绘图使用父目录既有只读 `.venv`。RPM/生成式评测脚本会复用含JAX的公共评测组件，因此评测使用 `.venv_rpm`。不要求重新下载数据或训练checkpoint。

训练脚本对已有完成结果会跳过，评测同样跳过已完成输出；不要在没有新实验问题时重复运行训练。完整重跑应建立新的独立实验目录，并复制冻结计划与固定源码依赖。

统计限定：三颗训练初始化共享一套数据生成种子；额外RPM独立告警控制仅有seed41。oracle拥有真实观测方程。不同base measure的训练loss数值不能跨模型直接排名。经验原子reference外的连续新观测，只报告plug-in ratio posterior，不声称精确连续生成likelihood。
