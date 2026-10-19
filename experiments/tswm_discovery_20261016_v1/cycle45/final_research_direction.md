# C45 Current framework research direction — not yet accepted

UTC 2026-10-17T11:05:22.881781+00:00。本轮PROGRESS；整体ACTIVE_UNACHIEVED，最终接受方向0。固定官方NCDSSM commit 7212bba9616d81c948f507eef9eb1f7d5cf1954e，48个选定source/config/license文件/Gitblob+SHA核查。原生线性推断与独立NumPy/dense Gaussian一致；非线性原生core完成10 discarded updates/862参数与预测回放，0正式比较fits。普通联合Markov路径控制核查了线性采样定义区别，不是新模型收益。

**Research Question**：一个时间序列world model能否用同一训练配方，把观测证据接口与动力学机制学成可复用组件，并在未见sensor–system组合保持可靠的概率未来推演？

**Evidence**：C42只有普通因子化HMM的人工held-out结构信号及seen代价。C44完成16模型的分布评分和强普通控制筛查，未支持新增query层级机制；余下查询提前停止。C45已实际运行强完整SSM core并明确其采样合同，但没有新方法效果。

**Existing Methods**：NCDSSM已有auxiliary recognition/dynamics分离；modular模型已有动态模块复用；LIFTS已有joint inference/generation；积分观测和尺度对齐另有完整近邻。它们的直接强组合必须保留。采样接口差异有普通处理，不能制造新框架优势。

**Proposed Idea**：开发重点是观测证据×动态组合之间的必要学习交互。具体新配方仍缺证据，不把ordinary structured SSM或Markov采样改名。已知组件可以构成新配方，但必须有明确区别和真实独立效用。

**Falsification**：强直接组合覆盖未见组合收益；交互不成立；只改善seen cells；收益由额外信息/容量/预算解释；或需要每个数据集换机制。发生时放弃相应创新主张。

**Feasibility**：单RTX5090，原生NCDSSM数值core已在现有环境执行。下一步先完善普通非线性generative joint-path对照，再冻结完整神经任务、容量/预算和信息合同。资格权重全部排除，不在旧EVAL上调方法。

**Next Experiment**：先使非线性强基线具有明确的联合轨迹与独立outer-history抽样定义；随后检验一个具体新学习交互对比完整direct composition。需要未见组合、seen权衡和独立任务证据；不能再以“11最好”代替机制证明。最终idea和顶会潜力尚未验证，整体目标继续ACTIVE_UNACHIEVED。
