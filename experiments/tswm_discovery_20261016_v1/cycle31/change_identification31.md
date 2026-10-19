# C31 相邻领域核查：冲击、机制变化与有限预算辨识

日期：2026-10-17。问题是哪些统计量应保留、哪些应重置，以及短日志中是否有足够信息作此决定。本笔记核查一手论文与源码；未执行模型、训练或 SSH。2026 三篇 meta-ID 文献与 C29 重复，不充作本轮新证据。

## 1. 已有机制的边界

| 路线 | 真正覆盖 | 不应冒充的新意 |
|---|---|---|
| [Dual/joint UKF，Wan 等，NeurIPS 1999](https://papers.nips.cc/paper_files/paper/1999/file/f50a6c02a3fc5a3a5d4d9391f05f3efc-Paper.pdf) | 同时推断隐状态与模型权重，神经转移亦可 | state/parameter 双时间尺度、分开更新 |
| [APE，Nemeth 等，IEEE TSP 2014](https://arxiv.org/html/1510.02604v1) | §4.1/4.4 保留固定参数，重置变化参数的先验或充分统计量；§4.5 联合估计状态、连续转弯率、过程/观测噪声 | “IMM 参数网格爆炸，所以改连续后验”；“只重置变化子集” |
| [MOCA，NeurIPS 2020](https://papers.nips.cc/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf) | 学先验、未知分段、run-length 后验、短上下文更新联合训练 | 给可复用模型添加 Bayesian change detector |
| [DM-BOCD，ICML 2023](https://proceedings.mlr.press/v202/altamirano23a.html) | 稳健广义后验，避免离群点导致虚假变点 | 把稳健检测说成已识别冲击/参数/噪声三种原因 |

区别在于：MOCA 的 task statistics 重置并不自动保留物理隐状态；反过来，APE 已能在参数切换时继续传播当前状态。一个新神经模型没有显式 cause 标签，不代表普通概率组合不会处理。

**源码核查**：[FilterPy IMM](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/IMM.html) 有模式混合、状态协方差与似然更新，要求共同状态维度；它不自动产生“参数重置”模式，需要自行定义。[MOCA 源码](https://github.com/StanfordASL/moca/blob/master/metacpd/main/moca.py) 每步扩展全部 run lengths，未见 top-K 截断。[DSM-bocd/bocpd.py](https://github.com/maltamiranomontero/DSM-bocd/blob/main/bocpd.py) 的 K 只保留部分 message，但仍分配 T×T posterior；[models.py](https://github.com/maltamiranomontero/DSM-bocd/blob/main/models.py) 仍更新全长统计量。改成滚动存储、只维护 K 个充分统计量是普通修复。APE 原作者算法代码未核实；不能称已复现。2024 CPAKF 所链接的[仓库](https://github.com/xlhou/CPAKF-CPAKS)只有 BenchmarkTrajectory.mat，不能列作算法开源。

## 2. 最强普通修复与可识别性

先做共同状态空间的 Gaussian-sum/粒子 bank：正常延续；仅本步加入状态冲击、保留参数后验；仅重置参数、保留状态；仅重置 R 的统计量、保留状态与参数。后验按多步条件似然更新，保留 K 个分支；已观测输入改变只改变条件输入。连续参数用 UKF/APE，无须先离散化整张参数网格。允许短固定滞后平滑与 validation 选择 K、hazard、协方差。此为自主普通组合，不是任何论文原生实现，更不是新候选方法。

单次新息不能判因。在线性化短窗口，白化后的变化满足 ΔY≈JxΔx+JθΔθ；只有 Jθ 在 Jx 列空间之外仍有信息时，才能区别机制变化与状态误差。用该投影的最小奇异值和 SNR 分层评分；这是本笔记推导的诊断，不是新算法。病态层报告不确定，不强迫分类。噪声尺度变化另需多点证据；输入支撑外失败不能记为动态更新失败。

## 3. 唯一建议的淘汰实验

**身份：自主扩展 APE §4.5 的 coordinated-turn tracking；不是实测复现，也不证明跨系统通用性。** 该问题对应雷达对机动目标的跟踪，不靠隐藏输入或故意禁止普通修复制造难度。

状态为二维位置/速度，机制为连续转弯率 ω；观测仅 range/bearing，另提供雷达位置 s。所有方法获得相同公式、训练 I/O 日志与 s；真实状态、ω、变化时刻/类型仅给评分器。先用同一多轨迹数据估计共享先验，不给新方法额外机制标签。

Δt=1，长度400；随机时刻分别注入速度冲击、持续 ω 跳变、R 四倍变化、已观测 s 的轨迹分布变化，并保留无变化对照。各类按初始白化 surprise 分层匹配；ω 在训练连续支撑内，未知边界；高噪声与低可辨识层单列。观察变化后8/16/32点，再预测未来1/8/32点，未来 s 同时提供。训练/测试按目标与随机种子分离。

同预算比较：dual UKF；四类 IMM/Gaussian-sum（K=16/32）；连续参数 APE（N=128/512）加相同冲击/R分支；MOCA+ALPaCA 历史回归作学习对照。允许所有基线见到全部变化类型的训练日志。拟议上限：同一单5090主机，每法离线拟合与调参合计2小时；在线预算5/20ms每点，测实际p95延迟与峰值内存。不能只卡粒子数而给神经方法额外时间；本轮实际执行量为零模型调用。

主量为条件预测 log score、长预测误差、机制误更新率及恢复延迟。若普通 bank 在可识别层消除误更新且达到最佳误差/延迟前沿，立即撤销方向；若全部方法同败且信息矩阵病态，撤销实验归因。只有新增具体机制在同预算稳定胜过上述修复，才重新讨论机会。

**结论：NONE。** 当前没有测得的普通组合余差；“保留状态、选择性重置机制、学先验后快速更新”均不足以构成研究增量。有限预算假设管理可能有工程余量，但现有证据尚不足以指定一个值得立项的新环节。
