# C58 相邻领域检索：记忆动力学与可复用模型

日期：2026-10-18。角色：独立 adjacent-fields 检索。边界：仅检索、阅读代码和写本笔记；未访问 SSH、训练、安装依赖或修改旧实验。

## 摘要

本轮没有确认一个已成立的新方法方向。显式长期记忆、随机闭包和模块化动力学都有直接且强的成熟近邻。只保留一个**有应用意义、代码可落地，但尚未通过新颖性与残余失败门槛的条件性问题**：当永久性 hysteretic memory 与会随时间衰减的 relaxation 同时存在时，能否从有限上下文学习可跨设备/系统复用的条件模拟器，并在未见过的时序安排和 reversal 组合上保持正确响应？不能把加入 Preisach/play、exponential filters、context encoder 或 particle initializer 本身当作贡献。

一个额外的代码发现比“新 idea”更确定：HystRNN 官方 Exp1 脚本存在输入轴与隐状态使用问题；直接复用它作为记忆基线会产生不公平的弱对照。该实现问题不构成研究缺口。

## 1. 检索问题与证据方法

冻结 brief 的 RQ1–RQ3 保持不变，本视角用一个具体问题落实它们：

**Q：在输入/观测信息、历史长度与容量公平时，共存的非衰减 hysteresis 和衰减 relaxation 是否会给“跨系统复用的条件模拟”留下普通方法修复后仍存在的失败？**

先检索 system identification、Mori–Zwanzig/GLE、fractional/delay memory、sampling/semigroup、port-Hamiltonian composition 和 hysteresis；再围绕命中的方法检索最新论文与原始代码。事实只取论文原文、作者/机构页面与官方仓库；没有使用第三方 AI review。以下“来源事实”与“研究推断”分开记录。论文存在及所述机制已经核对；数学定理未独立证明，代码未运行。

## 2. 已有方案覆盖到哪里

| 分支 | 核验的一手证据及适用条件 | 对候选创新的约束 |
|---|---|---|
| 紧凑随机记忆闭包 | Yang & He 的 [Memory-Conditioned Diffusion Model for Generalized Langevin Dynamics](https://arxiv.org/html/2609.28371v1)，2026-09-23：递推 exponential-filter bank、条件随机 flow map，测试包括多时间尺度及非交换的向量 memory。原文 §5 明说要求 stationary/ergodic/fading memory，也承认 Gaussian-head LSTM 的比较不能排除更强输出族。 | “显式多尺度记忆 + 生成器 + 递推 rollout”已经存在；其适用边界不能被描述成所有 recurrent models 的失败。 |
| 可学习时间延迟 | Monsel et al. 的 [Neural delay differential equations: learning non-Markovian closures for partially known dynamical systems](https://arxiv.org/abs/2410.02843)，2024 初稿、2026 修订/发表：学 constant delays，从 partial observations 学 non-Markovian dynamics，并含实验测量。 | “用延迟代替额外 latent”不新；不能拿 instantaneous NODE 当唯一强对照。 |
| 不衰减的物理记忆 | Roussel et al. 的 [Differentiable Preisach Modeling…](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.204801)，PRL 2022：可微 Preisach 与 Bayesian response model，含 accelerator magnet 的实际测量。 | 神经网络、可微 hysterons 和 Bayesian observation model 的结合不是空白。 |
| 通用非线性状态空间辨识 | Noël et al. 的 [A nonlinear state-space approach to hysteresis identification](https://arxiv.org/abs/1610.09138)，MSSP 2017：polynomial nonlinear state-space、weighted least squares，在 Bouc–Wen 合成响应上做 broadband/sine validation，不预设 Bouc–Wen 形式。 | 不能声称有限维 state-space 天然不能处理 hysteresis；一般 multistable state-space 是必须排除的强近邻。 |
| 神经 hysteresis 与长期损伤 | Chandra et al. 的 [HystRNN](https://research.tue.nl/nl/publications/generalizable-models-of-magnetic-hysteresis-via-physics-aware-rec/)，CPC 2025：论文报告未训练 reversal curves/minor loops；Jeon et al. 的 [Unified Hysteresis Modeling via Physics-Based Deep Learning and Data Augmentation](https://onlinelibrary.wiley.com/doi/10.1002/eqe.70081)，2025-11 online / 2026 卷期：改造 LSTM、累积损伤、耗散约束和 resampling augmentation，含真实 yielding-link 数据。 | “非衰减记忆”“累积损伤”“时间重采样不变性”均不是新概念。下文另给 HystRNN 的代码保留意见。 |
| 最近的混合条件预测器 | [PI-HNO](https://arxiv.org/html/2608.02965v1)，2026-08：以 B–H history、未来 B(t)、temperature/context 预测未来 H(t)，融合 recurrent boundary state 与 Preisach-inspired global branch；使用每种材料各自的模型。 | 边界历史 + 未来条件 + 温度 + 局部/全局记忆已经覆盖。跨材料复用不在其已报告设置中，但“没有报告”不是新方法成立的证据。 |
| 泛序列 hysteretic memory | [Preisach Attention](https://arxiv.org/html/2605.23603v1)，2026-05：提出 learned relay thresholds 与 extrema stack；原文把 empirical validation 列为 open question。 | 把 extremum stack 塞进 sequence model 已有明确提案。本笔记不采信其未独立审计的普适充分性/表达能力定理，也不将其当实证强基线。 |
| 动力学模块组合 | Neary & Topcu 的 [Compositional Learning of Dynamical System Models Using Port-Hamiltonian Neural Networks](https://arxiv.org/abs/2212.00893)，L4DC 2023：独立训练子系统后组合，已知或学习 interconnection；[van Otterdijk et al.](https://arxiv.org/abs/2411.05730)，2024：从 input-output 测量辨识嵌在大系统中的子系统。 | “独立学模块再组合”“不用内部状态的子系统辨识”本身不是新 world-model 框架。 |

**综合判断（推断）：** broad memory、broad observation/dynamics separation、broad compositionality 均应降级。Kalman/ARX、proper likelihood、Koopman/bilinear、HiP-RSSM/MTS3、CoDA、conditional neural processes 等仍是 brief 指定的强控制；本轮不以这些方法暂时没出现在某论文表格里作为缺口。C57 的 normalizer 检验不为本问题提供正面证据。

## 3. 唯一条件性 lead：两种记忆的复用与时序变换

### 3.1 为什么值得提出这个问题

来源事实：PRL accelerator 实验说明 hysteresis 会影响重复性及优化；PI-HNO 的磁性器件场景同时涉及 minor loops、dc bias、temperature 和 rate-dependent response；结构 hysteresis 还涉及积累损伤。相比“把 toy 变长”，这些是具体的可重复运行需求。

研究推断：相同输入幅值路径可以有不同的时间安排。一个纯 rate-independent memory 只应随输入路径更新；relaxation 则应在 dwell 期间变化。模型若把两者混为一个按样本步数衰减的状态，可能把停留时间或采样密度错误地当成擦除历史的事件。这里的目标是检验**实际条件响应能否跨环境复用**，不是要求恢复物理真实 latent，也不要求因果解释或大规模预训练。

### 3.2 最小机制，只是候选实现骨架

给 context encoder 相同的输入输出前缀，得到环境条件 c；状态含 r 和 z：

\[
r_{t+1}=R_{\theta(c)}(r_t,u_t,u_{t+1}),\qquad
z_{t+1}=\exp[-\lambda(c)\Delta t]z_t+B(c)\Delta u_t,
\]

\[
p(y_{t+1}\mid y_{\le t},u_{\le t+1},\Delta t)
=p_\theta(y_{t+1}\mid r_{t+1},z_{t+1},u_{t+1},c).
\]

R 可以先直接用标准 play/relay 更新；z 用普通 exponential state；输出给所有模型同样的 Gaussian mixture/flow 或确定性误差预算。此骨架**不是新方法**。它首先是杀死过强新颖性叙事的基线。未知初始化时的 particle/importance update 同样只是控制，不能重新命名成创新。初轮应统一给可知的 saturation/reset prehistory，先排除初始化不可辨识性。

### 3.3 一个完整框架究竟还需要增加什么

即使 primitives 已知，一个框架仍可能有用，但必须做成以下完整可复用能力：同一训练流程从上下文决定何种记忆需要长期保留、何种状态随物理时间松弛；在未见系统上初始化并持续更新这些状态；接收未来外部条件进行多步模拟；同时覆盖纯 fading、纯 hysteretic 和混合情况，而不为每个数据集手选方程或换模型。理论/实证须说明它比**同样 context-conditioned 的标准 hysteresis+relaxation hybrid**多解决了什么。

仅统一 API、增加 branch、接 CNP、改 loss、加入不确定性 head 或得到比弱 GRU 更好的曲线，都不足以满足这个门槛。当前尚未发现一种已经有证据能通过门槛的具体新机制。因此 lead 的状态是 **question retained / method unaccepted**。

## 4. 最小反证设计与一张 5090 的可执行性

### 4.1 先固定任务语义

- 合成 forward 任务可规定 u 为驱动，y 为响应；所有模型都得到同一过去 u,y、未来 u、真实 Δt、reset 信息。不要把真实参数只给新模型。
- MagNet/PI-HNO 常见方向是已给 B(t) 预测 H(t)，与某些 H→B 的 forward toy 相反。若用该数据，应称为 conditional response simulation，单独统一方向；不能靠悄悄换输入语义制造胜负。
- dwell invariance 仅属于 rate-independent 部分。混合系统整体在 dwell 中可以改变，不能强迫全部输出不动。

### 4.2 先用能被普通方法正确解决的诊断，而不是设计普通基线必败的 toy

预先固定三组：标准 Preisach/play；普通线性 relaxation；二者级联/并联的混合。约 64 个参数环境、每环境 16 条长度 512 的短序列、32–64 个 relay/play units、8–16 个 relaxation modes 足够做表示与实现核查。模型控制在约 10^6 参数以内。此规模是内存可行性的工程估计；没有计时，不能承诺运行时长。

分割在生成前封存：heldout 环境参数 seed、heldout reversal-path seed、以及相同路径下 1×/2×/4× dwell/clock 安排。保持输入幅值与参数范围有覆盖，不在发现失败后再增加嵌套深度、传感器非线性或噪声来保住方向。所有标度与 memory budget 只由 train/validation 决定。

**必须包含的强控制：**

1. 标准 identified Preisach/play + linear/exponential relaxation，包含正确或充足阶数；其参数可通过有限 context 拟合。
2. 同一 context encoder 驱动上述 hybrid 的参数或非参数 density，输出族、历史和预算与候选一致。
3. 长上下文 GRU/LSTM/SSM + 同等 expressive head；加上 input-increment 与真实 Δt，并训练 resampling augmentation。
4. 正确 hidden-state persistence 的 HystRNN/physics-based LSTM；NDDE/EMA-memory stochastic flow map 作为有需要时的记忆控制，而非默认瞬时 NODE。
5. PNLSS 以及充分容量的 nonlinear neural state-space/output-error 模型：state order 和 polynomial degree 用 validation 选；按物理时间或 input increment 更新，含相同初始化与 context adaptation。不能只用稳定线性 Kalman 模型代表“state-space 已排除”。同样允许其从 context 做参数拟合或采用共享参数/context modulation，需报告适配计算。

**决定性否定控制：** 如果 context-conditioned standard hybrid 在 heldout 环境与时序安排上达到 oracle/有限样本误差附近，候选只胜过通用 RNN，那么停止“新 world-model 方法”叙事。这说明普通建模修复已经够用。若候选收益仅来自为控制补上 Δt、Δu、state carry 或更强输出 head 后消失，同样终止。这个结论不应通过换一个更难 toy 来规避。

**可一次结案的预注册实验：** 封存一个 2×2 评估，轴分别为 seen/heldout 环境与 seen/heldout reversal+timing，并在上述三组模型族及一套真实 complete-run 留出上同时评估。先保证标准 hybrid 在其自身生成族达到 oracle floor，否则本次运行判为实现/优化无效。然后比较候选、context-conditioned hybrid 和充分阶 nonlinear state-space，使用独立运行级 bootstrap。若任何一个成熟控制在所有预定能力格都非劣，或候选的改善只出现在无物理意义的超长 dwell，而正常 heldout complete runs 无可重复收益，则本 lead 记为 **no residual method opportunity**。不能在结果出来后只保留候选赢的一格。具体非劣容差应在看 test 前由 measurement repeatability/有限样本 oracle floor 封存，而非事后挑百分比。

若诊断无实现错误，再做真实数据的全材料/全实验 run 留出；禁止随机切相邻窗口当跨系统泛化。多步响应误差、loop area/耗散误差、长 dwell 后的响应，以及纯 fading 对照的退化风险分开报告；噪声任务再报告 proper log score/CRPS 与联合轨迹统计，不能只看 MSE。

### 4.3 真正存在的代码与数据入口

| 入口 | 已核验到的可用内容 | 限制 |
|---|---|---|
| [roussel-ryan/diff_hysteresis](https://github.com/roussel-ryan/diff_hysteresis) | PyTorch Preisach；读取了 `hysteresis/base.py`、`states.py`；仓库有 fitting examples 与测试。 | 未安装或执行。作为成熟 baseline，不是新框架。 |
| [pawsen/pyvib](https://github.com/pawsen/pyvib) / [PNLSS guide](https://pyvib.readthedocs.io/en/latest/user-guide.html) | 官方 repo/docs 明确有 polynomial nonlinear state-space 的辨识与模拟；docs 标明由 VUB MATLAB PNLSS 实现移植。 | 只核对 repo/文档，未审计核心文件或运行。CPU 可执行性仍需 root 核查，不能宣称已复现 2017 论文。 |
| [chandratue/HystRNN](https://github.com/chandratue/HystRNN) | API tree 确认 `Exp1/HystRNN.py` 与 B/H CSV；另有 DC-bias、dt、Duhem、noise 等目录。 | 下节代码缺陷必须修正。单一脚本的审计不能外推到所有目录。 |
| [mlmathphy/DiffusionMemory_GLE2](https://github.com/mlmathphy/DiffusionMemory_GLE2) | API tree 确认 multiscale 与 vector-GLE 的 `generate_data.py`、conditioning、train、exact_refs/evaluation 等实际文件。 | 已有 fading-memory 对照；不把其 oracle/reference-assisted bank selection 直接作为新任务的可用信息。 |
| [Princeton 官方 MagNet 数据入口](https://www.princeton.edu/~minjie/magnet.html) / [官方代码](https://github.com/PrincetonUniversity/magnet) | 官方页面链接 transient Challenge 2 数据，列 10 种材料；代码为实际 excitation/response 数据平台。 | 本轮未下载数据。PI-HNO 报告的 14-material MagNetX 与本入口不能直接视为同一 release；先核实版本、文件大小和 complete-run IDs。 |
| [JaehwanJeon/Testing_torch](https://github.com/JaehwanJeon/Testing_torch) | 论文直接给出的 code/data 入口。 | 本轮只核对论文链接，未进一步审计仓库；不要表述成已复现。 |

## 5. 确定的实现风险：HystRNN Exp1

读到的 [原始脚本](https://raw.githubusercontent.com/chandratue/HystRNN/main/Exp1/HystRNN.py) 给出如下可定位证据：

- L103–104 输入为 `(batch=1, sequence_length, input_size=2)`。
- L136–142 `forward` 以 `x.size(1)` 初始化 hidden，却沿 `x.size(0)` 循环。因此在这个训练调用中循环一次，各时间点实际成为独立 batch 项，不能从该执行路径推断跨时间 recurrent memory。
- L163–173 测试每一步重新调用 model；L137–139 每次清零 hidden。这个路径相当于由当前 B,H 做 one-step 更新后自回归，未携带跨调用 hidden。
- L49–50 对整个 `Major2FORC` 数组做 min/max，随后 L55–61 才分 train/test。用于新研究时应改成 train-only scaling。

这是静态代码推断，没有实际运行。应修复轴和 state-carry 后重建公平控制，再核对 paper 设定；不能把这一脚本的问题当作“RNN 无法表示非衰减记忆”的科学证据，也不能以修复该脚本作为方法贡献。

## 6. 回答 brief 与建议

**RQ1：** 相邻领域已经实现 partial-observation 的记忆闭包、随机 rollout、真实 hysteresis 条件响应，以及动力学子模块组合。这些能力未必使用 world-model 命名，但必须纳入同层比较。它们无需被强加作者未声称的因果能力。

**RQ2：** 非衰减与衰减记忆共存时的跨系统复用是可清晰检验的问题；目前没有证据证明它会在 standard hybrid、合理时间输入、expressive heads 和有限上下文参数辨识后仍失败。观察投影变化、普通 colored noise、一般模块组合在本轮都没有形成比既有方法更明确的新 lead。

**RQ3：** 当前接受方向数仍为零。若根任务要保留一个便宜的 triage，可保留本 Q，先审计真实数据与标准 hybrid；只有它们出现事先定义且能复现的残余失败，才值得继续设计机制。不要直接启动一组大规模新神经训练，也不要把“context + Preisach + filters”包装成已成立的新框架。

## 参考核验状态

表中已使用的论文均通过标题/作者存在核查，且机制表述对照了摘要或全文。GLE、PI-HNO、PAL、Jeon 等读取全文；PAL 只用于确认提案存在，不把其未审计理论当作可信通用定理。HystRNN/Preisach/GLE 仓库存在与上述具体文件通过官方 tree/raw 内容核对。没有未验证参考文献进入论证；没有将缺代码等同于 research gap。
