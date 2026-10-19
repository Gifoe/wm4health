# C41 独立批评与方法学审查：状态信息、模型误差与推断补偿

研究范围：只读共享 `RESEARCH_BRIEF.md`，未读其他 perspective 笔记；核验公开一手论文，未运行 LIFTS、训练模型、下载数据/权重或访问服务器。本文的精确矩阵算例已用本地标准库复算；其他实验均为待执行方案。核验日期：2026-10-17。

## 摘要

现有证据不足以把“更接近生成模型后验”当作“更有用的 learned state”。编码器可能补偿错设转移；消除这种补偿会提高模型内部一致性，同时降低真实预测质量。这个现象本身不新，PSIM 已直接针对 predictive-state inference 优化。可保留的研究问题是：在观测视图与动力学的已见因素、未见组合中，有限锚定信息是否能让共享状态降低组合预测的样本或计算成本？必须先排除普通联合 factorized HMM/SSM、转移重新拟合、有限状态对齐混合及强直接预测器。没有这一残余证据，不能据此宣称新 world-model framework。

## 1. 冻结问题与证据方法

RQ1：已有 complete framework 的结果到底说明 state information、transition approximation、inference 中的哪一项？

RQ2：能否构造已知真值的有限实验，将三项分开，同时承认其交互？

RQ3：在强普通方法后，是否还存在“共享状态 × 有限观测锚 × 动力学复用”的可检验残余？

本笔记从批评/方法学视角进行两轮检索：先核验当前最接近的 WorldTS、LIFTS，再检索直接 inference learning、粒子推断训练目标、多元评分和 HMM 可辨识条件。以下只保留六个一手来源；没有把二手网页的判断当论文结论。

## 2. 六个决定性一手来源

| 来源及核验位置 | 确实支持的事实 | 对当前方向的约束 |
|---|---|---|
| Zhu et al., 2026, [WorldTS](https://arxiv.org/html/2609.31162v1)，§4.5、Table 4、A.4 | 一阶段对照保留 causal encoder/decoder，但同时去掉 future-state branch/state losses，改为端到端 observation MAE；总 epoch 匹配。 | 支持完整设计有用，未单独识别“新增状态信息”与训练路径/正则化的贡献。future encoding 作为训练监督本身不构成信息泄漏。 |
| Wu, Wei, Wang, 2026, [LIFTS](https://arxiv.org/html/2610.03916v1)，§3–6、Eq.18、§6.1 | 联合 measurement/mask 模型及 filtering/forecasting/smoothing/imputation；synthetic encoder calibration 在作者实验中无系统改善，后续设 `lambda_h=0`。 | “给 encoder 加模型后验校准”已是近邻，且作者当前证据不支持其普遍收益。本文未执行作者源码，不把理论/源码或小诊断写成完整复现。 |
| Sun et al., ICML 2016, [Learning to Filter with Predictive State Inference Machines](https://proceedings.mlr.press/v48/sun16.html)，摘要及论文 §1–4 | 直接学习 predictive-state inference predictors，讨论 realizable 与 agnostic 情形。 | “不要求物理状态、改学对未来有用的 state”不是新问题；生成模型拟合与 inference usefulness 不一致也不是新动机。 |
| Maddison et al., NeurIPS 2017, [Filtering Variational Objectives](https://proceedings.neurips.cc/paper_files/paper/2017/file/fa84632d742f2729dc32ce8cb5d49733-Paper.pdf)，§4、§6、Table 1 | 同类 VRNN 架构通过 ELBO/IWAE/FIVO 训练可得到不同测试 likelihood bounds，包含计算预算匹配。 | 训练/推断目标足以改变结果，不应把架构相同但目标很弱的对照当“普通方法极限”。该文不直接证明 forecasting 优劣。 |
| Scheuerer & Hamill, MWR 2015, [Variogram-Based Proper Scoring Rules](https://journals.ametsoc.org/doi/10.1175/MWR-D-14-00269.1)，§3–5、Table 1；[NOAA 作者稿](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf) | ES 对错误相关结构的区分能力有限；案例中 random-Q 与 ECC-Q 的边际相同，但相关结构不同，VS 的区分明显更强。 | ES 是 proper 不等于在当前样本量下对特定动力学差异有充分检验力。VS 也不能单独认证完整联合分布。 |
| Gassiat, Cleynen, Robin, [Finite state space non parametric Hidden Markov Models are in general identifiable](https://arxiv.org/abs/1306.4657)，Theorem 2.1 | 对已知状态数、stationary HMM，在 transition full rank、emission laws 线性无关时，三连观测分布识别参数，至多差一个状态置换。 | 单个 HMM 可辨识到置换，不自动保证不同 view 的置换能跨环境对齐。下文的组合反例是自行推导，不是该定理的反例。 |

这些证据并非共同证明所有 learned-state 方法无效。它们共同否定的是从“latent loss 下降/模型后验更准/ES 变好”直接跳到“新增可复用动力学信息”的论证。

## 3. 三种误差必须在预测分布层面区分

令历史为 `H_t`，任务未来为 `Y=x_(t+1:t+h)`，学习摘要为 `Z=f(H_t)`。研究目标应是对约定任务族的 predictive sufficiency，不要求 Z 等于模拟器状态。

### 3.1 状态信息：弱 probe 的失败不是信息丢失证明

对任意 readout `r(Y|Z)`，在 log score 下有：

```text
E log[p*(Y|H) / r(Y|Z)]
  = I(Y;H | Z) + E_Z KL(p*(Y|Z) || r(Y|Z)).
```

第一项才是该任务上的不可恢复预测信息损失；第二项是 readout 近似/优化误差。因此实际 probe 的 excess NLL 是信息损失的上界，不能由“大 probe error”推出“大信息损失”。低误差 probe 可以正面证明其保留了所测未来信息。该等式是定义推导，不是引用论文中的实验结论。

最少要做：冻结各 encoder，使用相同训练样本、充分的非线性概率 readout 与样本/容量曲线；同时比较 history-input readout。若普通 readout 修复差异，应按 readout 结果关闭该动机，不回到已关闭的 latent geometry 故事。

### 3.2 已知有限状态诊断：用共同坐标作 swap，而非跨模型硬换 latent

在已知真值的小 HMM scaffold 中，固定共同 emission B，记真实 transition 为 P*、拟合 transition 为 Pθ，真实 belief 为 b*、拟合模型 exact-filter belief 为 bθ、amortized belief 为 bφ。M(P,b) 表示从 belief b 出发、使用 P/B 生成任务未来的 law。至少计算：

| 预测 law | 隔离用途 |
|---|---|
| `M(P*,b*)` | 真正 oracle |
| `M(Pθ,b*)` | 当前真实 belief 已知时的 transition error |
| `M(Pθ,bθ)` | 拟合生成模型自身的 exact-inference prediction |
| `M(Pθ,bφ)` | amortized prediction |
| `M(P*,bφ)` | encoder 在共同诊断坐标中的误差；与上一行比较可见补偿 |

比较后两种 inference 对象时，既报告它们彼此的分布距离，也报告对真实未来的 proper score。后者差值可以为负：更不忠实于拟合模型的 inference 可能有更好的真实预测。不要把这些有符号风险差当独立、非负、可加的误差分解。

共同有限状态坐标是诊断设计的便利条件，不能推广为 native LIFTS/WorldTS 必须恢复物理坐标的要求。任意学习 latent 的 g、f、h 之间硬换“真转移”会发生坐标不兼容。隐式 decoder 若无可评估 likelihood，也不能声称普通 PF 已给出 exact posterior；应保留“native 输出诊断”和“共同坐标 scaffold”两类证据的边界。

## 4. 精确反例：状态校准可能破坏一步预测补偿

这只是指标/归因反例，不是待发表方法，也不是普通模型必败设定。四状态、均匀稳态 π，`J=1π`，真实与错设转移为：

```text
P* = 0.8 I + 0.2 J
Q  = 0.9 I + 0.1 J.
```

先用 fully observed `s_t` 作零混淆诊断，故真实与 Q 的 exact current posterior 均为 `e_s`。令 amortized state law 为：

```text
bφ = (8/9)e_s + (1/9)π.
```

则 `bφ Q=e_s P*`：一步预测完全正确。将 encoder 校准成 Q 的 exact posterior `e_s`，反而变为错误的一步预测。但开放滚动时：

```text
bφ Q^h = π + 0.8 × 0.9^(h−1)(e_s−π)
e_s(P*)^h = π + 0.8^h(e_s−π).
```

本地标准库对 endpoint categorical law 的复算结果如下，KL 单位为 nat：

| h | KL 真值→补偿 encoder | KL 真值→Q exact posterior | TV 补偿 encoder | TV Q exact posterior |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.0320983 | 0 | 0.0750000 |
| 4 | 0.0371726 | 0.0796586 | 0.1302000 | 0.1848750 |
| 8 | 0.0523042 | 0.0783335 | 0.1611490 | 0.1970213 |

因此一步观测任务、状态校准和多步动力学任务可以给出不同排序。反过来，重新拟合 Q 使之等于 P*，或直接学习 predictive-state inference，就能消除此反例的研究动机。不能靠长期冻结错误 Q 来制造一个只有新框架能解决的问题。

最小实验：固定同一数据/同一 Q，对 encoder 的 `lambda_h=0` 与预注册校准权重比较 h={1,4,8}，再增加可自由重新拟合 transition 的完整普通对照。若可训练 joint transition 或同预算直接 GRU 清除差异，停止“校准 × 转移”方法主张。若需要研究残余，应另行证明真实有限数据/预算下存在稳定的 transition/inference 交互，不能把上表当作证据。

## 5. 可恢复问题：已见 view × 已见 dynamics 的未见组合

### 5.1 覆盖图连通不是充分条件

训练格为 `(A,d1),(B,d1),(A,d2)`，测试格为 `(B,d2)`。每个 view、每种 dynamics 都已出现；没有要求处理任意新 sensor map。

取三状态，uniform initial/stationary distribution，令 d1 为 `αI+(1−α)J`，d2 为：

```text
P2 = [[0.7, 0.2, 0.1],
      [0.1, 0.6, 0.3],
      [0.2, 0.2, 0.6]].
```

A 的 emission 为状态标签，B 可以是同样标签，也可以交换标签 1/2。d1 对这个置换不变，所以两个假设的三格训练 observable laws 完全相同。然而 B,d2 的 observable transition 分别是 `P2` 与置换共轭 `SᵀP2S`，不同。uniform current state 下，两种 held-out 一步 law 的平均 KL 为 **0.1014841 nat**，已用本地标准库复算。

这证明的只是：缺少 alignment 信息时，存在两个训练数据无法区分、但测试预测不同的假设。没有用 physical identifiability 为难模型。将训练中无法消除的歧义，写成某 baseline 的能力缺陷是不成立的；新方法如果在该情形下“零样本准确”，应先查额外锚/先验/数据泄漏。

### 5.2 足够且可审计的锚；不宣称通用最小定理

可用以下任一机制建立可检验任务：

1. **少量同轨迹 paired prefixes**：在训练已有格中同时记录 A/B 视图。无噪声有限状态例中，覆盖全部会影响测试预测的状态的配对即可固定标签；有噪声时需要足够的联合样本，不存在“一对样本普遍足够”的保证。
2. **共同 anchor sensor**：每格都有同一个已见观测通道，其 state-dependent laws 足以区分对未来不同的状态。可采用线性无关 emission 的有限 HMM 作为透明充分条件。
3. **有辨识力的桥接 dynamics**：若共同 d1 的状态置换自同构群是平凡的，足量数据下可以通过匹配 d1 对齐 view。近似对称虽未必不可辨识，有限样本对齐可以很差。

真正必要的只是消除会改变 held-out observable law 的剩余对齐歧义；预测等价的 latent label 不必区分。上述条件是本诊断的充分设计，不是任意非线性系统的最小覆盖定理。Gassiat 等的单 HMM 定理也没有自动给出跨 view 的结果。

### 5.3 ordinary complete baseline 必须先通过

首要 baseline 是联合 factorized HMM/SSM：所有训练格共用一个 latent，transition 为 P_d，emission 为 B_v，联合 likelihood 拟合并使用正确 forward filter。它已经具有“共享状态、view-specific readout、dynamics-specific transition、跨组合复用”的完整结构。不能只与每格独立模型比较。

有限状态控制还必须包含：

- 对全部训练格联合重新拟合 P_d/B_v，使用相同 paired anchors，多起点或可验证优化；
- 对 K=3/4 的残余 view permutations 做小型 mixture/marginalization，而非故意只取一个错对齐点估计；至多 6/24 个置换，成本有限；
- 同预算、相同 view/dynamics 标签与配对信息的 conditional GRU autoregressive density/mixture 模型；
- 对连续传感器版本，普通 hierarchical nonlinear SSM，用足够但有限的粒子/推断预算曲线确认是否仅为 inference 未收敛。

这些对照可能直接解决任务。那是有效 negative，不应因为它们使用已知模块便略过。WorldTS/LIFTS 的 native 范围与此新组合任务不完全相同；“未直接提供该接口”不等于证明它们预测失败。

### 5.4 一个最小、可证伪的有限资源问题

**RQ-C：当桥接动力学接近对称、但 paired anchors 有限时，predictive-state training 能否以更少训练样本或更少计算降低 held-out view×dynamics 的真实 joint predictive risk，超过联合层次 SSM 与有限对齐 marginalization？**

这项交互具体是 `anchor budget × bridge distinguishability × rollout horizon`。不是再证明“state 学习有用”，也不是只添加 consistency loss。其价值必须来自跨格共享状态的样本/计算效率；当前检索未证明该残余存在。

建议第一关用 K=3 的 categorical/Gaussian HMM，所有因素均已见：

- d1 由对称矩阵与固定非对称矩阵按 ε 混合；预先选 ε={0,0.1,0.3}，ε=0 且无 anchors 只作为不可辨识 sanity check，不作为方法胜利；
- 三个训练格、一个测试格，全部固定；每训练格 256 条长 64 序列，prefix 16，h={1,4,8}；
- paired prefix 数 m={0,4,16}，从训练轨迹独立抽取；不得从测试格取 anchors；
- 主结论采用有 anchors 且可辨识的格；额外 1024 条/格只检验样本效率曲线，不通过调 ε、m 或 seed 挑出“成功区”；
- 比较上节完整普通对照。时间序列的连续版本可将状态输出换成固定 Gaussian sensors，视图使用各自已见的不同 sensor maps；仍不引入任意未见映射。

**诊断 swaps**：oracle B_v／learned P_d、learned B_v／oracle P_d、oracle alignment、完全 learned；对相同 fitted generative model 比较 exact 与 amortized filter。已知 state scaffold 的这些 swaps 只用于误差归因，测试任务最终以 observable future law 计分。

**决定性 falsifiers**：

1. 联合 HMM/SSM 或小型 alignment mixture 消除差距：该 fixture 不支持新框架动机；
2. 只给 oracle alignment 就消除全部差距：主问题是普通有限对齐，不是更强动力学；若普通 marginalization 同样有效，停止该方法角度；
3. 换强 readout 就清除收益：不是新增 state information；
4. 放开 transition joint refit 后收益消失：先前只是在补偿冻结错设转移；
5. 高锚/强非对称桥接下差距不缩小：所谓 anchor/state-reuse 解释受到反证，必须寻找其他归因；
6. 仅 ES、latent MSE 或某单阈值事件改善，joint log risk/预声明可观测任务未改善：不支持 useful dynamics 主张。

## 6. 指标、检验力与资源边界

有限 HMM 的条件 likelihood 可用 forward algorithm 精确求值。主指标为 held-out joint future conditional NLL，相对真模型 oracle 的 expected excess 即 conditional KL；期望用独立模拟未来估计，报告 Monte Carlo 误差。endpoint KL 只诊断多步衰减，不能代替 joint trajectory law。

同时报告有限任务族：一阶/滞后差分分布、累计和或运行均值、两个预声明 interval-event 的 Brier score；可加 VS 与 ES，但不能把 moment 或 VS 一项当完整 law 认证。Scheuerer–Hamill 的经验反例恰说明边际相同不等于联合质量相同；这不意味着 ES 理论上不 proper。

至少用两档训练预算、两档 inference 预算检查有限资源解释；不能诉诸无限粒子/无限数据必败。独立轨迹作重采样单位，避免把相邻窗口当独立样本。先用验证集固定配置，测试组合只执行一次主比较。小 fixture 是机制诊断，不是原论文 benchmark 复现，也不是完整新框架结果。

## 7. 回答研究问题与新颖性不确定性

**RQ1**：WorldTS 支持完整 latent-supervision/two-stage 设计有效；尚不能单独归因于新增状态信息。LIFTS 已提供 complete joint model/inference，synthetic calibration 的现成负结果禁止把校准本身当未经探索的突破。直接 predictive-state inference 至少已有 PSIM；训练/推断目标变化也有成熟普通控制。

**RQ2**：共同 finite-state scaffold 的 truth belief、learned transition、exact fitted-model posterior、amortized posterior 交叉是最便宜的清晰诊断；冻结 latent 的 readout 诊断须按可恢复信息的上界解释。上面的精确补偿反例说明必须同时评估 state calibration 与真实多步预测，并允许普通 joint refit。

**RQ3**：可恢复候选是有锚、可辨识的 factor-combination reuse，明确检验有限锚、桥接对称性和 rollout 的交互。首关很可能被 ordinary factorized HMM/SSM 与小型对齐混合解决。目前没有证据支持一个新 world-model framework，只有一个比泛化“状态不准”更可审计的有限研究问题。跨组合本身、共享 latent、对齐不确定性与 predictive-state training 均有明显既有方法覆盖风险；须另做最近邻检索，不能把“六个来源没有同名方法”写成 novelty。

本笔记不重新打开 C15–40 中已经关闭的 covariance、posterior conditioning、readout geometry、rank coupling、internal-law inconsistency 或 generic observation-updater 动机。
