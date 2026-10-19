# C32 独立核查：预测状态目标能否构成一个新的 TS world-model 框架

日期：2026-10-17。视角：critic / methodology；按 `deep-research` 的 clean-context 子视角执行。边界：只读本轮 brief、相关 WorldTS 源码与公开一手论文；没有训练、SSH、删除、外部发布或读取 cycle6。这里只写本文件。

## 摘要与判定

本视角目前提名 **NONE**，但不是因为“组件已有”或“尚未实测”而否决一切有限预算组合。被否定的是一个具体推理：**未来 target encoder 只看局部未来观测，所以它不是完整 history state，所以原生条件预测任务必然损失必要信息。** 最后一步不成立。未来特征的条件预测本来就可以是 predictive state；对 point forecast，需要保留的也可能只是相应的条件均值或分位数，而不是完整未来分布。

仍然可做的精确问题是：**在 native 输入、native 维度与相同总预算下，现有 latent target 是否形成了一个可重复的、对最终 forecast 有害的有限特征压缩，而普通 loss 校准、decoder 扩容、EMA 或直接 observation 监督不能同样便宜地修复？** 这需要下面的 attribution 实验。它可以保留一个小而具体的机制研究入口；目前不能仅凭“观测 ≠ 状态”把入口升级为新框架。

## 1. 研究问题与本轮证据级别

本报告回答 brief 的三个问题：RQ1 的目标究竟预测什么；RQ2 的强先例真正保证什么；RQ3 怎样用一个单卡可执行的小实验区分 target、优化、容量、输入协议和 decoder 问题。用户允许 per-dataset training；不加入 zero-shot、动作、因果或物理解释作为合格条件。

本轮进行了公开网页检索，重点覆盖 2026-04-17 至 2026-10-17，回溯 PSR、PSIM 和 self-predictive history learning 的必要经典。下面列入的方法均检查了正文目标、命题或实验，而非仅看标题/摘要。WorldTS 本地快照做了静态源码检查，**没有运行模型**。VJEPA 的 ICML 2026 正式 29 页 PDF 及 HSE-PSR PDF 通过 Python 在内存解析，未落盘；其余主要用官方 arXiv HTML。没有把第三方综述和自动 review 当事实证据。

检索发现 MDN-VJEPA 的 arXiv 编号为 2610.13956，而官方页面列出的 v1 日期是 2026-08-21；两者存在表面日期不一致。本文只将其作为截至本次访问已公开、正文可核查的方法先例，不依赖其精确先后日期判断 novelty。

## 2. 原生 WorldTS：应批评哪里，不应批评哪里

论文 §4.2、Eq.18–21 的关键事实是：future target 编码采用局部 causal patch；Stage 1 是 **L1 + flatten 后的 trajectory cosine + VICReg**；Stage 2 冻结 encoder/predictor，**同时在预测 code 和真实未来 code 上监督 decoder**。decoder 还接收最近历史 states 的 patch。因此 `D(E[Z]) != E[D(Z)]` 不能直接构成对它的攻击：decoder 学的可能是由实际预测 code 到条件最优观测的映射。它也不是精确的 conditional-mean feature regression，除非另行去掉 cosine/regularizer 并切换平方损失。[WorldTS 正文](https://arxiv.org/html/2609.31162v1)

本地快照进一步确认：

- `encode_future_target` 将最后 `state_patch_len - 1` 个历史 target observations 拼到未来，因此“target 完全不见历史”字面上不对。
- `forward` 先用历史窗口的 RevIN mean/std 归一化 future target；真实 decoder 输入不仅有 `h_pred`，还包括 `h_hist` 的尾部，以及输出时恢复的 RevIN statistics。
- `LatentStateLoss` 已支持 `mae/mse/smooth_l1/normalized_mse`，不能把切换一个 latent loss 宣称为新架构。
- 本地训练器支持 `stage2_decoder_pred` 与 `stage2_decoder_mix`；论文是 mix 的表述。诊断时必须保留实际运行配置，不能用错误训练流程制造差距。

检查位置：[模型](D:/chenyu-iclr/tswm_discovery_20261016_v1/upstream/WorldTS/DAG-bench/ts_benchmark/baselines/worldts/models/worldts_model.py)、[latent loss](D:/chenyu-iclr/tswm_discovery_20261016_v1/upstream/WorldTS/DAG-bench/ts_benchmark/baselines/worldts/losses/latent_state_loss.py)、[训练器](D:/chenyu-iclr/tswm_discovery_20261016_v1/upstream/WorldTS/DAG-bench/ts_benchmark/baselines/worldts/worldts.py)。这些是**当前本地快照**，未核验为未经修改的上游 commit；涉及论文原始方法的判断以上面的正式正文为准。

## 3. 最相关先例：解决了什么，保证到哪里

这里的“强”指必须面对的直接机制先例，不表示所有方法在 TS benchmark 上已被证明 SOTA。

| 路线与一手来源 | 已有的直接机制 | 真实保证/适用条件 | 对 C32 的含义 |
|---|---|---|---|
| [HSE-PSR，Boots et al., 2013](https://proceedings.mlr.press/r11/boots13a.html)，核查 PDF §3–5 | 用 characteristic RKHS 的 conditional embedding operator 表示 future tests 的条件分布，kernel Bayes update | characteristic mean map 的 injectivity；还需足够长 core tests、conditional operator 存在等条件；sample representation 随样本数增长 | “预测未来特征的条件期望可成为 state”是已有完整思路；但不能把无限 RKHS 的充分性无条件转移给固定 32/128 维神经 code |
| [PSIM，Sun et al., ICML 2016](https://proceedings.mlr.press/v48/sun16.html)，核查作者 PDF §3–4、Thm4.1–4.3 | 直接监督学习 `m_t = E[phi(future block) | history]` 及其 filtering update，允许 nonlinear regressor；含 agnostic/finite-sample reduction | `k`-observability 和 sufficient future features 是假设；exact recovery 还需 realizability、无限数据、每步问题解到 Bayes optimum；finite-sample bound 依赖 learner class | 新增 future-feature bank、回归 history-to-future moments 或称它 predictive-state interface，单独不构成新机制 |
| [Ni et al., ICLR 2024](https://arxiv.org/html/2401.08898v3)，核查 §3–4 | target 可以是下一步 **history representation**；明确区分 full latent-distribution prediction (ZP) 与 expected latent prediction (EZP)；分析 stop-gradient/EMA | self-prediction 本身允许常量解；平方误差与 full distribution matching 不等价；抗坍塌定理使用线性 encoder/transition 和优化条件 | “history-aware target + stochastic ZP”是必须比较的成熟基线；不能把保留已知 history 视为充分创新 |
| [VJEPA，ICML 2026](https://proceedings.mlr.press/v306/huang26ba.html)，核查正式 PDF §3、§5、App.E/G | 预测未来 latent distribution，变分目标；BJEPA 加 PoE prior；可进行 latent sampling | Thm5.2 是 **若**预测充分且 cost 对 latent 可测，则可用 latent 做 control；不是训练自动获得观测充分性的定理。App.E 的抗坍塌论证还存在下文的条件缺口 | “随机 JEPA + sufficiency”不是空白；但不必夸大成已经解决所有 finite learned target sufficiency |
| [MCJEPA / HMM interpretation，2026-08](https://arxiv.org/html/2608.13621v1)，核查 §4–7 | 离散 transition matrix 实现组合；明确做 restricted/current-state 与 history-augmented residual predictor 的 held-out 比较 | 有限 time-homogeneous chain 的矩阵幂有精确 CK composition；HMM equivalence 另有更强条件。§6 明确 residual diagnostic 只单向否定，不认证充分性 | `Delta_hist`、cross-fitting 和“区分 state 与 transition misspecification”已有；这是一套应复用的诊断，不是待命名的新方法 |
| [Orthogonal JEPA，2026-08](https://arxiv.org/html/2608.20065v1) / [JEPA-Anything，2026-09](https://arxiv.org/html/2609.20800v1)，核查 factorization 与 proposition | 互补 target factors、独立 prediction pathways、factor activity / encoder variance | exact orthogonal、full coverage `Kr=d` 时保证无重叠分解与稳定 synthesis；不是统计独立、语义 disentanglement 或预测充分性保证 | “dominant feature 吞掉其他因素，所以拆多个 head、正交化”已有最直接对手；但允许检验其在有限预算 TS 上是否仍欠缺 |
| [SC-JEPA，当前版本](https://arxiv.org/html/2602.04643)，核查 §3、App.A.3 | soft codebook、fine/coarse future target、多尺度预测和辅助 reconstruction | bounded prototype geometry 的 drift bound；non-collapse 下界需要有效 code usage、sharpness、prototype separation 等条件；未保证 forecast task sufficiency | “多尺度 future state + codebook 稳定训练”已是 TS 专门先例；若只是添加这些组件，应当作为普通修复基线 |
| [UWM-JEPA，2026-05](https://arxiv.org/html/2605.25313v1)，核查 §2–3、App.S1 | density-matrix belief latent + unitary blind rollout | joint-state spectrum/entropy 保持；作者明确不保证 reduced state 的可读性，长 horizon probe 也有失败范围 | “保住 uncertainty 的 latent geometry”已有具体机制，但这不是当前 point TS 任务必须满足的 gate |
| [MDN-VJEPA，当前公开正文](https://arxiv.org/html/2610.13956v1)，核查 §3–7、§8.6 | mixture predictive head、共享 trajectory-level mode、direct-versus-composed distribution consistency | 有实测 controlled branching/coherence；component semantic identity 无保证，CK 不是 predictive fidelity；尚未展示大规模真实 learned-encoder 整体系统 | “多模态 stochastic latent + coherent trajectories + CK”也有直接先例。不能只把 diagonal Gaussian 换 mixture 当新框架 |

额外核查的 [NE-Dreamer](https://arxiv.org/html/2603.02765v1) 在 RSSM 上用 history-conditioned causal Transformer 预测 next observation embedding，提供 memory-heavy task 的实证。它进一步说明“future observation embedding target”与“history-aware predictive model”并不互斥。其任务是控制，不能拿其返回值排名直接推断 TS 预测优劣。

这组文献的共同边界很清楚：**几何稳定、避免全坍塌、表示有历史、能表达随机未来、有限链 CK 一致、对原始 forecast functionals 充分，是不同结论。** 前几个已覆盖得很直接；最后一个不能靠前几个推出。但“未有无条件充分性定理”也不能自动构成一个有实用价值的新问题。

### 3.1 对 VJEPA 抗坍塌保证的独立核查

正式 ICML PDF App.E 声称 target diversity + predictor family 能随 context 变化，就排除 constant-context global optimum。证明中的严格 gap 用的是 `I(Z_T; X_T | xi) > 0`，但 predictor 只能看 `X_C`。两者不等价。

取固定、非退化 target encoder，`X_C` 与 `X_T` 独立，predictive family 足够丰富且能依 context 改变。两个所写条件成立；但最优预测仍为 aggregate target law，constant context 同样达到最优。正确的预测收益应由 `I(Z_T; X_C | xi)` 控制，并还需可表达/优化等条件。若 target 可一起更新，KL 向 reference prior 本身也不能无条件排除 target collapse。**这是本报告的数学反例，不是论文报告的实验结论。** 它不否定该方法作为 stochastic baseline，只否定把这一定理当成通用 target sufficiency 的证明。[正式 PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/huang26ba/huang26ba.pdf)

## 4. 精确 no-go：局部 future target 不是必然的 forecast 信息瓶颈

以下是本报告推导；只用于校准科学问题，不声称新定理。

设 `U=(合法历史, 合法 covariates)`，未来 block 是 `Y`，冻结 target feature 为 `phi(Y)`。在无限容量、population squared feature loss 下，预测 code 为

`m(U) = E[phi(Y) | U]`。

此 code 对 MSE forecast 充分，当且仅当存在 readout `d` 使

`E[Y | U] = d(m(U))`。

它**不要求** `phi(Y)` 本身是物理 state，也不要求 target encoder 看整个历史。特别地，若 `phi(Y)=[Y; psi(Y)]`，直接取出前一块就能得到条件均值。target 只是 observation outcome，也完全可以足够当前 point forecast。这是对广义“未来观测不等于 state，所以任务必坏”论证的反例。

更精确地，若未来只取有限支持 `{y_1,...,y_K}`，令 `F` 的第 `k` 列为 `phi(y_k)`，`V` 的第 `k` 列为 `y_k`。对于**所有**该支持上的条件分布，均值都能从 `E phi` 恢复，当且仅当

`rowspan(V) ⊆ rowspan([1^T; F])`。

证明很短：若包含，则 `V = a 1^T + B F`，所以条件均值可仿射恢复。若不包含，存在 `v` 满足 `[1^T;F]v=0` 但 `Vv != 0`；从一个 interior probability vector 取 `p±epsilon v`，得到同 feature expectation、不同 observation mean 的两种未来 law。此时任何 decoder 都无法恢复两者的均值差，才是真正的 feature-induced floor。

这揭示两种容易混淆的情况：

1. `phi` 对每个样本可逆、reconstruction 很好，**仍不意味着**其有限条件 moments 对整个条件分布族充分。
2. `phi` 不保留完整分布，**仍可能完全满足**当前 MSE/MAE 的任务。

WorldTS 的真实 feature 还通过历史 prefix/RevIN 依赖 `U`，decoder 也直接收到一部分 side information `B(U)`；因此其 sufficiency 要考察 `d(m(U),B(U))`，而不是只考察 `m(U)`。上面的有限支持判据是在固定 side-information 条件下对固定 feature map 的判据，不能直接当成对其整个实现的定理。

WorldTS 用的是 L1+cosine，不能把上述 conditional-expectation 方程直接写成其精确解。纯 L1 对应 feature-wise conditional median；cosine 和 pooled regularization 又改变 optimum。若指控的是 MSE 指标上的 mean/median 不匹配，先比较普通 metric-aligned loss；若只关心 MAE，则不能要求模型恢复不影响条件 median 的随机分支。

这是 **narrow no-go**，不是“任何有限预算 target framework 都不可能优于 WorldTS”。有限数据、优化、capacity allocation 完全可能有重要增量；但其机制必须具体落在同预算风险差上。

## 5. 可执行的最小判断：先做 native headroom attribution，再决定是否救援

### 5.1 第一步：一个固定 checkpoint 的信息路径矩阵

优先用已存在、协议合格的 native checkpoint；若没有，另行授权训练，**本轮没有启动**。先选择 native DAG-bench 的 Energy/短 horizon 作为小规模诊断；它不是 C11 Gaussian fold toy。只在诊断通过后再加 Sdwpfh1/长 horizon，最后才扩到 text 或 image 场景。第一数据集只能筛方向，不能建立通用框架结论。

记录 inference-available 的完整信息集：预测 states、decoder 实际使用的 historical states、RevIN statistics、所有合法 covariates。下表每个 probe 用相同训练窗口、同等容量/regularization 的两种 learner（例如 ridge/random features 与小 MLP），chronological out-of-sample；重叠窗口用 block bootstrap，block 长度至少覆盖 lookback+forecast，避免把高度相关窗口当独立样本。

| 比较 | 只改变什么 | 能支持的归因 |
|---|---|---|
| 原 decoder vs 更灵活 readout `q(B)`，其中 `B` 是 **完整原 decoder 输入** | decode capacity/optimization | `q(B)` 明显改善表示已有可利用信息；不能宣布 target 先天不足 |
| `q(B)` vs `q(B, 全部 h_hist, encoded legal covariates)` | 加入 predictor 已经能看的 latent 信息 | 额外改善提示 predictor/latent prediction objective 未把已有 latent context 传给输出；还不是 history encoder 丢信息 |
| 上一行 vs 加同协议 raw history/covariates 的 matched probe | 加原 encoder 之前的信息 | 额外改善才支持 encoder compression/feature choice 的有限预算不足；必须控制 probe flexibility |
| 使用 oracle future target code 的 readout | 用真未来 code 作诊断，绝不用于部署比较 | 测样本层面 target 可解码性；成功不证明其 conditional moments 充分，失败也可能只是 decoder 未拟合 |

HMM-JEPA 已明确提出同类 residual tests，故这里不把 probe 包装成新研究贡献。其价值是避免再用一个人为二维瓶颈或一个 target loss 数字错误归因。[现有方法学先例 §6](https://arxiv.org/html/2608.13621v1)

### 5.2 第二步：仅在 residual 存在时，比较最强普通修复

固定相同输入、split、normalization、总 wall-clock/parameter budget（同时报告预算，不能只对齐 epochs），保留三个 seeds。先跑四个短臂；每个臂只能用相同数量的 validation choices：

1. 原始两阶段 WorldTS；decoder training input 与实际 inference code 一致。
2. 相同 backbone 直接 observation forecasting / end-to-end，匹配最终 MAE 或 MSE 目标。
3. 原结构的 loss/calibration repair：EMA 或 frozen-target control，latent L1/cosine 比例调整，以及最终指标一致的 loss；选项总数预先固定，给其他臂同等搜索预算。
4. 一个必要的保底控制：若 native 每步输出维度 `C <= D`，在相同总 latent width 下保留 `C` 个 raw identity target 坐标，用平方损失监督相应 conditional mean；其余坐标照常预测。它检验上面的 task-moment 充分性救援是否已经足够，不声称这是新框架。若 `C > D`，普通低秩 projection 只保留相应线性 functionals，不具有完整均值充分性保证，不应把它冒充 identity control。若主指标是 MAE，另用 pinball/median 版本，不混报 objective。

如本轮想推进 stochastic future 表达，再加一个 conditional MDN/flow observation head 和一个 VJEPA/MDN target head；必须用 CRPS/适当 joint score 等正确评分，并与同输入 observation density model 比。仅 MSE 改善不能识别“完整随机状态”机制；仅 latent energy/CK 改善不能识别真实 forecast 改善。

### 5.3 明确的继续/撤销条件

以下数值是一个可预注册的工程筛选阈值，不是普适统计定律：在至少两类真实 native 数据上，相对最强普通修复有超过 3% 的主要 forecast-risk 改善，三个 seeds 同向，时间 block CI 排除零，且不是 decoder 扩容或更长训练独自解释，才值得做 target/framework 方向。若基线之间的 noise 已达 3%，应先扩大估计精度，不以正负一个 seed 做决策。

若普通 readout 已消除差距，走 decoder 问题；若 loss 或 identity control 消除，走 objective/metric repair；若 raw-input probe 没有稳定额外收益，则撤销“遗漏预测信息”的主张；若 stochastic 方法只有 distribution score 收益，就按用户真正需要的 probabilistic forecasting 价值评估，不能倒推 WorldTS 原生 point task 不成立。

## 6. 有限预算救援的边界

仍可保留的最小研究假设是：**对不同未来 functionals 的有限 target capacity 分配不合适，而一个根据 held-out task residual 分配有限 target 坐标的机制，能在固定预算下提高条件预测。** 这是精确、可证伪的假设，也不需要物理/因果语义。但当前不能把它直接提名为新机制：PSIM/HSE-PSR 已有 future test features，OJEPA 已有分因素 capacity/gradient allocation，SC-JEPA 已有多尺度 target，HMM-JEPA 已有 residual sufficiency audit；C21 又已经显示普通 response target 与 compressor 很强。

因此下一次真正的机制候选必须指明：**哪一个被上述 native probe 找出的 forecast functional，在固定 `D` 下被哪一项 objective 舍弃；新分配规则为什么不能被 identity、适当 target loss、OJEPA 或同预算 direct predictor 同样修好。** 可以是有效的已知组件组合，但必须在这个残差上给出实证差异。不能只说“更充分”“history-aware”“multi-scale”或“stochastic”。

本视角不再建议重跑 folding/role-selection toy，也不增加全新 oracle simulator 来制造失败。当前最具体行动是：**对 native frozen checkpoint 完成 §5.1 的 attribution；若没有稳定、对 forecast 有害的缺失信息，停止这条 target-insufficiency 主线。若有，再用 §5.2 的四臂普通修复判断是否还有一个值得做的新框架机制。**

## 7. RQ 回答

- **RQ1：** encoded future outcome 本身不是 history state；但其条件预测可以是 predictive state。WorldTS 的原生目标需要逐损失分析，不能直接套 conditional-mean 或 decoder inverse 攻击。重要的可检验问题是 native finite-budget forecast residual，而不是“state”命名。
- **RQ2：** history-target、conditional future moments、stochastic latent laws、orthogonal allocation、多尺度 codebook、residual tests、trajectory mixtures 均有直接先例；其保证分别受 observability、feature sufficiency、几何或 realizability 条件限制。没有一项自动证明当前问题不存在，也没有留下一个只靠改名就成立的缺口。
- **RQ3：** 用完整 decoder 信息集的 frozen native readout 矩阵先归因，再用同预算 direct observation、loss/EMA、identity-preserving targets 等普通修复做 gate；先做一个小数据集诊断，通过后扩到至少第二类 native 任务。此处是实验设计，未执行。

## 一手资料索引与核查定位

1. [WorldTS](https://arxiv.org/html/2609.31162v1)：§4.2、§4.5，Eq.6–11、18–21；native decoder 实际输入与目标。
2. [HSE-PSR](https://proceedings.mlr.press/r11/boots13a.html)：[PDF](https://raw.githubusercontent.com/mlresearch/r11/main/assets/boots13a/boots13a.pdf)，§3–5；characteristic embedding、core tests、sample representation。
3. [PSIM](https://proceedings.mlr.press/v48/sun16.html)：[作者 PDF](https://homes.cs.washington.edu/~bboots/files/PSIM.pdf)，§3–4、Thm4.1–4.3；sufficient features 与 finite-sample learner 条件。
4. [Ni et al.](https://arxiv.org/html/2401.08898v3)：§3–4，ZP/EZP、Practical objectives、linear non-collapse 条件。
5. [VJEPA 正式论文](https://proceedings.mlr.press/v306/huang26ba.html)：[正式 PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/huang26ba/huang26ba.pdf)，§3、§5、App.E/G；充分性为前提的控制定理与抗坍塌证明。
6. [HMM-JEPA / MCJEPA](https://arxiv.org/html/2608.13621v1)：§6–7；conditional-mean residual、cross-fitting、representation/predictor 归因边界。
7. [Orthogonal JEPA](https://arxiv.org/html/2608.20065v1)：§2、Prop.1；[JEPA-Anything](https://arxiv.org/html/2609.20800v1)：§2.2、Prop.1/Cor.1；orthogonal geometry 的实际保证。
8. [SC-JEPA](https://arxiv.org/html/2602.04643)：§3、App.A.3；soft codebook 和 multi-resolution 的定理条件。
9. [UWM-JEPA](https://arxiv.org/html/2605.25313v1)：§2–3、App.S1；joint spectrum guarantee 及 reduced-readout 边界。
10. [MDN-VJEPA](https://arxiv.org/html/2610.13956v1)：§3–7、§8.6；trajectory-level modes、CK regularization、controlled-experiment 范围。
11. [NE-Dreamer](https://arxiv.org/html/2603.02765v1)：§3.2–3.3；RSSM、history-only temporal predictor 与 observation-embedding target 的并存。
