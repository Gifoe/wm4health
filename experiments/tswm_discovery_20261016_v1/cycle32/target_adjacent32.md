# C32 独立相邻领域核查：预测状态、目标几何与 surrogate compatibility

2026-10-17。范围来自 `cycle32/RESEARCH_BRIEF.md`。本报告为 deep-research 的 adjacent-field perspective；不运行训练、不连接 SSH、不删除文件。只使用公共通用关键词检索和本地已下载 upstream 源码，不把私有研究描述提交到搜索服务。

## 摘要与证据状态

PSR、system identification、history abstraction 和 multitask representation 均不支持把“预测未来编码”直接等同于“已得到可复用的系统状态”。它们要求给出保留了什么统计量、面向什么任务、经过什么预测损失及 decoder link 后仍然可用。反过来，WorldTS 是给定观测历史和可用 covariates 的有限时距预测器；不能用“没有恢复全部物理状态”“不能无限滚动”否定它。

本轮可保留的最窄救援是 **learned target、surrogate loss、decoded task 三者的兼容条件**。它不同于 C11 的分布 folding，也不依赖 C21 的不可预测 driver 或小瓶颈。然而，理论基础属于已有 calibrated surrogate / loss embedding；raw-coordinate anchor、匹配观测损失和 target-only regularization 都是强普通修复。目前没有本轮 native-trained 失败证据，也没有超过普通修复的机制收益，故**立项推荐暂为 NONE；保留一个明确的实验门槛，不把反例当新方法。**

本轮证据分为：

| 类型 | 本轮实际完成 | 能支持什么 |
|---|---|---|
| 文献核查 | arXiv / PMLR / 作者论文及官方代码说明 | 原方法、精确 overlap 和定理假设 |
| 源码审计 | WorldTS pin `269426ea6217d7152a244da0e209d922acd46d4f`；LatentTSF 本地 upstream | 输入、目标、loss、decoder 训练接口 |
| 数学推导 | 下文 population L1+cos 反例及全局最优证书 | 固定编码与 loss 的兼容性反例 |
| 实际执行 | 无训练、无新数据实验；下文数值为代数计算 | 不支持 native 模型发生该失败或新方法有效 |

## 1. 先按原生代码排除不成立的动机

WorldTS future patch target 会拼接最后 `state_patch_len-1` 个历史点；stage 2 冻结 predictor，并同时用预测 latent 与 target latent 拟合 decoder。因此，“目标完全没有历史”和“decoder 只在 oracle target 上训练”都不是其一般事实。论文 §4.2、§4.5 与源码一致。[WorldTS paper](https://arxiv.org/html/2609.31162v1)

本地精确证据：

- `upstream/WorldTS/TimeMMD-bench/models/worldts/models/worldts_model.py:231` 的 `encode_future_target`；`:328` 的 `configure_for_stage2`。
- `upstream/WorldTS/TimeMMD-bench/exp/exp_worldts_forecasting.py:172` 的 `_stage2_batch_loss`：预测解码 loss 加 `0.3` target 重建 loss。
- `upstream/WorldTS/DAG-bench/scripts/w_future/worldts.sh:25` 等 numeric 原生配置使用 `latent_alignment_loss=mae`、latent 权重 10、cosine 权重 15、VICReg 权重 0.1，stage 2 observation loss 为 MAE。
- `upstream/WorldTS/TimeMMD-bench/models/worldts/losses/latent_state_loss.py:42` 同时提供 MSE/MAE 等分支；`:67` 对 predictor 与 target 两边施加 variance/covariance regularization。
- 文件名为 `sigreg_loss.py`，但实现为 variance-floor + off-diagonal covariance，即论文所述 VICReg，不是特征函数 Gaussianity test。不能因文件名把它当 LeJEPA 的完整 SIGReg。
- `upstream/LatentTSF/my_utils.py:500` 明确 native latent MSE 权重 10、cosine 15，decoded observation MSE `perceptual_weight=0` 为默认；`my_train.py:524` 后实现此路径。普通 observation-loss 修复本来就在代码接口中。[LatentTSF official repository](https://github.com/Muyiiiii/LatentTSF)

所以不能把 WorldTS 全部写成 `D(E[phi(Y)|H])`。只有去掉 cosine、预测分支正则并选纯 squared latent loss 后，population 最优量才确实是 `E[phi(Y)|H]`。native MAE 对应的量更接近坐标中位数，而 cosine 和 batch regularizer 进一步改变目标。

## 2. 相邻领域到底已经解决了什么

### 2.1 PSR 与 system identification：状态是预测分布的统计量，不是未来样本编码本身

PSTD 用 history-to-future reduced-rank regression 寻找保留 predictive information 的压缩，并与 PSR subspace identification 联系起来；Predictive-State Decoders 用未来观测统计量直接约束 RNN hidden state。两者已覆盖“给隐藏状态增加未来预测监督”与“按可预测内容压缩”的宽泛提案。[PSTD author PDF](https://homes.cs.washington.edu/~bboots/files/pstd_arxiv.pdf), [Predictive-State Decoders](https://arxiv.org/abs/1709.08520)

PSR 观点允许 compact tests/features，但需要这些 features 对所声称的任务构成充分统计；对任意有限 horizon 预测任务，并不要求它们恢复全部物理状态。一个未来样本的可逆编码也不自动等于该未来**条件分布**的充分统计。

DPAD 已把预测相关与其他动力学分成不同状态并优先学习前者；PSID 的 filtering/smoothing 扩展进一步处理两条时间序列间的状态估计。因此，“用 task-relevant state 替代 reconstruction-dominant state”与“用未来 smoothing teacher 改进状态”都不能只改名后立项。[DPAD primary paper](https://www.nature.com/articles/s41593-024-01731-2), [PSID filtering/smoothing](https://arxiv.org/abs/2507.15288)

### 2.2 历史状态与近期 JEPA：history、multi-horizon、随机 latent 都已有直接先例

| 已核实一手工作 | 日期与状态 | 精确覆盖范围 | 不能过度外推的部分 |
|---|---|---|---|
| [Ni et al., self-predictive RL](https://arxiv.org/abs/2401.08898) | ICLR 2024 | 联系统一 state/history self-predictive abstraction，讨论 stop-gradient 和目标 | RL 的 task preservation 条件不能免证明移植为通用 TS 充分性 |
| [CF-JEPA](https://arxiv.org/abs/2606.07031) | 2026-06-05 preprint | 多时距 forward future representations；online/EMA encoder 按任务使用 | 并未证明任意未来编码对任意 decoded loss 兼容 |
| [HEPA](https://arxiv.org/html/2605.11130v4) | 2026-05-11，v4 06-03；ICML FMSD workshop | horizon-conditioned future interval 表征；冻结 encoder 后学 event CDF | 信息界假设事件信息由 target 表征中介，不能据其界推出未知 target 自动满足此前提 |
| [VJEPA](https://proceedings.mlr.press/v306/huang26ba.html) | ICML 2026 正式 PMLR | 预测 latent 分布、variational objective、PSR/Bayesian filtering 联系 | sufficient information state 的结论仍以保留 task-relevant predictive information 为条件 |
| [Sub-JEPA](https://arxiv.org/abs/2605.09241) | 2026-05-10 preprint | 在随机子空间施加 Gaussian constraints，缓解全空间 regularization 偏置 | 换 regularizer 不能独占“几何与预测任务可能不匹配” |

这几条在 2026-04-17 至 2026-10-17 窗口内，已足以排除“加 multi-horizon/历史/概率 encoder/子空间正则即可成为新框架”的宽泛方案。按数据集训练没有问题；上述比较也不以共享 checkpoint 零样本为条件。

### 2.3 状态可识别性与预测任务兼容性是两个不同问题

[When Does LeJEPA Learn a World Model?](https://arxiv.org/html/2605.26379v1)（2026-05-25）的精确定理讨论 matched dimension、Gaussian OU positive pairs 和直接 alignment `||h(z')-h(z)||²`；不是任意 nonlinear predictor、native MAE+cos 或任意真实 TS 分布下的识别定理。不能用其 abstract 中的宽泛措辞给 WorldTS 提供自动担保，也不能因原生任务不满足此定理就断言预测失败。

calibrated surrogate 文献已经研究“surrogate report + decoding link 是否恢复目标任务最优动作”。其中 quadratic surrogate 采用 feature embedding，并要求它与原 loss 的 embedding/decomposition 兼容。把该原则应用到 TS learned targets 是可用诊断角度，但理论原则不是本轮首创。[Nowak-Vila, Bach and Rudi, 2019](https://arxiv.org/html/1902.01958v2)

## 3. 最小救援：把“预测充分”缩到可检验的兼容条件

设 `H` 是全部允许历史与 covariates，`U` 是 decoder 另外能看到的历史 patch、RevIN stats 等，`P` 是固定 `U` 后的可达未来条件分布。固定编码 `phi` 与 latent surrogate `s` 定义

\[
T_\phi(P)=\arg\min_r E_P[s(r,\phi(Y))],\qquad
A_\ell(P)=\arg\min_a E_P[\ell(a,Y)].
\]

所需条件是存在 decoder `D`，使对所有可达 `P` 和 `r∈T_phi(P)` 都有

\[
D(r,U)\in A_\ell(P).
\]

若两个条件分布共享 `U`，它们的 latent Bayes set 有共同 report，而 raw Bayes actions 不相容，则后续再大 decoder 也无法修复该 report 的歧义。必须把 `U` 纳入；若 decoder 仍能直接看到区分两者的完整历史，反例就不适用。

对于**纯 squared latent → raw MSE**，在任意有限支撑条件分布族上，distribution-free mean recovery 等价于 raw output 在 `{1,phi_1,...,phi_d}` 的 affine span 中（适当可积性下）。充分性来自 `Y=A phi(Y)+b` 与期望线性；必要性来自同 feature expectations 的混合分布必须有同 raw expectation。若仅讨论真实数据可达的狭窄分布族，该 affine 条件过强，只需要该族上的 fibre compatibility。

若 `Y=A phi(Y)+b+eps(Y)`，则 squared-latent 的 oracle mean decoder 误差满足

\[
E\|E[\epsilon(Y)\mid H]\|^2\le E\|\epsilon(Y)\|^2.
\]

因此 native learned encoder 若已有很低的 held-out linear reconstruction error，纯 squared→MSE 的几何缺口就被限制得很小。反之，linear reconstruction 差也不是现实条件均值 collision 的证明。

**这个 affine 条件不适用于 native L1 surrogate。** 坐标中位数不与任意 affine decoder 交换，下面反例甚至具有完美 affine 单样本重建。

## 4. 独立解析证书：可逆、线性可重建的目标仍可被 L1+cos 合并

这是一项固定编码、固定两个条件分布的数学诊断；不是 learned WorldTS 训练失败，也不是由真实数据检出的失败。不要与任何执行结果混写。

令 `Y∈{-1,0,1}`，`M=100`，

\[
\phi(-1)=(M,M),\quad\phi(0)=(M+1,M),\quad\phi(1)=(M,M+1).
\]

这是一到二维的展开，没有 latent bottleneck；三点仿射独立，边际 covariance full rank，而且

\[
Y=\phi_1(Y)+2\phi_2(Y)-3M-1.
\]

两种条件分布为

| context | `P(Y=-1)` | `P(Y=0)` | `P(Y=1)` | 唯一 raw MAE median |
|---|---:|---:|---:|---:|
| A | 0.45 | 0.40 | 0.15 | 0 |
| B | 0.60 | 0.10 | 0.30 | -1 |

令 `m=(M,M)`，原生数值权重对应的单步 surrogate 为

\[
R_P(r)=10 E_P\!\left[\frac{|r_1-\phi_1(Y)|+|r_2-\phi_2(Y)|}{2}\right]
+15 E_P[1-\cos(r,\phi(Y))].
\]

本节首先不包含 VICReg。两种条件分布的唯一全局最优预测都是 `r=m`，证明如下。

1. 对 A，第一坐标上高于 `M` 的概率为 `.4`，第二为 `.15`。加权 L1 risk excess 至少为 `1*||r-m||_1`。对 B，相应概率为 `.1`、`.3`，excess 至少为 `2*||r-m||_1`。这来自 Bernoulli 两点 absolute-risk 的线性 median margin，且对区间外的 `r` 也成立。
2. cosine 部分的总取值在 `[0,30]`。任何比 `m` 好的候选必须落在 `||r-m||_1≤30`（A）或 `≤15`（B）。
3. 在较大的 30-ball 及连到 `m` 的线段上，`||r||_2≥sqrt(2)*100-30>111.42`。因为 `||∇_r cos(r,z)||_2≤1/||r||_2`，weighted expected cosine risk 的 L1-Lipschitz constant 最多 `15/111.42<0.135`。
4. L1 的最小增长率为 1，严格大于可能的 cosine 改善率 0.135。因此 ball 内除 `m` 外全都更差；ball 外已由第 2 步排除。两个 law 的唯一全局最优均为 `m`。

也可用更保守的单一半径 100 证书：ball 外 L1 改善不可能被至多 30 的 cosine 改善抵消；ball 内 cosine Lipschitz 常数不超过 `15/((sqrt(2)-1)*100)<.363`，仍小于 1。

若 contexts 等概率，decoder 只能看 `m` 且另有 side information `U` 相同，那么最优公共 raw MAE decoder 的 risk 为 `.70`，分别解码两个 context 的 Bayes risk 为 `(.60+.70)/2=.65`，不可避免 excess `.05`。它是 surrogate Bayes report 的 collision，既不是 phi 的 folding，也不是 phi 没编码 driver。

### 普通修复在这个反例上足够

- **raw-coordinate anchor**：让 target 含有 `Y` 的受保护坐标，以原观测任务匹配的 loss 训练该坐标；cosine/regularizer 只作用于 auxiliary 坐标。此时 raw median 可直接读取。若 anchor 也受 cosine 耦合，不能不检查就宣称保证；需要单独的 margin 证书。
- **纯 target MSE + 重拟合 raw-MAE decoder**：这里 `E[phi(Y)]=(M+p_0,M+p_1)` 恰好保留三类的完整概率，故可判断正确 median。注意 affine decoder读出的是 raw mean；若评价 MAE，应重拟合适当 decoder，不把 mean 当 median。
- **decoded observation loss**：直接匹配被评估的 observation loss，或与 latent loss 混合并对 predictor 回传梯度。这是既有普通修复；LatentTSF 已有接口，WorldTS 已有 observation-space variant。

VICReg、EMA、历史前缀、不同 target 训练和实际 decoder 额外输入均须另行检验。固定编码反例不能推出整个自由 joint-training objective 的所有最优编码都失败。本轮不把 regularized 神经拟合放进上述解析证书。

## 5. 是否能把救援变成一个通用框架机会

可以提出可证伪的工程问题：**是否能在保持 learned latent 对有限预算有益的同时，强制目标与 native loss 的统计答案兼容，并用同一机制跨 forecaster/数据集获得收益？** 不要求 classical methods 在无限算力下失败。

但目前能直接构造的机制就是 protected raw anchor / loss-matched feature bank / 普通 mixed observation loss。若写成“task-compatible predictive-state world model”，只是把 calibration、PSR 和已有 skip loss 组合后命名；尚无支持其超过普通组合的证据。若未来收益只来自减少 cosine 权重、移除 predictor variance floor、换 decoder 或增加训练时间，应归入原模型优化与 baseline qualification。

值得检查的 narrow residual 必须同时满足：

1. 原生 target 确实产生与 native decoded loss 相关的损失，而非只有人为反例。
2. 更宽/更充分训练的 decoder 不能抹掉它；输入和 native side information 正确。
3. 参数与总预算匹配的 observation-only、mixed observation loss、weight tuning、target-only regularizer 是强 comparator。
4. 一个具体兼容机制在至少两个任务族或两种 backbone 上仍有一致收益，而非只在造出的 support 上赢。

## 6. 单 5090 最小裁决：先解析 witness，再 native 残差门槛

以下是建议协议，尚未执行。解析 witness 不需要 GPU；GPU 只用于必要的 native residual qualification。

### 6.1 最小数学执行：40 行有理数 dataset

两个 context 各 20 行，A 为 `(-1,0,1)=(9,8,3)`，B 为 `(12,2,6)`。使用精确 target `phi`，直接计算 native `F.l1_loss` 的 mean reduction 与 `F.cosine_similarity`；同时打印上面的解析证书参数。独立网格/求解器仅作实现核对，不充当全局最优证明。

报告四个量：`phi` 的 injectivity 与 covariance rank；两 context 原生 surrogate 最优 report；公共 decoder 与 context decoder 的 raw MAE；普通 anchor / pure-MSE+refit decoder 修复。native regularizer、维度扩展、float64/float32 可作为普通诊断，但必须与无正则解析命题分开。

**这一步的成功只确认 compatibility 问题定义正确。不能据此声称发现实际模型故障，更不能开启大规模合成训练。**

### 6.2 有限预算 native qualification

若确需进入训练，预注册两个 native 数据任务：WorldTS numeric `NP.csv` 的 h=24，以及 LatentTSF `ETTm1` 的 h=96；两者各自用原生输入和 split、允许按数据集训练。优先使用现成 native checkpoint；是否已有可用 checkpoint 本轮未核查。每个协议 3 seeds，同 backbone、history、available future covariates、target dimension、总训练预算。不要把未来不可见条件交给方法或 comparator。

| arm / probe | 改什么 | 解决哪个混淆 |
|---|---|---|
| Native | 原样协议，含其 target loss 与 decoder 训练 | 真实基线 |
| Decoder refit | 冻结同一 predictor；去掉混合 oracle-target reconstruction 项，decoder 单独对预测 latent 拟合 native observation loss | decoder 训练分布与混合权重 |
| Ordinary repair envelope | 同等 validation trial/训练预算选择 observation-only 或 latent+observation；cosine weight 包含 0，预测分支 regularizer 可关闭、target-side 保留 | target 权重、优化与已知修复 |
| Protected anchor | 在同 latent dimension 内保留 raw target coordinate；其 loss 匹配评价 task，auxiliary 保留 learned target；不能另加原始输入信息 | 最窄兼容机制是否有有限预算增益 |
| Capacity diagnostic | 只对仍有明显 residual 的冻结目标扩大 predictor/decoder，并实际确认优化收敛；作为单列诊断，不冒充同预算主结果 | 有限容量与优化未达成 |

主比较必须先指定 MAE 或 MSE 的 native 任务目标；另一个作为 secondary metric。不能训练 median 后以 mean metric 的差异伪装 latent 独有问题。保存 normalization stats 和 decoder history-prefix 的实际输入，排除 side-information 不一致。

为定位而非只看排名，应保存 oracle-target reconstruction、frozen linear/nonlinear reconstruction、latent validation risk、decoded native risk、学习曲线与 wall-clock。对 squared-loss 分支可利用前述 affine residual 上界；对于 native L1 分支不能用 affine reconstruction 通过就判为兼容。

**裁决规则**：

- decoder refit 抹掉差距：decoder 问题；不立新 target 方向。
- 普通 repair envelope 抹掉差距：已知 loss/regularization 修复；不立新框架。
- capacity/收敛诊断抹掉差距：不能声称信息不足；有限预算方法仍可研究，但须改成清楚的效率收益，并与 envelope 比较。
- anchor 在两 native 协议上稳定超过 envelope 且总预算不多：才进入下一轮最小 mechanism study。预注册实用门槛可为平均 native loss 至少改善 3%，每协议跨 seed 同向；这是项目决策门槛，不是显著性定理。
- 只在 40-row witness 成功、native 无残差：结束此线，保留诊断命题。

真实时间序列通常每条历史只有一个未来样本；仅靠普通 held-out regression 无法精确识别每个 `P(Y|H)`。因此 native 排名与探针能淘汰很多解释，却不能单独证明 population Bayes-report collision。若要作此强主张，之后还须有与真实任务相连、可重复条件采样且可核对 oracle 的模型；不得把人为两维瓶颈或同一个 Gaussian-fold toy 再包装。

## 7. 对三个 RQ 的回答与最终建议

RQ1：future observation encoding 是一个 target feature map；经过 native surrogate 后，预测的是它所诱导的 Bayes report，而非自动获得 PSR。WorldTS 已包含历史前缀、covariates 与预测 latent decoder refit，不能假设这些不存在。本轮只建立固定编码的 loss compatibility 反例，未建立原生 learned-model failure。

RQ2：PSR/PSD、self-predictive history abstraction、近期 multi-horizon/probabilistic JEPA 和 task-prioritized system identification 已覆盖宽泛机制。剩余可检验角度是具体 target-loss-decoder compatibility；普通 observation loss、protected anchor、target-only regularization 是必须赢过的强修复。

RQ3：先用 40-row exact witness 核实问题，然后只在需要时进行两个 native 协议的有界 qualification；用 decoder refit、ordinary repair envelope 和 conditional capacity diagnostic 区分混淆。无需零样本、动作、causal online adaptation 或 classical infinite-compute failure。

**当前建议：NONE，附带一次可执行救援门槛。** 能改变判断的不是更多“sufficiency”措辞，而是 native 任务上仍存在、且 protected/matched target 机制在相同预算下超过普通修复的残差证据。
