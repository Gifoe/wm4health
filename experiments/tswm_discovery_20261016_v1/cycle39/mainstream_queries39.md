# C39 主流定向复核：跨查询一致性必须指明是哪一个联合律

2026-10-17。独立 mainstream 视角。冻结问题来自本目录 RESEARCH_BRIEF.md；未读 sibling notes，仅编辑本文件。沿用 deep-research 的原文验证流程；未运行训练/服务器，未下载 weights 或 signal。论文的作者结果不写成本项目复现结果。

## 摘要

概率世界模型至少有四种容易混淆的合同：同原点联合律的边际、同一联合律的条件化、同一潜 Markov 状态的 CK 组合，以及新增历史后重新编码的预测版本。AR/SSM 和固定联合密度可从构造上满足其各自范围内的一致性；NP、flow/inpainting 与近期 CK/self-consistency 微调则提供不同的近似和成本权衡。最近原文已有真实残差：CK 的一致性改善未必改善概率质量；随机条件训练也没有普遍的跨查询保证。但缺少保证不等于严重质量失败。本轮最值得保留的具体未排除项，是**高质量 block joint generator 在任意中途观测查询上实现兼容条件更新的质量—成本前沿**。它尚不是已建立的新方向：简单 canonical-block AR、同一 joint 的条件采样、普通 mask-conditional flow 和已知一致性微调都必须比较。没有定义出超过这些控制的精确机制，因此本视角不接受新方案；给出能改变判断的两阶段证伪合同。

## 1. 冻结问题与取证方法

RQ1：同原点未来联合律、模拟中途后重查、新真实观测条件律，已有方法分别保证什么？RQ2：最近完整方法和普通 AR/SSM/joint density/conditional flow、NP、bridge/inpainting、一致性微调后，是否仍有可测的有限质量/成本残差？RQ3：能否给出最多一个精确不同的机制，并在单 5090、至少两种机制/真实载体上被证伪？

本次先检索 `time series probabilistic world model / joint law / conditioning consistency / projective consistency / CK / forecast revisions`，再沿检索命中核对 MDN-VJEPA、MPNP、From Drift to Coherence、CNP gap、FlowNP/NDP、Sundial、TSFlow、MissPath-FM 的原文与作者实现。主窗口为 2026-04-17..10-17；MPNP、NDP、TSFlow、Sundial、FlowNP 是必要早期先例，另列优先权，不作为本轮发现新原理。检索中的 hierarchical reconciliation 多数只约束同版本的线性加总，不拿来替代跨原点条件更新。未检索到完整方法在本报告全部合同上同时给出无条件保证。

本报告不把 C23–25 的 unknown-system posterior、C14 normalization/cache、C15–16 covariance recursion、C34 first-hit、C37 rank coupling 或 MDN-VJEPA 的轨迹 mode/CK 当成新发现。也不以“RNN 理论上能表达”直接否决有限预算方法。

## 2. 先把检验对象写清楚

令原始已知历史为 \(h\)，同原点未来为 \(Y_{1:H}=(A,B)\)，其中 \(A=Y_{1:k}\)，\(B=Y_{k+1:H}\)。区分：

| 对象 | 精确定义 | 不能替代它的证据 |
|---|---|---|
| 同原点联合律 | \(P_h^H(dA,dB)\) | 各点 quantile 或各 horizon 独立 density 不自动指定 joint；本轮不利用已知 rank-coupling 人造失败 |
| 边际查询兼容 | 较短查询等于 \(P_h^H\) 相应边际 | 只检查均值、只使用相同 seed，不证明整个分布一致 |
| 同一 joint 的条件律 | \(C_h(dB\mid A=a)\)，由 \(P_h^H\) disintegration 得到 | 输入相同已知数值的另一个条件网络，不自动就是这个 conditional |
| 重新编码查询 | \(Q_{h,a}(dB)\)：把 prefix 当历史后重新调用模型 | 与 \(C_h\) 不相等不自动是 bug；必须先说明用户需要此合同、同信息、同坐标/时间、无截断，并证明实际后果 |
| 潜状态 CK | \(K_{s+t}(z,dz')=\int K_s(z,dv)K_t(v,dz')\) | 不包含新的观测似然、Bayes correction 或任意 \(E(h,a)\) 与 \(E(h)\) 的关系 |
| 真实预测版本演化 | 固定实际结局 \(B\)，随真实信息集 \(\mathcal F_t\) 增长而更新的 \(\mu_t=\mathcal L(B\mid\mathcal F_t)\) | “相邻预测不变化”不是正确目标；真正新观测应该改变预测 |

对一个固定的 \(P_h^H\)，若使用它自己的精确条件律，则对任何可积 \(f\)，有

\[
\mathbb E_{A\sim P_h}\mathbb E_{B\sim C_h(\cdot\mid A)}f(B)
=\mathbb E_{B\sim P_h}f(B).
\]

这是 tower identity。换成 \(Q_{h,A}\) 后，它成为需要检验的兼容性要求，不能仅凭概率学定理宣布成立，也不能预先宣告会严重失败。反过来，几个 \(f\) 的 tower moments 通过，只是必要的有限维检查，未证明所有条件、所有 path event 或整个 joint 一致。

这是与已知 NP conditioning 和 CK 原文相容的分类：[CNP gap §3–5](https://arxiv.org/pdf/2604.19312)、[NDP §5、Appendix E](https://proceedings.mlr.press/v202/dutordoir23a/dutordoir23a.pdf)、[MDN-VJEPA §4](https://arxiv.org/html/2610.13956v1)。

## 3. 已有框架到底保证了什么

### 3.1 普通生成构造是必须使用的强控制

1. **固定逐步/固定块 AR。** 若定义
   \(P_h(Y_{1:H})=\prod_j q(Y_j\mid h,Y_{<j})\)，同一个 kernel 从完整历史继续采样就是该 joint 的条件律。固定块 AR 同理，但首先只保证其 canonical 块边界的继续合同。参数学习是否准确是另一问题；重新拟合系数、丢弃历史或改变核不能再援引这个构造结论。
2. **SSM 的同一 generative law。** transition + emission + 固定初始/参数律定义 joint；精确 filter/smoother 与预测满足其概率恒等式。近似 filter 的质量与成本要实际测。冻结 brief 已给出普通 SLR/defensive IS/full Gaussian 的有效历史控制，因此不能把“后验难”重新当作已经成立的失败。
3. **一个固定有限 universe 的 joint density/flow。** 精确 marginal 和 conditional 在数学上兼容；主要问题是可计算性、数值近似及各 query 是否真的使用同一 joint。单独训练许多 \(p_\theta(B\mid A)\) 不能仅凭 conditional 一词获得该保证。
4. **保留原轨迹后继续返回其 suffix。** 若仅要求对自己刚采样的一条路径继续，保存 full sample 即保持同一 joint，几乎没有额外推理成本。它不能提供某个 prefix 下大量独立后续样本，也不能解决外部真实 prefix 的条件化。这一普通缓存对照可防止把无需新学习的“轨迹不断裂”作为研究贡献。

这些是具体构造控制，不是用表达能力包络否决所有新方法。高质量和低成本能否同时达到，仍是经验问题。

### 3.2 NP 系列：交换性、边际一致性和 context 更新各有边界

| 已有框架 | 已验证的保证/训练合同 | 不应扩大的保证 | 可执行状态 |
|---|---|---|---|
| CNP / latent NP | 固定 context 后，factorized/共享 latent decoder 可得到 target marginal compatibility | 把新点 append 后重新跑 context encoder，未必是之前 joint 的 Bayes conditional | 不是本轮新发现；本轮用 CNP gap 原文限定定理范围 |
| MPNP（ICLR 2023） | 用 exchangeable pseudo-context generator 和相应 martingale posterior 表达函数不确定性 | 论文的 pseudo-context/函数任务与条件假设，不能直接变成任意有序真实 TS 的 observation re-encode 定理 | 原文 supplementary 代码；本次未核实独立作者 repo，不声称执行 |
| NDP（ICML 2023） | 原文 §5 特别限定 **target∪context 不变的同一有限 universe**；conditional 从 joint 出发 | 改变 union/query cardinality 不保证 projectivity；实际有限步 REPAINT sampler 也不能等同解析精确 conditional | 官方源码存在；本次读论文与 sampler 描述，未运行 |
| FlowNP（NeurIPS 2025） | set Transformer/CFM，任意 observed context 与 target query；保证其排列交换性 | 作者 §3 明写 marginal 和 conditioning consistency 不是 architecture guarantee，而由随机 context/target 训练推动 | 公开 PyTorch 作者代码，可作为 ordinary amortized conditional baseline |

MPNP 的概率来源是伪样本不确定性；FlowNP 的主要实用增益是并行条件采样与 NFE 调节。两者都不能被缩写成“NP 已经严格解决任意跨查询更新”。另一方面，FlowNP 既然已有可用条件预测和真实 weather 实验，就不能因它没有理论一致性而称其不可用。[MPNP §2.3–3](https://arxiv.org/html/2304.09431v1)、[NDP §3.2/5](https://proceedings.mlr.press/v202/dutordoir23a/dutordoir23a.pdf)、[FlowNP §3、Appendix A](https://arxiv.org/html/2512.23853v1)。

2026 年的 CNP gap 定理只针对 bounded **mean aggregation** encoder、Gaussian decoder、Lipschitz 条件与正方差下界等假设，新增单点的 KL 可按 \(O(n^{-2})\) 衰减；其数值实验是 scalar decoder 构造，不是 Sundial/Transformer 的实训证据。“长 context 所以任意 TSFM gap 可忽略”不成立。更细的解释是：CNP 的固定-context factorization 使同 joint 内新点本来不更新另一 target，该 gap 因而度量 encoder 更新幅度；让 encoder 不响应也可缩小 gap，仍需预测质量控制。[CNP gap Theorem 1–4、§8–9](https://arxiv.org/pdf/2604.19312)。

### 3.3 CK 与 martingale 微调：已有完整先例，不是待发明的普通正则

MDN-VJEPA 已明确 shared trajectory mode 与 distributional direct/composed CK，且有 stop-gradient direct anchor。它比较的是学得的潜 Markov transition 路径，不是观测 prefix 的重新编码/条件化；其理论等式要求真实 Markov transition 与对应 direct law，实际训练目标给近似匹配。原文还明确区分 consistency 与 accuracy，不能把作者已经承认的局限写成揭露。[MDN-VJEPA §4.3–4.4、§7.7、§8.4](https://arxiv.org/html/2610.13956v1)。

From Drift to Coherence 针对**重复同一个 QA 问题**的 answer resampling，用较后 resampling marginal 监督较早 marginal，提出 SC 微调。它不是新真实时间点观测；理论还需要 MC 一致、horizon 覆盖等条件，作者明确实际只训练有限 horizon pairs。把这个 loss 换成时序输出，不能称首次发现 tower/coherence 微调。[ICML 2026 官方身份](https://proceedings.mlr.press/v306/kim26ah.html)、[原文 §4、Appendix C/D](https://arxiv.org/html/2606.17832v1)。

近期 Wilkinson/Ferro 的 fixed-event forecast sequence 理论进一步明确：内部 martingale 之外，还需最后预测与实际 outcome 的校准，才得到其 sequential auto-calibration；单独均值 revision 或内部平稳不足以证实概率质量。这给真实新增观测评估提供了精确标准，而非要求新预测和旧预测保持不变。[Calibrated Probability Forecast Sequences, Proposition 2.4、§7](https://arxiv.org/html/2606.31621v1)。

## 4. 生成、条件化与重编码的可执行底座

### 4.1 Sundial：原点内 prefix 一致性已经由实现提供

本次只读取公开源码与 config，未重新下 weights。HF 当前固定 revision 为 `3212e42564493f520593e5414af4367fc4b49226`，config 的 `input_token_len=16`、`output_token_lens=[720]`。这应由 root 与其既有 pin 再比对，而不是默认覆盖历史固定版本。

源码 `flow_loss.sample(hidden_states, num_samples)` 生成整头，然后按请求截 prefix；公开 128M 配置只有 720。故在同历史/同坐标/同 sampler 下，h96、h192 的共同 prefix 由同一个 joint head 导出，已有 marginal compatibility。原点移动 96 后再编码并生成下一段则使用新 hidden state，是另一个 query；短/长 prefix 相容不能推出它与原 joint 的 conditional 相容。[固定 config](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/config.json)、[modeling_sundial.py L480–497](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/modeling_sundial.py)。

同文件 L438–442 的 RevIN 使用输入历史 mean/std，L496–497 恢复坐标；因此 root 的 growing-history 和 reference mean/std 控制是必要排除项。本报告不预判它们解释或不能解释诊断，也不把 normalization/cache 当新机制。Sundial 官方公开示例本来就允许样本预报，但没有在 README 承诺任意中途 cut 的同-joint conditional API。[作者仓库](https://github.com/thuml/Sundial)、[ICML 2025 原文](https://ise.thss.tsinghua.edu.cn/~mlong/doc/Sundial-ICML25.pdf)。

一个无训练的强普通对照是固定 \(b=8\)：将原 720 头的前 8 点边际作为 canonical transition kernel，每次 append 完整已知/模拟历史再调用。它定义了一个固定 block AR law，且 8 同时整除冻结的 96 和 360 cuts，在这些边界继续遵守自身 joint 的条件合同。它可能需要许多更多的 head calls、也可能改变预测质量；这正是需测的普通质量—成本对照。它不改变 root 的冻结诊断，更不是新方法提案。

### 4.2 Flow/inpainting 可以用，但必须分开 endpoint fitting 与跨 mask compatibility

TSFlow 已有 informed GP prior、conditional prior sampling、guided generation 与直接 conditional training。其 inference-only 条件采样需要假设观测 likelihood，并使用近似 flow integration；因此既不能说“joint 模型不能条件化”，也不能说该实现已经精确解决任何 prefix 的条件律。直接 conditional 版本是更强的普通对照，不能只拿迭代 guidance 的成本与新方法比较。[TSFlow §3.1–3.2、ICLR 2025 官方全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/ee1a1ecc92f35702b5c29dad3dc909ea-Paper-Conference.pdf)。

2026-10-13 的 MissPath-FM 已将 source prior、mask/time-gap context 和 missingness-aware flow path 联合训练，提供公开 PyTorch 实现。源码 `sample` 最后返回 `mask*x_obs+(1-mask)*x`，保证**返回值**在已观测位置相等；这不保证不同 mask 得到的未知部分条件分布来自同一 joint。作者做的是 imputation/forecasting 误差与组件消融，不应替它添加跨 query tower 承诺。[原文 §3–5](https://arxiv.org/html/2610.17715v1)、[model.py L285–293](https://github.com/zgp110/misspath-fm/blob/da6647502630bae0a38d2e4e6bcf06e09bddd6d8/model.py#L285)。

FlowNP 的公开 `models/fnp.py` 直接把 observed context 与 target tokens 放进同一 velocity 模型，允许 query-specific conditional sampling；其源代码给出的 likelihood 与 ODE sampling 接口，适合作为这一类控制。换条件重新生成可以是它的本来任务，而不是某种观察到 re-encode 不等后的“修 bug”。[FlowNP 固定代码](https://github.com/danrsm/flowNP/blob/f3403b6479dad33a7b93ba6edcf93c53d40f8e65/models/fnp.py#L53)。

| 底座 | 本次源码资格 | 能用于什么控制 | 不宣称什么 |
|---|---|---|---|
| Sundial 128M | 固定 HF 源码/config 已读；weights 未下载 | 同 origin joint、canonical block AR、有限 native re-query 诊断 | 本视角未运行；不报告 quality failure |
| FlowNP | 作者 repo 固定 `f3403b6479dad33a7b93ba6edcf93c53d40f8e65` | ordinary amortized conditional flow，NFE—quality 曲线 | 不保证任意 query 一致 |
| MissPath-FM | 作者 repo 固定 `da6647502630bae0a38d2e4e6bcf06e09bddd6d8`，公开训练从零脚本 | mask-conditioned flow + informed prior 的最新普通完整控制 | 末端 clamp 不等于 exact Bayes conditioning |
| MDN-VJEPA | 原文与 5-seed ablation 细节已核对；本次未找到作者 code link | 最近完整 CK/trajectory-mode 先例边界 | 不称本项目执行过，也不称有真实 TS re-encode 失败 |
| From Drift to Coherence | ICML 2026 proceedings 身份 + 原文已核对；未核实作者代码 | 现成 SC/martingale distillation 的新颖性强边界 | QA 结果不外推 TS forecast quality |

## 5. 最新实际失败与仍未排除的残差

### 5.1 已有原文实测：一致性改善与概率质量之间仍有有限权衡

MDN-VJEPA Experiment 2 的 CK 降低某种 direct/composed discrepancy，但 Table 23 的 branch probability MAE、failure ECE 变差；这是作者表内观察，不是推测。其更受控 Experiment 4 用 direct anchor 后改善综合 discrepancy，但 terminal h=16 几乎没有改善，composed Energy Score 的均值也未改善。后两者差异不能在没有 paired significance 的情况下称显著恶化。[MDN-VJEPA Table 23/25/32/33、§8.4](https://arxiv.org/html/2610.13956v1)。

由此真正保留的残差是：在**有限容量、有限优化和有限 sampler budget** 下，怎样让跨计算路径兼容性带来更好的 useful conditional/joint forecasts，而不是只降低模型之间的距离。它不是 CK 新原理；最小普通对照必须包括作者已经有的 anchored CK、等预算继续 proper-score 训练，以及同一 fixed AR law。若普通 anchored/teacher-frozen loss 即能达到质量—成本前沿，便关闭新机制解释。

### 5.2 最新普通组合已有实效，不能把旧组件组合一概否定

MissPath-FM 的作者 ablation 显示 informed prior 单独在随机 missingness 下有效，但在某些 block gaps 下会失败，flow 与 mask-aware path 才补回。这是已知组件交互具有有限预算价值的具体先例；不能以“普通条件网络已能表达”抹掉它。反之，论文主要给 MAE/RMSE 和生成评估，没有做本轮 exact conditioning/tower 检验，不能借其表格证明跨查询问题已普遍解决或已严重失败。[MissPath-FM Table 3、§5.2](https://arxiv.org/html/2610.17715v1)。

原始 NDP Appendix E 的 query-set 变化实验也必须完整解释：作者承认无严格保证，但其观察中多数波动可由 MC 解释。这是对“只要换 query 就必然严重坏”的直接反证。CNP 论文的少 context 理论不能推翻这项实测，更不能覆盖最新大模型。[NDP Appendix E.1](https://proceedings.mlr.press/v202/dutordoir23a/dutordoir23a.pdf)。

### 5.3 本轮最具体、尚未排除的能力残差

**R：高质量 block-joint 模型在同一真实任务上，能否以接近单次 block 采样的成本，在中途新增观测后返回兼容且有质量的 conditional，并保持真实输出相关结构？**

该残差的必要性来自三项不同证据的接合：Sundial 的 joint head 与 re-encoder 确实是不同计算接口；NDP/FlowNP/conditional flow 已提供几类现成解决路径；MDN-VJEPA 已实测一致性和 fidelity 并非同一优化目标。三者尚未给出本轮全部信息合同下的 head-to-requery 质量—成本比较。因此 R 是**待测的前沿缺口**，不是“所有模型都有 bug”的结论，也不是已经证明普通方法失败。

R 与 C23–25 的区别在研究对象：这里首先固定同一个公开 block generator 的输出 joint 与其可用 queries，不推断未知系统参数 posterior；不在旧 latent posterior 配置上再做 IS tuning。新增真实观测质量只在独立真实 held-out suffix 上打 proper scores，自生成 prefix 仅测本模型内兼容性。两层证据缺一不可。

最强普通控制如下，不能只比原模型：

- **构造一致的 AR/SSM**：固定系数 law 作数学正控制；Sundial canonical b=8 wrapper 作同权重的真实成本/质量控制；有训练比较时再加 capacity/time-matched AR-MDN 或 AR-flow。
- **同 joint 条件化**：低维可解析 full Gaussian/GMM 或既有有效 SLR/IS 控制用于资格与近似误差；高维若使用 learned conditional sampler，要明确只是近似，不能把它标作 exact teacher。
- **完整普通 conditional 训练**：FlowNP 或普通 random-mask CFM；需要对其 uncertainty/joint quality 评分，不能只比 point MAE。
- **已知 coherence 训练**：同容量、同训练样本下 anchored distributional CK/SC，与等预算普通 proper-score 继续训练、self-generated-prefix data augmentation 分开。若新 loss 正是 conditional MLE，明确归为普通基线，不复活 C34。

## 6. 能改变判断的单 5090 证伪合同

### 6.1 当前不接受一个新方案

本视角没有精确到足以绕过上述控制的新机制，所以 accepted candidate=0。这个状态不是检索任务的自动成功；真正产出是 R 和下列能使判断改变的实验。`joint head + history re-encode + consistency loss + proper score anchor` 本身已接近已知 SC/CK/conditional augmentation 的组合描述。有限预算交互仍可以成为框架贡献，但必须先出现它超出普通控制的能力证据，不能仅换术语。

### 6.2 第一阶段：只建立或关闭 native residual

完全沿用 root 冻结诊断，不改 cuts 或采样资格。固定 weights、full-growing history；native/controlled sampler 先过 same-noise replay，分别报告 native normalization 和 reference-coordinate 结果。h96+96 为主、h360+360 为预定敏感性；按 outer path 分组 MC，保留 finite-sample null 与固定 AR 正控制。

这里的直接可观察量是单次 joint 后段与重查询混合后的 tower moments/样本距离。若置信区间覆盖 null，或坐标/正确 API 解释差异，停止该动机；若不解释，只记 self-consistency residual，不写 forecast quality degradation。也不把显著 mean gap 直接升级为整条条件律失败。

加入两个几乎不需要新研究成本的解释控制：保留原 sample suffix；canonical b=8 wrapper。前者说明单轨迹 continuation 无须新模型，后者说明兼容 queries 可以用更多调用构造。若用户任务只需要前者，则 R 没有任务价值；若后者已在可接受成本达到足够质量，则 R 被普通方法关闭。

### 6.3 第二阶段：只有 residual 存在后才测实际质量—成本

需至少两种现实机制，推荐使用**公开 Electricity 的季节负载/聚合能量**与**公开 Weather/ETT 的局部状态变化/连续 ramp**两类载体；它们只是建议，不替代 root 已有合法数据选择，不本轮下载。一个预测任务内固定 origin、真实新增 prefix 和真实 suffix，所有方法接收相同数值/协变量与时间信息。不用“模拟 prefix 误差”代替真实 forecast quality。

预先只选两类 path functional：未来 block sum/mean 与连续 ramp/阈值持续事件。先给这些 functional 的 CRPS/energy/Brier 等 proper scores，再给 raw path Energy Score 与相关结构敏感的 variogram score；不靠逐点 marginal CRPS 一项断言 joint 有用。阈值、持续长度与 functional 权重在 train/validation 固定。若仅改善一个 coherence 指标但 proper scores 无收益，关闭；若只是对错误信息更钝，不能算条件更新能力。

用同一 5090 记录 sample 数、NFE、head calls、wall-clock、峰值内存，画实际 quality—cost 曲线；canonical AR 的额外调用要算，conditional flow 的训练/采样成本也要算。第一次仅两 seeds 的 bounded screen，配对 block bootstrap 以 origin/series 为单位；两类载体均不过 screen 就停止，不扩大超参搜索。只有两载体都超出 anchored CK、ordinary conditional augmentation 和 canonical AR 的实际前沿，才值得定义一个具体机制并增加独立 seeds。

此阶段必须在比较前确定“快多少/非劣质量多少”的工程阈值；本次未跑 timed baseline，不能从论文 NFE 或显卡型号捏造分钟数。若无 exact conditional teacher，不报告 exact KL，只报告可复核的 sampled/proper-score 与近似采样误差控制。

## 7. 对三个 RQ 的结论

**RQ1：** 同一 fixed joint 的 marginal/conditional、fixed AR/SSM 的 generative continuation 可以有构造保证；MDN-VJEPA 的 CK 是潜 Markov 路径兼容，NP/FlowNP 多为固定 context/set 的不同保证，NDP 同-universe 与换 universe 必须分开。Sundial 同原点 prefix 兼容已存在；它不等于任意中途 re-encode conditional。真实新增观测的版本演化也不应被要求静止。

**RQ2：** 已有真实有限 tradeoff，尤其 CK discrepancy 与 predictive fidelity 可分离；但缺少保证不证明严重失败。R 是尚未排除的具体质量—成本残差；目前没有完整同合同实验表明强普通控制不能达到它。经典表达能力也不能替代这个实测。

**RQ3：** 当前新方案 0。最有决策价值的是先完成冻结 native diagnosis，再比较真实 prefix 条件更新的 proper-score—cost 前沿，纳入 canonical block AR、完整 conditional flow 和 anchored SC/CK。若只发现 re-encode 差异而没有真实质量或成本后果，就不足以推进新框架；若新组件组织能在两种实际机制上跨过普通前沿，则允许其作为有限预算交互贡献，而不要求全新数学算子。

## 参考文献与定位

[1] Yongchao Huang. “MDN-VJEPA: Multimodal Probabilistic Joint-Embedding World Models.” arXiv:2610.13956, 2026。[身份](https://arxiv.org/abs/2610.13956)、[全文](https://arxiv.org/html/2610.13956v1)。定位 §4、§7.7、§8.4–8.6、Tables 23/25/32/33。检索 metadata 日期与 arXiv identifier 月份有不直观之处，因此不据搜索卡片单独确定优先日；其截至本日可访问全文及具体主张已核对。

[2] Songeun Kim, Seungyoo Lee, Edwin Fong, et al. “From Drift to Coherence: Stabilizing Beliefs in LLMs.” ICML, PMLR 306, 2026。[官方 proceedings](https://proceedings.mlr.press/v306/kim26ah.html)、[全文](https://arxiv.org/html/2606.17832v1)。定位 §4、Appendix C/D；未扩大其 repeated-QA 的合同。

[3] Robin Young. “On the Conditioning Consistency Gap in Conditional Neural Processes.” TMLR, 04/2026；arXiv:2604.19312。[身份](https://arxiv.org/abs/2604.19312)、[全文 PDF](https://arxiv.org/pdf/2604.19312)。定位 §5–9。arXiv HTML 与直接 OpenReview 本次不可用，但 PDF 已读。

[4] Thomas Wilkinson, Christopher Ferro. “Calibrated Probability Forecast Sequences and Measure-Valued Martingales.” arXiv:2606.31621, 2026。[身份](https://arxiv.org/abs/2606.31621)、[全文](https://arxiv.org/html/2606.31621v1)。定位 Definition 2.1/2.2、Proposition 2.4、§7。

[5] Hyungi Lee, Eunggu Yun, Giung Nam, et al. “Martingale Posterior Neural Processes.” ICLR, 2023。[全文](https://arxiv.org/html/2304.09431v1)、[官方 OpenReview PDF](https://openreview.net/pdf?id=-9PVqZ-IR_)。定位 §2.3–3、Appendix C。已知先例，不是 C39 新发现。

[6] Vincent Dutordoir, Alan Saul, Zoubin Ghahramani, Fergus Simpson. “Neural Diffusion Processes.” ICML, 2023。[官方全文](https://proceedings.mlr.press/v202/dutordoir23a/dutordoir23a.pdf)、[作者代码](https://github.com/vdutor/neural-diffusion-processes)。定位 §3.2、§5、Appendix E。

[7] Hussen Abu Hamad, Dan Rosenbaum. “Flow Matching Neural Processes.” NeurIPS, 2025。[官方身份](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a92519f525c00085095fa41c5c46cdb5-Abstract-Conference.html)、[全文](https://arxiv.org/html/2512.23853v1)、[代码](https://github.com/danrsm/flowNP)。定位 §3、Appendix A；代码 pin 见 §4。

[8] Marcel Kollovieh, Marten Lienen, David Lüdke, et al. “Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting.” ICLR, 2025。[官方作者页](https://www.cs.cit.tum.de/daml/tsflow/)、[官方全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/ee1a1ecc92f35702b5c29dad3dc909ea-Paper-Conference.pdf)。定位 §3.1–3.2。未将搜索结果中误标 ICML 2026 的二手引文作为年份依据。

[9] Yong Liu, Guo Qin, Zhiyuan Shi, et al. “Sundial: A Family of Highly Capable Time Series Foundation Models.” ICML, 2025。[作者全文](https://ise.thss.tsinghua.edu.cn/~mlong/doc/Sundial-ICML25.pdf)、[作者代码](https://github.com/thuml/Sundial)、[HF 固定源码/config](https://huggingface.co/thuml/sundial-base-128m/tree/3212e42564493f520593e5414af4367fc4b49226)。本轮没有下载或运行 weights。

[10] Genpei Zhang. “MissPath-FM: Flow Matching with Structured Priors for Partially Observed Time Series.” arXiv:2610.17715, 2026；作者项目页标 NeurIPS 2026。[原文](https://arxiv.org/html/2610.17715v1)、[作者项目页](https://zgp110.github.io/misspath-fm/)、[代码](https://github.com/zgp110/misspath-fm)。定位 §3–5、Table 3；本次 pin 见 §4。不以作者项目页单独证明官方会议录状态。

补查并未据此新增主张：Kumar, Nagaraj, Roy, Sarkar, “Conditional Flow Matching for Transport Between Markov Processes,” arXiv:2610.07229, 2026，[全文](https://arxiv.org/html/2610.07229v1)。其总体保证针对同质 Markov transport、coverage/contraction 等条件，不是通用 observation-history conditional compatibility；不作为 R 已解决或已失败的证据。
