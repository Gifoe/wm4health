# C39 独立 critic：跨查询一致性必须测联合依赖，而不只是后半段边际矩

## 摘要

本报告按冻结 C39 brief 核查跨查询概率律和测量合同。最可能改变 root 实验判断的反证只有一个：**中途重询完全忽略生成前缀，仍可能通过后缀所有边际矩的 tower 检查。**因此需要在同一 outer prefix 上加入 prefix-weighted probes，才能筛查前后段依赖。该修复是标准 conditional moment restriction 思路，不是新方法。即便检出差异，它也只说明旧 joint law 与 re-encode kernel 不兼容；旧 joint 若错设，中途版本反而可能更准确。本轮没有建立一个新状态／转移学习候选。已有 joint AR/SSM、真正从固定 joint 条件化的方法在各自合同内解决概率代数；masked conditional flow、bridge 近似和 coherence fine-tuning 则不能仅凭名称推成任意 history 重编码的精确保证。

## 1. 冻结问题与研究范围

本报告回答：已有方法对同原点 joint、模拟中途重询和真实新增观测条件律分别保证什么；有限质量／成本残差如何与 ordinary repair 区分；是否有足够证据启动一个最小新机制。研究仅使用冻结 brief、parent 的执行前协议补充，以及公开一手论文／作者实现入口；不读取 sibling notes，不训练、不访问服务器、不下载权重或大数据。

主窗口为 2026-04-17 至 2026-10-17；必要基础来源单列。检索先用 `conditioning consistency / predictive martingale / flow neural processes`，再收窄到 `Sundial multi-patch / conditional moment restriction / conditional flow / probabilistic forecast dependence scoring`。确认标题身份后，具体主张回查原文位置。不把 C23–25 已关闭的 unknown-system posterior 或 defensive IS 重新列为候选；也不把 C14 normalization/cache reuse 或已有 CK 正则改名。

## 2. 先把四种“同一概率律”分开

固定原始 history 和可用信息为 h，原点 t 的 joint 为

\[
P_h(dU,dV),\quad U=X_{t+1:t+a},\quad V=X_{t+a+1:t+a+b}.
\]

将 U 作为数值追加后重新调用模型，得到一个 kernel

\[
Q_h(dV\mid U=u):=F_\theta(h\mathbin{\Vert}u;\text{target timestamps }t+a+1:t+a+b).
\]

这只是重新调用接口的定义，不能先写成 `P_h(dV|U)`。跨查询的精确要求是

\[
P_h(dU,dV)=P_h^U(dU)Q_h(dV\mid U).\tag{1}
\]

只要求同一 latent 上 Chapman–Kolmogorov，或只要求短请求是长请求的 prefix marginal，均不能代替式 (1)。Sundial 的短请求截取固定长预测，与 FlowNP 的不同 context/target set 重新求值，正好说明这几种接口不能混称。[1,3]

| 合同 | 可以保证什么 | 没有自动保证什么 |
|---|---|---|
| 一个固定 joint 的边际／条件化 | 若所有计算确实来自同一个归一化 joint，概率链式法则成立；模型是否拟合真过程不影响这点 | 不同 history encoder 或不同查询维度输出属于同一个 joint |
| 同一 latent Markov kernel 的 CK | 沿同一状态、同一转移 kernel 的组合律 | 先 decode，再把观测 history 重新 encode，是否恢复相同过滤／预测状态 |
| 任意 mask/context 的 amortized conditional model | 可直接学习不同条件任务，减少逐次推断成本 | 有限训练后所有 mask/context 彼此兼容 |
| 同一绝对未来目标的 forecast versions | 正确 filtration 下，理想预测的条件期望满足 tower | 未来状态本身必须是 martingale，或不同目标时刻的预测必须相同 |

这里的 flow/denoising time 也不等于真实 forecast time。生成 ODE 在 noise time 的组合性质，不能直接充当时间序列未来轨迹的 CK 保证。

真实新增观测 u 若来自外部真实过程 P*，其频率不必等于模型的 P_h^U。即使 Q 是 P_h 的正确条件律，`E_{P*}[E_Q g(V)|h]` 也不必等于模型旧边际的 `E_{P_h}g(V)`。因此 root 用模型自己生成的 U 做内部测试是正确合同；不能拿真实前缀的经验频率与旧模型边际做 tower 检查，再把错设造成的偏差认作概率代数错误。

## 3. 近期完整先例解决到哪一层

下表的结论是：精确概率代数已有普通解法；“无须再训练就能从任何 observation history 高效重编码且保持质量”不是这些来源共同建立的结论。

| 先例 | 实际机制／保证 | 对 C39 的限定 |
|---|---|---|
| Sundial | TimeFlow 以 lookback representation 为条件生成 multi-patch；Base 配置 patch=16，预测 head 支持 16/720，短任务截取 720 输出 [1] | 同原点 prefix 请求可以天然共用一个 720 law；中途重新编码是另一个条件接口，原文没有给出式 (1) 定理 |
| Conditioning Consistency Gap in CNPs，2026-04-21 | 对 mean-aggregating Gaussian CNP，比对新增 context 与原 joint 条件化；在线性／有界 encoder、Lipschitz decoder、variance 下界等条件下给出 O(n^-2) gap [2] | 不是对 Sundial/Transformer/flow 的统一界，也不是精确一致；小 gap 可能来自新增观测影响被平均稀释，而非正确利用新证据 |
| FlowNP，NeurIPS 2025 | arbitrary context/target 条件 flow；§3 明确 consistency 不由架构保证，随机划分训练只促进它；App.A 分开 marginal 和 conditional consistency [3] | 可执行 conditional flow 是强普通近邻，不能把“能 conditional sample”写成跨 context 精确兼容 |
| CSDI | 直接训练 observed/missing mask 下 conditional score，§4.2–4.3 用多种 target-choice 策略 [4] | 它解决实际条件生成，不提供任意 masks 的有限样本跨查询 identity；其 §1／§3.3 已明确旧式 unconditional diffusion conditioning 近似不等于精确条件律 |
| TSFlow | GP prior、条件 GP 训练，以及 unconditional model 的 inference-time conditioning [5] | 是条件生成／forecast 的完整普通对照；其 Langevin／guidance 和不同训练模式的存在不等于任意重查询精确满足式 (1) |
| From Drift to Coherence，ICML 2026 | MCQA 中 PPR、seed prompting、self-consistency loss，降低生成答案历史下 belief drift；App.C 给带条件的近似 residual bound [6] | “加 rollout consistency loss”已有直接先例；它保持同一个问题的答案语义，不能把不同真实目标时刻的状态分布都压成相同 |
| Conditional Flow Matching for Bayesian Posterior Inference，AISTATS 2026 | block-triangular velocity、posterior transport 和 recovered posterior 的统计一致性 [7] | 这里 consistency 的含义是估计／后验恢复保证，不能据标题推断所有 context re-encoding 的代数一致性 |

MPNP 的 §2.3–2.4 已把 predictive c.i.d./martingale 条件与 exchangeable joint predictive 联系起来；这与 CNP consistency-gap 分析共同说明，相关原理和训练方向并非空白。但 exchangeable pseudo-context 的合同也不应被扩大成任意真实时间序列重新分窗后的保证。[2,8]

普通 AR/SSM 若保持参数、状态更新和条件化合同不变，就定义一个 joint law；它即使错设也满足自身概率法则。相反，一组分别训练得很准的多步 forecast heads 仍可能不兼容。是否有价值的残差，要看有限预算下的 joint 质量与查询成本，不看“模型理论上能表示什么”。

固定 joint 的 bridge/conditional sampler 在精确求解时当然满足式 (1)，因为那就是其条件分布定义。但“观察到端点／前缀就直接 clamp，并反复去噪”不自动等价于该精确 bridge；CSDI 和 TSFlow 对直接学习条件律与近似 test-time conditioning 的区分已经足够反驳这个泛化。此次不把旧后验更新配置重新打开。[4–5]

## 4. 唯一关键反证：所有后缀边际矩都能通过，而跨查询条件律仍完全错

### 4.1 一个可解析的反例

令原 joint 为

\[
U\sim N(0,1),\qquad V=\rho U+\sqrt{1-\rho^2}\,\epsilon,
\quad\epsilon\sim N(0,1),\quad 0<\rho<1.
\]

其正确条件律是 `N(rho U,1-rho^2)`。令错误 requery kernel 为 `Q_bad(V|U)=N(0,1)`，即完全忽略 U。二者后缀边际都严格为标准正态，因此对所有可积 g 都有

\[
E_{P_h}g(V)=E_{P_h^U}E_{Q_{bad}}g(V\mid U).
\]

均值、二阶矩、任意分位数和完整后缀边际分布都无法发现错误。但选 `w(U)=tanh(U)`、`g(V)=V`，则

\[
E_{P_h}[w(U)g(V)]=\rho E[U\tanh U]>0,
\qquad E_{P_h^U Q_{bad}}[w(U)g(V)]=0.\tag{2}
\]

这就是会改变诊断结论的反证：**未加权 tower 通过，不能关闭跨查询依赖不一致的动机。**反过来，式 (2) 非零也不能定位谁更好：若真实过程是相关 joint、旧 direct 错学成独立而 requery 学对条件，检出的不一致对应的是中途版本改进。把 requery 强行蒸馏回旧 joint 可能损害质量。

这不是一个新模型提案。将条件矩转成带 test function/instrument 的无条件矩，是既有 conditional moment restriction 方法；KCM 原文 §3、Theorem 3.2、§4 已给出更一般的 kernel 表述和检验。[9]

### 4.2 对应到 root 冻结协议的估计量

每个固定真实 origin h：

1. 独立采样 `i=1..N` 条原 joint 路径 `(U_i,V_i)~P_h`。
2. 对同一个 `U_i`，使用独立新 noise 采样 `j=1..M` 个 `V'_ij~Q_h(.|U_i)`。
3. 对预冻结的 w、g 计算

\[
D_i^{w,g}=w(U_i)\left[\frac1M\sum_{j=1}^M g(V'_{ij})-g(V_i)\right],
\quad\widehat\Delta_{w,g}=\frac1N\sum_iD_i^{w,g}.\tag{3}
\]

所有路径与 probes 都转换到**原始 history 的同一 reference 坐标**；式 (3) 的统计单位是 outer path。不能把 `N*M` 个 nested draws 当独立样本，也不能对各 inner 点重新 bootstrap。共享 prefix 的 pairwise MC CI 可直接由 `D_i` 的样本分布构造；inner 数增加只降低其中 requery 采样噪声，不会消除 direct suffix 的随机性。

parent 已在任何新采样前告知采用：

- `w=[1,tanh(prefix_mean_z),tanh(prefix_last_z)]`；
- `g=[suffix_mean_z,suffix_last_z,suffix_mean_square_z]`；
- 保持主 96+96；在 360+360 敏感性外新增 352+352，用于区分 16-point patch 相位；
- frozen-coefficient AR 正控和 prefix-independent 负控。

这九个矩是合理的小型筛查，不能声称完整 joint equality。若要将任意有限探针通过写成式 (1) 成立，证据会超过测量能力；不需要因此要求无限计算或无限 probes。

两条必须写进统计器的细节：

- 二阶 probe 用 `mean_j[g(V'_ij)]`，不能用 `g(mean_j V'_ij)`；后者会把条件方差人为压掉。
- exact-noise replay 用于 native/controlled sampler 的实现资格；不应用来让 requery 使用与生成 U 相关的旧 noise。720-dimensional joint flow 可能耦合所有 base-noise 坐标，所谓“剩余 noise”未必在给定 U 后仍是独立新噪声。

CI 需区分 MC 不确定性与 origin/dataset 泛化不确定性：固定 h 的 outer CI 只回答这个 origin；跨 origins 的实证汇总应另外按真实时间块处理。不同 origins 的有符号差异可能抵消，不应只展示 pooled signed mean。若九个矩作为一组作拒绝判断，采用预先声明的 Holm/Bonferroni 或同一 outer bootstrap 的 simultaneous interval，避免九次 pointwise 95% 区间带来的选择性显著。

## 5. 哪些差异先由 ordinary explanation 排除

| 可能解释 | 有效区分办法 | 不应作出的结论 |
|---|---|---|
| 有效信息窗口变了 | 记录 encoder 实际收到的原历史索引、输入长度和 wrapper crop；full-growing 名称不能替代张量审计 | 因截掉旧观测而变，不能称相同信息下的 state inconsistency |
| normalization/reference 改变 | 所有输出回原坐标；保持原 mean/std 的 controlled arm；确认没有二次内部归一化 | 数值单位／尺度差异不是新概率原理 |
| patch phase / padding / position 变了 | 主 96 对齐；比较事前固定的 360 与 352，并记录 padding/patch 索引 | phase 效应先属于接口／架构解释，不自动支持新 coherence objective |
| query semantic 改了 | 固定 suffix 的绝对时间、calendar/covariates 和 forecast vintage | 不同目标时刻或新增外部信息的 forecast 不该被压成相同 |
| MC／solver 误差 | 按 outer 分组；用精确 AR 正控、独立 suffix 负控；同 noise 验证 sampler 实现；在固定预算内核一档 solver 精度 | 偶然误差或求解器误差不能归到 state learning |
| 原 direct law 错设 | 自一致性诊断与真实数据 proper score 分开；比较修复方向 | requery 与旧 law 不同不等于 requery 更差 |

Sundial 原文报告 patch=16、固定长 head 及 shorter prediction truncation，可支持这些合同审计；FlowNP 明说 conditional consistency 不由 architecture 保证，CSDI 也分开 conditioning approximation。因此“这是 flow 所以应当精确”和“重编码不同就是科学失败”都不成立。[1,3–4]

对于新增真实观测，还必须区分新数值与新信息源：若未来天气预报 vintage、政策公告或真实传感器信息更新同时出现，P_h 可能根本没有对那些新变量定义 joint。此时只能比较共同信息合同；不能把新增真实信息当由旧 forecast 自己生成的证据。

## 6. 质量与成本：检出残差还不够立项

本轮若非零，仅得到式 (3) 的有限探针残差；若为零，只表示在该功效下未检出。两者都不能直接回答真实 forecasting quality。尤其逐时间点 CRPS 只看边际，可能对反例中的依赖变化完全不敏感。已有多变量 scoring 文献明确指出 energy score 对相关性错误的区分力可能弱，variogram score 更敏感但不是严格 proper；应搭配使用并固定要测的路径性质，而不是选一个最显著分数。[10]

若未来确需推进质量验证，可预先固定一项相关性敏感的真实路径指标（例如累计量的 CRPS、路径事件 Brier 或跨时点 variogram），同时保留边际 CRPS/energy score；同一真实 test window 比较 old joint、native requery 和 ordinary coherence/control。不同分支样本数不应以有限 ensemble bias 换取表面优势，成本应包括完整采样、重编码、conditional inference 与任何维护状态。

普通原理有完整解，不代表有限成本质量已被所有方法解决；反过来，一个 native 128M model 暂不一致，也不证明新 state framework 的必要性。SC fine-tuning、固定 temporal-order AR、单 joint 正确条件 API 和已有 conditional flow 都是必须靠近的普通基线。不能只对一个无一致性约束的模型做“修前—修后”，就把 regularization 收益命名为新概率世界模型。

## 7. 对冻结三个 RQ 的结论

**RQ1。**固定 joint 的真实条件化和固定 kernel 的组合律已有精确概率保证；context/mask 重编码则是另一个接口。近期 CNP gap、FlowNP 和 coherence fine-tuning 已明确定义和处理相邻问题，但不能将各自有条件或近似结论扩大到任意 Sundial history re-encoding。真实新增观测、模型生成前缀与外部信息版本更新应分开。

**RQ2。**目前尚无执行结果支持一个有限质量／成本新残差。root 的 bounded native diagnosis 合理，但必须把未加权 tower 扩为 prefix-weighted probes；否则独立重抽后缀这个严重负控也可能全通过。parent 已在执行前采纳，本文没有据实验输出改变判据。

**RQ3。**本轮不提出新机制。唯一能立即改变实验判断的贡献是上述测量反证及普通统计修复。若差异被窗口、reference、patch phase、MC 或正确 conditional API 解释，停止该动机；若不解释，也只建立 bounded self-consistency residual。下一步必须证明它在至少两种真实机制载体上有质量或查询成本后果，并超过已有 coherence 修复，才有资格提出新状态／转移学习交互。C23–25 旧后验问题、已知 CK 方案及 normalization/cache 方案保持关闭。

## 一手来源与精确位置

[1] Liu, Y., Qin, G., Shi, Z., et al. “Sundial: A Family of Highly Capable Time Series Foundation Models.” ICML 2025；v4 修订于 2026-05-09。[原文 v4](https://arxiv.org/html/2502.00816v4)：§4.1.1 tokenization、§4.1.3 TimeFlow/multi-patch；App.B Table 5；App.C.4 明确截取长输出；[作者代码入口](https://github.com/thuml/Sundial)。本文未重下权重，root 使用的固定 revision 以其资格记录为准。

[2] Young, R. “On the Conditioning Consistency Gap in Conditional Neural Processes.” arXiv:2604.19312，2026-04-21。[身份](https://arxiv.org/abs/2604.19312)；[原文 PDF](https://arxiv.org/pdf/2604.19312)：§3–4 的 gap 定义、§5 Theorem 1、§6 Theorem 2、§8 的标量数值实验。只引用有界／Gaussian／aggregation 条件下的结论，不推广为所有 NP/TSFM。

[3] Abu Hamad, H., and Rosenbaum, D. “Flow Matching Neural Processes.” NeurIPS 2025。[官方全文](https://papers.neurips.cc/paper_files/paper/2025/file/a92519f525c00085095fa41c5c46cdb5-Paper-Conference.pdf)；[HTML](https://arxiv.org/html/2512.23853v1)：§2.1 Eqs.3–5；§3 的 consistency 限定；App.A；[作者代码](https://github.com/danrsm/flowNP)。

[4] Tashiro, Y., Song, J., Song, Y., and Ermon, S. “CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation.” NeurIPS 2021。[官方全文](https://proceedings.neurips.cc/paper_files/paper/2021/file/cfe8504bda37b575c70ee1a8276f3486-Paper.pdf)：p.1 Introduction、§3.3 conditioning approximation、§4.1–4.3 Eqs.5–7；[作者代码](https://github.com/ermongroup/CSDI)。

[5] Kollovieh, M., Lienen, M., Lüdke, D., et al. “Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting.” ICLR 2025。[原文 v2](https://arxiv.org/html/2410.03024v2)：§3 的 unconditional／conditional generation 与 conditioning；[官方论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/ee1a1ecc92f35702b5c29dad3dc909ea-Paper-Conference.pdf)；[作者项目](https://www.cs.cit.tum.de/daml/tsflow/)。

[6] Kim, S., Lee, S., Fong, E., et al. “From Drift to Coherence: Stabilizing Beliefs in LLMs.” ICML 2026。[官方 proceedings](https://proceedings.mlr.press/v306/kim26ah.html)；[原文](https://arxiv.org/html/2606.17832v1)：§4.2 Definition 4.1、Theorem 4.2；App.C Theorems C.1–C.2，特别是全 horizon coverage／MC 一致等假设；§5 的 MCQA。

[7] Zhai, P. S., Jeong, S., and Rockova, V. “Conditional Flow Matching for Bayesian Posterior Inference.” AISTATS 2026，2026-05-02–05。[官方 proceedings](https://proceedings.mlr.press/v300/zhai26a.html)：abstract 的 block-triangular velocity、posterior transport 与 statistical consistency 定义范围。本文仅引用已核 abstract 级主张，不声称其处理本轮 TS 重编码。

[8] Lee, H., Yun, E., Nam, G., et al. “Martingale Posterior Neural Processes.” ICLR 2023。[原文](https://arxiv.org/html/2304.09431v1)：§2.3–2.4 的 c.i.d./exchangeable predictive 与 §3 的 pseudo-context construction；[官方 PDF](https://openreview.net/pdf?id=-9PVqZ-IR_)。

[9] Muandet, K., Jitkrittum, W., and Kübler, J. “Kernel Conditional Moment Test via Maximum Moment Restriction.” UAI 2020。[官方 proceedings](https://proceedings.mlr.press/v124/muandet20a.html)；[原文 PDF](https://proceedings.mlr.press/v124/muandet20a/muandet20a.pdf)：§3 条件矩到无条件矩、Theorem 3.2、§4 Eq.10 与 bootstrap。本文有限 9 probes 只借该原理，不宣称获得其无限函数类的完备性保证。

[10] Scheuerer, M., and Hamill, T. M. “Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities.” Monthly Weather Review 143, 1321–1334，2015。[作者机构全文](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf)：§1 的 energy score 相关性敏感度限制；§2 的 variogram score 及非 strictly proper 限制；§3 的相关性实验；§5 的组合评分建议。
