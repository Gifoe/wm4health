# C39 constructive/adjacent：固定未来目标的观测重查询，候选数 0

2026-10-17。检索前读取本轮 RESEARCH_BRIEF.md；未读 sibling notes。沿用 deep-research 原文核验方法。本轮只检索、读公开原文/代码与写本文件，没有训练、访问 server 或下载 weights/signals。

## 摘要

目前未建立一个能区别于完整近邻的新机制交互。不能把“联合预测 + 条件查询 + 缓存 + 少步 transport”本身立为新框架：FlowNP 已直接 amortize arbitrary conditionals；Sequential Flow Matching（SFM）已学习旧预测分布到新观测下分布的少步更新；10 月 15 日的 PathFM 已组合联合路径、缓存、蒸馏与 proper-score 校准。

仍值得保留的**具体待证残差**是：同一个未来目标，在真实观测逐批揭示时，低成本 re-query 能否同时保留原始 joint law 的条件质量与真实 proper-score 收益。现有原文尚未给出这个共同信息合同下的完整 Pareto 比较。但“未找到共同实验”不是已经证明存在方法缺口；目前应把它当下一项诊断，而不是给一个普通 conditioner 换名。

## 1. 研究问题与合同

沿用 brief 三个 RQ：已有方法对 origin joint law、模拟后重查询和新真实观测的条件律分别保证什么；有没有强普通模型之后仍存在的质量/成本残差；有没有单 5090 可失败的小机制。

先把对象写清。原点历史为 h，固定未来向量为 Y=(A,B)，原始预测为 P₀(A,B|h)。在 A 的前 r 个值 a₁:r 被真实观测后，两个对象是：

- 原 law 的条件分布 P₀(B|h,a₁:r)。
- 重新编码 growing history 的模型 Qᵣ(B|h,a₁:r)。

这里 B 的绝对时间索引必须保持不变，不能把两次查询不同的目标片段作相等比较。若 Qᵣ 与 P₀ 的条件分布不同，只建立模型定义差异；在 P₀ 有误时，Qᵣ 完全可能更准确。必须另外比较真实 B 的 proper score。

对模拟 A，tower moment 关系是必要诊断：E[A~P₀]E[B~Qᵣ(·|A)]f(B)=E[B~P₀]f(B)。少数 moments 通过并不证明全部条件律正确；从自生成 A 得出的误差也不是新真实观测后的预测损害。本报告不改动 root 已冻结的 Sundial native diagnosis。

## 2. 最接近完整先例：必须同时面对四种普通能力

| 完整路线 | 已核验的具体能力 | 对构思的约束 |
| --- | --- | --- |
| FlowNP | 同一 transformer 直接学习 joint 和任意 context/target 条件采样；ODE 步数控制速度 | “训练一个可重查询的概率 decoder”已有完整先例。 |
| Sequential Flow Matching | source 从独立 Gaussian 改为旧预测，学习新观测下的分布 transport，并做少步更新 | “保留旧 belief，便宜吸收新 evidence”已有完整机制。 |
| ForcingDAS | joint-trajectory prior，inference schedule 选择 filtering、fixed-lag、batch smoothing | “一个模型支持多个观测条件化模式”已有真实框架。 |
| PathFM | joint future paths，history cache，1–2 步蒸馏，fair-CRPS 校准 | 不能再把原 Sundial 的慢采样当作 generative forecast 的普遍成本缺口。 |

### FlowNP：这不是新的 amortization 方向

[Flow Matching Neural Processes](https://arxiv.org/html/2512.23853v1)，NeurIPS 2025，§3/Eq.7–9 用 observed-context tokens 与 noisy-target tokens 同时预测 velocity。§3.3 明确 exchangeability 有保证，marginal/conditional consistency 没有架构保证；§5/Table 3 的 joint-only 与 conditional 版本在 RBF/EMNIST 的 LL 为 1.73/2.54 与 1.69/2.50。这个小差异支持其容量/方便采样的取舍讨论，不能直接推出真实重查询会损害预测。Appendix D 给 ODE 步数与运行时间实验。

作者代码可访问，当前核验 pin 为 `f3403b6479dad33a7b93ba6edcf93c53d40f8e65`。[models/fnp.py:53–65](https://github.com/danrsm/flowNP/blob/f3403b6479dad33a7b93ba6edcf93c53d40f8e65/models/fnp.py#L53-L65) 将 context 与 target 拼接并全 self-attend；[predict 接口](https://github.com/danrsm/flowNP/blob/f3403b6479dad33a7b93ba6edcf93c53d40f8e65/models/fnp.py#L113) 可直接接不同 context。它是可执行的最近完整 arbitrary-query 对照，而非只有原则。

一处必须独立判断的数学边界：其 Appendix A 讨论用 p(full)/p(context) 得到条件律，同时又允许不同 target-set 的 marginal 不一致。若分母是另一个 query 下单独计算的 density，未必是固定 full joint 的真实边缘，不能未经归一化核查就称其为同一 law 的 Bayes 条件。应固定 full joint 再条件化，或显式说明只是模型给出的 ratio approximation。

### SFM：旧预测作为更新源也已被训练

[Accelerated Sequential Flow Matching: A Bayesian Filtering Perspective](https://arxiv.org/html/2602.05319v1)，2026-02-05，§3.1/Table 1 的预测变量本身可以是整段未来轨迹；§3.3/Eq.6 将旧预测 law 作为 transport source。§3.4/Algorithm 1–2 实际采用 pretrained 旧预测样本与新 ground truth 的 hybrid pairs，再加 re-noising，降低 train/test source mismatch。§5.4–5.5 与 Appendix C 比较质量、latency 和构造消融。它直接覆盖 warm-start belief update 的宽泛方案。

其 Proposition 3.1 的 ideal coupling 条件不能直接等同于实践的 hybrid pairing，更不能据此宣称它是某个冻结原始 joint law 的精确条件 sampler。但这是保证范围，不是它无效的证据。作者已实际训练该组合，应优先做同信息质量/成本比较。[作者仓库](https://github.com/Graph-COM/Sequential_Flow_Matching)，当前核验 pin `04c7c637b2355cbd98bf49031141fb1296acb36e`，README 给 pretrain → generated fine-tuning set → fine-tune 的三阶段流程。

### ForcingDAS：观测条件化覆盖面广，但 calibrated law 的证据有限

[ForcingDAS](https://arxiv.org/html/2605.14285v1)，2026-05-14，§3.1/Eq.4–7 联结 causality-aware noise scheduling、observation-gradient guidance 和 filtering/smoothing 模式。Appendix S9.1 明确承认长 active window 成本增长、causal-only smoothing 限制，以及尚未系统评价 ensemble calibration。这个 calibration 边界支持继续核查，不支持声称其返回样本已经定义了被验证的跨查询概率 law。[作者代码](https://github.com/umjiayx/ForcingDAS)，核验 pin `ae2fa289693acbc306e41f7aab146012debfa845`；README 有按 schedule 切换的入口。

### PathFM：最新成本对照已经改变

[PathFM: Fast Generative Forecasting of Multivariate Trajectories](https://arxiv.org/html/2610.22358v1)，2026-10-15，§3.2 将 flow-independent history 与后续 path-dependent 运算分离并 packed-cache；§3.3/Eq.3–5 用 PDD 缩至 1–2 evaluations，再保留 teacher supervision 做 fair-CRPS 反传。§4 评价路径与聚合功能。这里的“query”主要是对同批 paths 计算 functionals，不能偷换成新增真实 observation 的同-law conditioning；其 joint law 也没有因此得到 tower 保证。Reproducibility statement 仍为将发布 code/weights，本轮没有找到作者训练入口，不假装已可直接跑官方 checkpoint。

## 3. 最近一致性原理不足以独立立项

[From Drift to Coherence](https://arxiv.org/html/2606.17832v1)，2026-06-16，§4.1–4.2 和 Appendix C 已给 predictive resampling、不同 rollout horizons 的 stop-gradient self-consistency amortization。其任务为离散 multiple-choice beliefs，不是本轮连续时间序列；但把连续 moments/样本距离塞进同类一致性微调，没有新增机制区别。

更早的 [ACFlow](https://proceedings.mlr.press/v119/li20a/li20a.pdf) §3–4 已训练 arbitrary conditional flow；[ACE](https://acerbilab.org/amortized-conditioning-engine/) 的作者说明和公开代码也把可 conditioning/predicting 的 data 与 latent 统一。故“统一查询接口”与“普通 amortized Bayes”需要被承认为已知能力，不因它们出自别的任务领域就变成新框架。

这里保留一个具体、尚未排除的问题：**当新 evidence 强烈改变同一未来 tail 的分布时，少步更新是否会在反复揭示观测的过程中损失 proper-score 质量；若损失，单纯增加采样步数、标准 amortized conditioner、SFM 或已有 coherence fine-tune 能否以合理成本消除？** 原文没有把这一固定-tail 合同逐项排除。当前它是可验证的问题，尚不是作者已证实的共同失败。

## 4. 为什么本轮没有新候选

最接近的构思会是“固定联合轨迹 generator + 可保存的 stochastic belief + evidence-conditioned 小 transport + 少步校准”。其每一步有明确最近先例：FlowNP 的条件化、SFM 的 belief source、PathFM 的缓存/蒸馏/校准，分别覆盖对应作用。已知组件允许形成新贡献，但本轮没有定位到一个新的、必要的相互作用：

- 若 transport 只是从 (history,observed-prefix) 学 tail 条件分布，就是普通 conditional generative learning。
- 若从旧预测开始 transport，就是 SFM 已做的设计；要证明有新内容，必须进一步说明哪一种 coupling/状态接口改变了可测的质量—成本关系。
- 若对不同 query 加 tower/CK penalty，就是已知 coherence training；降低 penalty 不等于提高真实数据预测。
- 若用 frozen generator 的自生成 (prefix,tail) 训练 conditional sampler，必须先承认这是 teacher conditional amortization；不能绕过 C34 等价问题。

本轮没有提出附加模块来遮掩这些等价关系。候选数为 0，不是因为要求新数学算子，而是尚未写出区别于完整普通方法的具体交互。

## 5. 下一项最有信息增益的试验：固定 tail 的真实 reveal 质量—成本 screen

这是一项**已知方法诊断**，不是新方法实训。先等 root 原 Sundial 的 coordinate/MC/native 资格结果；若其动机已被普通解释消除，本项也不自动启动。若继续，优先做真实 proper-score，而非继续扩 tower moments 列表。

### 输入与查询合同

两种真实载体预先固定：ETTh1 的 hourly oil temperature 与 Jena weather 的 hourly temperature。只用相应目标的历史数值；不引入 future covariates、隐含系统参数或额外传感器，因此不会重开 C23–25 unknown-system posterior 或 C35 protected multimodal 配置。

- ETTh1 数据提供方公开 CSV、字段和 hourly variant；[作者仓库](https://github.com/zhouhaoyi/ETDataset#ett-small) 与 [LICENSE](https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE) 可核查，其许可是 CC BY-ND 4.0，不能误写成 CC BY；本试验不发布派生数据包。
- Jena 提供方有 historical archive 与 CC BY 4.0 条款：[MPI-BGC 官方下载页](https://www.bgc-jena.mpg.de/wetter/weather_data.html)。使用已有冻结版本优先，数据范围/hash 在 train 前登记，本轮未取信号。

T=192、origin horizon H=96。所有重查询始终预测相同 B=Y₆₅:₉₆；逐次揭示真实前缀 r=8、32，剩余 gap Yᵣ₊₁:₆₄ 保持未知、由模型积分/采样，不能偷看。模型收到完整原 history 加已真实揭示前缀、绝对时间/lead 与 mask，不截成会遗忘原历史的固定新窗口。训练/验证/test 按时间隔离，normalization 仅用 train；同-origin 的坐标在三个查询间固定。

### 最小基线集合

| arm | 目的 | 必须保留的强普通能力 |
| --- | --- | --- |
| AR-spline joint | 可精确执行前缀条件化的正控制和质量对照 | 同一因果序列 law，非 Gaussian spline innovation，缓存 recurrent state，draws 并行；未揭示 gap 正常 ancestral sampling。不能用单 Gaussian AR 制造多峰失败。 |
| FlowNP-style conditional flow | 直接 amortization 对照 | 训练 r=0/8/32 的共同 query 分布，32 步 teacher；报告 1/2/8/32 步曲线，必要少步 student 使用标准蒸馏。 |
| SFM-style warm source | 核查旧预测能否低成本吸收新 evidence | 相同 flow backbone/训练输入，以旧 B 样本为 source，按原方法 hybrid pairs/renoise；更新目标仍为同一 B，不把 horizon shift 混进收益。此处是 SFM 的合同适配，不声称是新方法。 |

AR 模型定义的 finite joint 与其条件采样是同一 law，可先用它的自生成前缀检查整个 MC/tower 管道；真实 proper score 再使用真实前缀和真实 B。不能用 AR 自生成数据上的 law fidelity 取代真实预测质量。

先每数据两种固定 seeds，3 arms × 2 data × 2 seeds = 12 fits；FlowNP/SFM 尽量共享 pretrained flow 起点，额外 source-generation、蒸馏成本单列并以相同新增训练预算给普通对照，不把 teacher 成本当免费。模型规模先限制在约 1–3M，batch 32，先实测 100 steps 显存/耗时，目标峰值不超过 16 GiB；用统一梯度累积避免 arm 独享大 batch。训练上限与早停规则在 test 前固定，达到上限仍未收敛时只报 screen 不足，不能用 undertraining 宣判机制。

若出现稳定 residual，再预先保留 4 个 fits，将普通 FlowNP 加已知 resampling/self-consistency fine-tune，作为必须跨越的修复对照；此时上限共 16 fits。没有 residual 就不做这 4 个。对于当轮没有可运行官方 weights 的 PathFM，不以缩小自建模型冒充复现；其公开完整算法应作为少步/缓存/校准配方的近邻，而非被遗漏的竞争者。

### 指标、成本与可证伪门槛

所有模型在同一 B 上用 32 iid paths；主看 multivariate energy score 与 B 总量的 CRPS，辅看 point MAE、marginal CRPS、interval coverage 和 variogram。使用共同数据尺度；fair estimator 的有效样本数按独立 path 计算，不把 share-noise branches 当 iid。C37 已证明的 rank-coupled quantile completion 可以作低成本补充对照，不能再以 independent quantiles 当唯一 generative 替身。

成本同时报首次 origin generation、每次真实 reveal update、32 draws 的 end-to-end latency、GPU seconds 和 peak memory；cache 构建/失效/source-generation 均计入。NFE 只是分解，不把更小网络一次调用等同为大 backbone 一次调用。按时间块对同一 origins 的 score 差作配对区间；tower 的 nested MC 以 outer path 为簇。

**关闭规则：**若某个上述普通对照在两种载体上都在最佳真实 score 的 2% 内、同时把 update latency 降至少一半，当前预算下没有足够重要的质量—成本残差；采用已有方法，停止此动机。2%/2× 是预登记工程 screen 门槛，不是文献定律。若 native re-encode 与原 law 不一致但真实 score 更好，也不能把强制回原 law 当成功目标。

**保留问题而不晋升方法的规则：**只有新真实前缀对 B 的预测确有稳定收益、AR/joint law 本身已合格，而少步普通 conditional/SFM/coherence 修复在相同 latency 下仍稳定损失 ES 与总量 CRPS，才建立“这里有有限预算条件化成本”这一经验残差。即使成立，也还没有新机制；下一轮必须围绕测到的失效来写出可消融的交互，而不是现在先造一个。

## 6. RQ 回答

1. 已有方法的能力并不相同：fixed-law Bayes 条件、amortized context map、receding-horizon belief transport、trajectory functional queries 应分开。CK/tower、自生成 query、真实 observation 更新不能互相替代。
2. 目前最具体的剩余项是固定同一未来 tail、分批真实 reveal 的质量—成本 Pareto 关系。它仍待实测，不能依据 re-encode 不等或旧 sampler 慢宣称已成立。
3. 新候选 0；最高信息增益是上述 12-fit（必要时 16-fit）普通方法 screen，而非新 CK 正则、masked flow 或 posterior sampler 命名。最终方向仍可为 0。

## 主要来源定位

- [FlowNP 全文](https://arxiv.org/html/2512.23853v1)：§3，特别是 §3.3；§5/Table 3；Appendix A、D；[官方代码](https://github.com/danrsm/flowNP)。主窗前直接近邻。
- [SFM 全文](https://arxiv.org/html/2602.05319v1)：§3.1/Table 1；§3.3/Eq.6/Proposition 3.1；§3.4/Algorithms 1–2；§5.4–5.5/Appendix C；[官方代码](https://github.com/Graph-COM/Sequential_Flow_Matching)。主窗前直接近邻。
- [ForcingDAS 全文](https://arxiv.org/html/2605.14285v1)：§3.1/Eq.4–7；Appendix S9.1；[官方代码](https://github.com/umjiayx/ForcingDAS)。主窗内。
- [PathFM 全文](https://arxiv.org/html/2610.22358v1)：§2；§3.2–3.4/Eq.3–5；§4；Reproducibility statement。主窗内，2026-10-15。
- [From Drift to Coherence 全文](https://arxiv.org/html/2606.17832v1)：§4.1–4.2、Appendix C。主窗内，已知近邻，仅用于去重。
- [ACFlow 官方 proceedings PDF](https://proceedings.mlr.press/v119/li20a/li20a.pdf)：§3–4；[ACE 作者项目页](https://acerbilab.org/amortized-conditioning-engine/)，[作者代码](https://github.com/acerbilab/amortized-conditioning-engine)。更早普通 conditioning 能力，不充当时间序列实测证据。

未采用二手 review 的性能数字。NP conditioning-gap 论文可检索到原文缓存，但 arXiv HTML 本轮不可用，未用它的泛化界为本报告的实测残差背书。检索亦读到 AFM 与 measurement-conditional flow 近邻，但不将图像逆问题的质量数字外推为时间序列重查询性能。
