# C34 adjacent constructive perspective: chronology-supervised proposals for reusable trajectory queries

Date: 2026-10-17 Asia/Shanghai. Research intake only. **No model executed, no training, no weights/data downloaded, no server work.** Only small paper/code/metadata reads and this note. The three RQs and all exclusions in `RESEARCH_BRIEF.md` remain frozen.

## Abstract and bounded judgment

保留一个**可执行、低置信度的构造性问题**，尚不接受为新方法：对于同一个已训练的数值轨迹模拟器，能否利用未来路径免费提供的 first-crossing-time 标签，训练一个跨历史、阈值和 horizon 复用的低秩 proposal，从而以更少的模拟调用回答一组路径事件查询？关键比较是同样条件信息、参数、训练模拟预算的普通 conditional mixture，而不是只胜 iid Monte Carlo。

最近的 PathFM 已明确提出完整路径、事件时间和有限 ensemble 的代价；HEPA 已研究少标签事件预测。另一方面，Exogenous Matching、ARIS、MPPCA importance sampling、FAMIS 和 Diffusion QMC 分别覆盖条件 proposal、跨阈值复用、低秩多模态 proposal、defensive mixtures 和生成器的 source-space integration。因此，**“把 IS 加到时间序列生成器”没有新颖性**。本候选唯一值得测量的具体交互是：forecast state 为 proposal 提供可复用上下文，输出路径的首次越界时段为多模态 proposal 提供监督分工，多个查询再共用一组精确加权路径。这个交互是否比泛用方法更省数据/计算，当前没有实验依据。

## 1. Scope, method and source discipline

RQ1：近期 numerical TS world-model/framework 在什么输入和训练合同下支持什么能力？RQ2：同信息的普通训练/推断/表示控制之后，是否还有值得测量的跨任务 tension？RQ3：什么机制、最近先例、独立公开任务及单 5090 小实验能作出判别？

本视角按 deep-research 的 adjacent-mechanism route 检索，补查同问题的反方先例。先核对作者、题名、版本/时间，再核对方法正文/官方代码。主窗口为 2026-04-17..2026-10-17；较早的直接先例明确标为 foundational。没有把检索不到的模型纳入事实库。以下 F=source/code fact，R=author-reported evidence，I=本次推断，U=unrun hypothesis。

从 latent adaptation 出发的另一条尝试没有保留：NormWear-2 已有非参数 latent-transition adaptation，MetaKoopman 已有 conjugate Bayesian adaptation，另检索到 few-shot closed-loop identification 的 Meta-ICI。普通 EIV/IV 修正和 C26–29 的成熟辨识控制过近；本轮没有建立足以重开的新交互。[NormWear-2](https://arxiv.org/html/2605.15465v1), [MetaKoopman](https://arxiv.org/html/2607.26345v1), [Meta-ICI primary manuscript](https://openreview.net/pdf?id=oUWRVhTrWQ)。这不是宣称所有 noise-aware adaptation 都已解决。

## 2. What the recent frameworks actually support

| Capability axis | Verified source contract and reported evidence | Consequence for this lead |
|---|---|---|
| Covariate-conditioned predictive states | **F:** Zhu et al., WorldTS, submitted Sep25. Stage1 jointly learns target/covariate encoders and future-state prediction; Stage2 freezes them and fits an observation decoder. Numerical covariates can be past-only or future-known. **R:** 21-dataset forecasting study. [Primary full text, §§3–5](https://arxiv.org/html/2609.31162v1) | 支持可复用 state/dynamics recipe 的研究层次；不要求因果状态、动作、zero-shot 或 shared checkpoint。本文不从它的 point readout 推导“不能做世界模型”。 |
| Horizon-conditioned event utility | **F:** Petersen et al., HEPA, May11, current v4 Jun3. Per-dataset latent predictive pretraining, frozen encoder, predictor/event-head fine-tuning and a survival CDF. **R:** 14 benchmarks/11 domains and label-efficiency experiments. [Identity](https://arxiv.org/abs/2605.11130), [v1 method §§3.1–3.2](https://arxiv.org/html/2605.11130v1) | “直接输出事件概率、调 predictor 而非 encoder”已有；本候选不能将一个 event head 本身算创新。HEPA 的真实事件监督和本候选的模型概率计算是两个不同目标。 |
| Complete forecast paths and event functionals | **F:** Yao et al., PathFM, Oct15. Joint path flow matching, fast distillation, shared history computation and path-functional scores; Appendix C separates model error from sampling error and discusses rare-event relative error. **R:** fps-bench aggregate path results. [Identity](https://arxiv.org/abs/2610.22358), [full text §§3–4, C.1](https://arxiv.org/html/2610.22358v1) | 不能再提“TSFM 只会 marginals”或“首次评价 crossing time”。正文/abstract 的公开入口未提供可核验的对应 code/weights；本轮不把 PathFM 当 runnable baseline，亦不冒用同名生物模型。 |

WorldTS 的预测状态和 HEPA 的事件预测并不相互否定；PathFM 则给出完整路径的另一种合同。它们共同引出一个更窄的问题：已有 joint simulator 时，有限计算应投向哪些路径才能回答重复的事件查询？这是 inference capability，不是要求所有 framework 必须提供这一功能。

## 3. Closest-overlap challenge: what is already done

下表的“剩余区别”仅是当前候选的待验证限制，不是已经证明的 literature gap。

| Closest primary work | Same problem/mechanism already covered | Remaining difference to test; capable repair |
|---|---|---|
| **Exogenous Matching**, Yikang Chen, Dehui Du, Lili Tian; NeurIPS2024, **foundational**. [Proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/ee94bf235482e4c1f689c04c81656dbf-Abstract-Conference.html), [full text Algorithms1–2 / §B.1](https://arxiv.org/html/2410.13914v2), [official code](https://github.com/cyisk/exom) | 一次训练的 conditional source/exogenous proposal；IS；一个 expression family；其 stochastic process 可以把复杂查询写成条件分布学习。 | 本任务是同一个数值预测器跨 forecast histories 的 FPT surface，而非 SCM counterfactual identification。**换领域不算贡献**。需比较同 h/query 条件下的 ExoM-style conditional density；只有显式 crossing-time component supervision 的有限预算优势可保留。Generic ExoM 可以表达这类条件，故不存在表示能力独占性。 |
| **ARIS author release**, Sep2026 code artifact. [Pinned README](https://github.com/sisl/ARIS.jl/blob/312ca721933d7aeb670513b1ec33516241d4edff/README.md) | joint input/robustness model，通过截断 robustness 构造阈值 proposal，一次适配复用多个阈值；包含 CE、AMS、PMC 接口。 | 已经覆盖跨阈值 amortization。不能把每个阈值单独拟合的弱 CE 当唯一对照。候选还必须跨历史复用并检验 FPT-bin 分工；在同 history 内先与 ARIS-style joint robustness 模型比较。本文核验的是代码能力，不断言其论文发表状态或未取得的结果。 |
| **Scalable Importance Sampling in High Dimensions with Low-Rank Mixture Proposals**, Kruse, Schlichting, Kochenderfer2025, **foundational**. [Paper](https://arxiv.org/html/2505.13335v1), [official Python code](https://github.com/sisl/MPPCAImportanceSampling) | 已把长 disturbance sequences 的多失效模式和低有效维度结合；MPPCA proposals，closed-form EM，CE/SIS；有 100/200-dimensional oscillator 和 F16 examples。 | 低秩、多模式、时序输入均已有。候选只主张跨历史的条件共享和已知 first-hit 标签可减少 finite-data mode-discovery 成本，必须打败同 rank/components 的 CE-MPPCA 及 conditional MPPCA。 |
| **FAMIS**, Helal & Elvira2026-09-18. [Identity](https://arxiv.org/abs/2609.21160), [full text §§4–5](https://arxiv.org/html/2609.21160v1) | 共享 NF transport、多 mixture components、defensive exploration、tempered adaptation、mode-diversity regularization 和 mixture reweighting。 | “避免丢 mode”“defensive proposal”“共享 mixture”都不能单独 claim。候选以已有 FPT 分工替代每个新 history 的 expensive mode discovery，须报告 discovery/adaptation 调用全成本。 |
| **Diffusion Quasi-Monte Carlo**, Jianlong Chen & Yifeng Yu2026-08-11. [Identity](https://arxiv.org/abs/2608.11055), [full text §6.2/A.4](https://arxiv.org/html/2608.11055v1) | Gaussian source 后接 deterministic transport，再用 RQMC；conditional vorticity 实验测 randomization variance/cost，明确其 implicit sampler 不作真实分布 bias correction。 | Sobol/stratification 不是本候选创新，必须作为廉价 baseline。源空间权重可精确，并不等于 corrected true-world forecast law。 |

**三轴区别：**问题是 conditional path-probability computation；机制是 observed first-hit labels 对 proposal mixture 的监督分解；证据是 held-out forecast histories/thresholds 上、含一次性学习成本的调用—误差曲线。上述先例分别已经很接近其中若干轴，尤其 ExoM 的一般框架已包含足够表达能力。只有在这一个具体交互上形成可重复的 finite-budget 优势，才有研究内容；标题组合本身不足。

## 4. Constructive candidate and exact allowed information

固定一个已学习的生成器 `X=Gθ(h,Z)`，`Z~φd=N(0,Id)`；`h` 是可见历史经该模型编码后的 state，θ 不因查询而改变。允许每个 dataset 独立拟合 proposal，也允许之后研究从头训练小型 simulator；不规定 universal checkpoint。

查询为 `A(q)={Tτ(X)≤K}`，`Tτ=min({t:Xt>τ} ∪ {H+1})`。先只做单变量路径。训练/验证/测试都向所有方法提供相同历史、阈值、horizon、已冻结 G，以及训练期生成的 `(Z,G(h,Z))`。测试的真实未来只用于单独评估 forecast calibration，不能用于 proposal、阈值、模型或 cohort 选择。不存在 action/counterfactual claim。

模型已有 h 同时输入 proposal。将输出 first hit 按固定16步 bin 标记为 b，训练 `πb(h,q)` 和低秩 Gaussian components `qb(Z|h,q)`；no-hit 只服务训练统计，查询 proposal 仍保留 prior 成分。对 event-positive 的训练模拟记录，比较两个损失：

`Ltime = -Σ w_i [log π_bi(h_i,q_i)+log q_bi(Z_i|h_i,q_i)]`

`Lmix = -Σ w_i log Σ_b π_b(h_i,q_i) q_b(Z_i|h_i,q_i)`.

二者使用完全相同的 positive records、架构、rank/components、优化步数及 weights；只有 component responsibility 是否受 first-hit label 监督不同。`Lmix` 是不能省略的普通 conditional mixture 对照。负样本数、positive sample hit rate 和每个 bin 的样本量都公开；不得通过增加 rare-event training simulations 偷加预算。若训练样本来自 adaptive proposals，w 必须用其**当时真实 sampling density**纠正；最小实验先只用 iid training bank，避免这项附加复杂度。

每个 component 取 `N(μb, Db+UbUbᵀ)`，`Db` 为正 diagonal，rank `r∈{2,4}`；components 与最大 bin 数固定对应。h 通过同一个小网络调节 μ、D、U；所有竞争 conditional mixtures 获得相同 h。低秩是已有 primitive。这里没有假定 Gaussian source 的第t个坐标只影响第t个 forecast：**Sundial 的 joint flow 会混合这些坐标，b 是输出路径标签，不是 source 的因果时间标签。**

对一组预先给定的 queries，建立一个固定的 pooled proposal，平均其代表阈值/horizon 的 component mixture，再加入 `0.1 φd`：`Qh(Z)=0.1φd(Z)+0.9Σk αk qk(Z|h)`。α 在验证集冻结；最小版均分，避免再引入 adaptive allocation。每条 path 的权重为 `v=φd(Z)/Qh(Z)`，所有 queries 复用该 path 和同一个 v：

`p_hat(q) = M^(-1) Σ_i v_i 1{Tτ(Gθ(h,Z_i))≤K}`.

**数学事实/本次推导：**给定训练好的固定 Qh，独立 final draws 下该式对**固定 Gθ 的事件概率**无偏；不需 G 可逆或计算 forecast density；`Qh≥0.1φd` 保证 support 和 `v≤10`。用实际 mixture density 的 logsumexp 计算权重，不能只除以抽中的 component。不同时间/阈值嵌套 queries 共用正 weights，所以估计保留其单调顺序；不因这点声称新 coherence 方法。未经 self-normalization 的有限样本估计可能超过1，不剪裁、不误称概率 simplex；SNIS 若另报需标记有限样本有偏。学习时与 final evaluation 分开，避免 adaptive recycling 的额外无偏性问题。

**U：为什么 interaction 可能有用。** 同一个 event 的 source-space conditional set 可以有多个首次越界时段；小训练 bank 下无监督 mixture 的 mode assignment 可能不稳定。h 提供已被预测任务学习的 regime context，first-hit bins 提供免费 mode labels，而 query-bank pooling 把同一路径的求值用于多个风险查询。预期收益只在“多 histories × 多 queries × 小模拟预算”同时存在时成立；单 history、单阈值、廉价 G 或低维单峰问题可能没有收益。cache、IS、低秩和 amortization 单独都已有。

## 5. Actually released execution anchor

**F：Sundial, foundational 2025 baseline, official author model.** [Paper](https://arxiv.org/abs/2502.00816), [official repository](https://github.com/thuml/Sundial), [weights/model card](https://huggingface.co/thuml/sundial-base-128m)。本轮 Hugging Face metadata 核验 ungated repository，revision `3212e42564493f520593e5414af4367fc4b49226`，包括 safetensors 和 sampler source；HEAD 返回 weights 513,341,448 bytes，未下载。

最重要的 source contract：

- [flow_loss.py, pinned, sample40–51](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/flow_loss.py)：初始 Gaussian noise；随后给定 condition/noise 的 deterministic loop。
- [config.json](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/config.json)：`output_token_lens=[720]`，`num_sampling_steps=50`，hidden768，FP32。
- [modeling_sundial.py491–497](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/modeling_sundial.py)：last hidden conditions flow，再截取 requested horizon。
- [ts_generation_mixin.py24–38](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/ts_generation_mixin.py)：native generate 用 history mean 和 population std+1e-5 做外层 normalization；输入只能 `[batch,L]`。不能使用另一个 forward 分支的 std floor 替代。

因此 H96/144 也必须采样完整 `Z∈R720`，然后截断。不能把 noise 降成 H 维自称 native sampling。native update 使用 **predicted target 减去 original source noise**，不能擅自改成一般 vector-field Euler 步。Dropout 关闭，权重和50步保持原样。一次 history transformer encoding 可跨 paths/queries共享；所有 baseline 同享这项已有优化。它是 univariate joint-path sampler，本文不声称跨变量联合建模。

以下为未来 adapter 的可执行核心定义，**未执行**；调用方加载上述 pinned model 并 `eval()`，`e` 是 full720-dimensional source，history长度取672，输出 H≤720。它显式复现已核验的 outer normalization 和 source loop；先以相同 random bank 对原生 sampler作 replay，max abs tolerance1e-5，没通过就停止实验并核查 source，不将差异写作贡献。

```python
import torch

@torch.no_grad()
def cache_history(mdl, history):
    center = history.mean(-1, keepdim=True)
    scale = history.std(-1, unbiased=False, keepdim=True).add(1e-5)
    state = mdl.model(input_ids=(history-center)/scale,
                      return_dict=True, use_cache=False).last_hidden_state[:, -1]
    return state, center, scale

@torch.no_grad()
def path_from_source(mdl, cached, e, horizon):
    state, center, scale = cached
    assert state.shape[0] == 1 and e.shape[1] == 720 and horizon <= 720
    cond = state.expand(e.shape[0], -1)
    position = e.clone()
    steps = mdl.flow_loss.num_sampling_steps
    for index in range(steps):
        clock = e.new_full((e.shape[0],), 1000.0*index/steps)
        terminal = mdl.flow_loss.net(position, clock, cond)
        position = position + (terminal-e)/steps
    return position[:, :horizon]*scale + center
```

这里 native replay 需在原 `sample` 的 randn点捕获/注入相同 bank；不能仅比较相同 seed、不同 batch排列然后宣布误差。Adapter 是 source exposure，所有方法共用，不是新 simulator。

## 6. Independent public task and smallest decisive discriminator

**F：数据/配置公开且大小已核验，未下载。** [FEV author repository](https://github.com/autogluon/fev), [dataset](https://huggingface.co/datasets/autogluon/fev_datasets)。Dataset revision `f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8`；task YAML revision `f9e2c7d3152ed7d15c3b1c2877bb3ae7d977046f`。[Exact task definitions](https://github.com/autogluon/fev/blob/f9e2c7d3152ed7d15c3b1c2877bb3ae7d977046f/benchmarks/fev_bench/tasks.yaml)。原始数据可在 HF metadata/README 的 source 栏追溯。

| Pilot | Exact existing public material | Intended restricted contract |
|---|---|---|
| A | `solar_with_weather_15T`, field `target`, H96, official20windows；`solar_with_weather/15T/train-00000-of-00001.parquet` HEAD2,301,980bytes。 | 本次是 target-only sampler-integration audit。所有方法只得到同一 target history；不与原 FEV covariate-aware leaderboard 比较。 |
| B | `jena_weather_10T`, preselected field `target_0`, H144, official20windows；`jena_weather/10T/train-00000-of-00001.parquet` HEAD2,342,073bytes。 | 同样单 target、history-only。此字段在看结果前冻结，不按哪个传感器最有利选择。 |

这是**在现成数据上的独立 query-computation protocol**，不是 PathFM/fps-bench 或 FEV leaderboard 的原生复现。Sundial 可能在预训练中见过这些数据；本 pilot 不作 zero-shot novelty 或 contamination-free forecast-quality claim。对同一固定 G 的 integration comparison 不依赖“G 没见过该数据”的假设。

### Frozen minimum experiment — proposed only

1. **Seals and split.** 每数据只用上述一行/字段。最后20个不重叠 H-window作为 TEST official origins；最小screen预先取 window indices `{0,5,10,15}`，共8个 test contexts。倒数TEST之前留下32个H-window作 VAL，再留672步 embargo；其之前192个按等距索引确定的history origins作 TRAIN，训练未来块彼此不重叠，history可重叠。若长度/有限值检查不支持该规则，记录具体失败而不事后换字段。训练selection不得读取TEST future。
2. **Query freeze.** 在 TRAIN 的672步 history normalization下计算真实未来各前缀的 normalized max。训练阈值取 pooled TRAIN quantiles `.80,.90,.95`，验证 `.925`；测试 `.975,.995`，各自只由 TRAIN 确定。查询 horizons为 `{H/3,2H/3,H}`（整数），训练/测试保持同一物理网格；另报一个预先固定的 H/2 interpolation query。不会挑一个事后使Sundial稀有的阈值。完整报告所有 queries；按 reference p 分层仅作说明，不按该分层调模型。
3. **Shared offline data.** 总 TRAIN simulator bank budgets `{4096,16384}` paths，均分给192contexts，余数按context索引分配。每条路径可产生所有阈值/horizon的 event 和 bin 标签，不额外调用 G。proposal仅有3个初始化 seeds；rank2/4和小网络宽64在VAL选择。base simulator不训练。预先固定50epochs maximum，按VAL **model-probability estimation error**选择而不是只按proposal likelihood。每个VAL context另固定1024个独立iid reference paths，所有方法共享；其有限reference误差报告为selection limitation，这些调用与proposal optimization成本均计入offline总账。所有对照获同样bank和选择次数。
4. **Exact online budgets.** Final iid draws来自冻结 defensive mixture，`M∈{128,512,2048}`，32个独立seed。所有query共用一个pool，记录其实际G-call数和 50×M flow-net row evaluations。CE/SIS/SMC的proposal discovery、rejuvenation、failed draws和final draws都计入同一M；给予它们和候选同一full offline bank作为warm-start，同时另报native cold-start曲线，不能只限制经典方法。
5. **Reference.** 每个固定 TEST context用两个独立iid banks、各`65536` paths形成模型概率reference及binomial SE。它们不参加任何学习/选择。若不足以解析某query，标记reference unresolved；不以zero计数证明零风险、不以此支持稀有区优势。这个最小screen主要分辨p约0.005..0.1的积分效率；p≪0.001需要另外预算，不能外推。所有reference计算开销单列，不从研究总成本中消失。
6. **Single5090 accounting.** FP32 model，flow paths microbatch64起步，只调整microbatch以适配显存，保持预测合同；prefill单独计时，50步sampler计时，proposal forward/density计时，GPU synchronize后计warm/cold latency及peak memory。先测64path latency，再估计上述固定任务总成本；若预计超过2 GPU hours，按预登记顺序先执行M128/512及A/B各4context，保留其余为pending，不少报调用。可行性来自公开128M权重和小头结构，**不是已测runtime**。

### Required baselines, all exact same G

- iid Gaussian source；每坐标 randomized LHS；scrambled Sobol Gaussian inverse-CDF。PathFM-style midpoint stratification可作为另一个明确的非iid算法，但不得引用iid方差公式替它作置信区间。
- **最决定性的B0：**同h、q、rank、components、数据和参数预算的 conditional mixture，用 `Lmix` 而非 `Ltime`；再用随机置换bin标签作监督等量control。若B0足够，则所谓chronology机制结束。
- **B1：CE-MPPCA / CE-GMM**，使用 [official Python implementation](https://github.com/sisl/MPPCAImportanceSampling)，pinned`83b7e55d80a0fff2ac24308b278029f79463bf1d`。把limit-state adapter设为`τ-max(prefix G)`；不沿用其原任务reference，也不执行repo的全量launch script。
- **B2：ARIS-style joint robustness+noise model**，跨阈值共享；官方Julia artifact`312ca721...`。若跨语言调用成本影响端到端latency，批处理并同时给出G-call与语言overhead，不能把语言开销算机制优势。没有原生history-conditioning的部分应标记为本实验transfer adaptation；同history内保持原算法能力。
- **B3：subset simulation / SIS with pCN rejuvenation**，可用 [author implementation](https://github.com/furibec/rare_event_simulation)，pinned`d4dc0f018f79396bfa04ccdf2dd6ff1eb9569004`。对标准Gaussian source使用阈值级别和MH/pCN转移；每次G评价均计数。Sundial joint block没有可无偿复用的causal forecast prefix，不能假装逐时间切断source就是合法splitting。报告chain dependence，独立运行的方差，不能用iid ESS公式冒充CI。
- FAMIS为最近机制benchmark；其作者code release未在本轮确认，所以先列method-level strong control，未来若自主实现须写明transfer/reimplementation，不能称原生复现。没有这个对照不是已经赢过它。

### Primary discriminator and stopping rule

主指标：每query相对于**固定G的reference**的MSE及relative RMSE，按context聚类汇总，给seed重复误差；另报first-hit bins覆盖、zero-event estimate比例、max weight、log-density稳定性与估计均值对reference偏差。比较M128/512/2048的误差—原始G调用—latency曲线。所有baselines都可返回多query估计，不重复收费。

**只有以下四项同时成立才值得下一轮，而非接受最终方向：**

1. 两数据的预先固定contexts和两阈值层上，chronology-supervised proposal在相同M下相对最强普通conditional/ARIS/CE/SMC/RQMC对照有稳定收益；最低筛选量级为平均MSE降低≥25%，context-cluster bootstrap95%区间不跨0。8contexts只是screen，不能当最终显著性证据。
2. 在 **相同offline模拟数、优化数、参数数** 的B0上也成立；随机bin、移除h或只增加components不能复现全部收益。否则只是普通density capacity、history feature或额外监督数量收益，不支持当前交互。
3. 归入实际学习成本后，在预定最多1000个new-history query banks内发生break-even；同时给只算在线的曲线。超过这个需求量也没有摊平，就停止“practical finite-compute gain”主张。已有cache对全部方法开放。
4. 权重/原生replay通过，并且在reference解析度内没有可检测systematic bias。若只改善importance hit-rate但probability MSE不改善，直接撤销。

**立即否定本机制的结果：**同参数 `Lmix` 就达到同等MSE/latency；普通CE-MPPCA/ARIS/SIS在同等全部调用预算内清除差距；只在TRAIN seen history有效；只在单阈值有效；需要看TEST future选threshold/cohort；收益来自改G/少flowsteps/错误noise维度；只有在线好看而offline永不摊平。否定结论只关闭该组合，不宣称所有 rare-event simulator querying 已解决。

## 7. Why this is distinct from the closed lines, and material risks

本候选固定生成器，不通过NP/PJM decoder再训练或epoch选择争取误差；没有补动作记忆、换协方差belief、particle posterior correction或semigroup coherence toy；不创建新generator逼出失败。与旧uncertainty问题的**实质差别**是目标为**计算一个已给定条件分布的多个路径积分**，并把model error固定不动。评判量是source-space sampling cost，信息完全相同。

风险不是抽象“经典方法也会做”。ExoM已覆盖最一般的条件proposal构造，ARIS已覆盖阈值共享，MPPCA已覆盖高维失效模式；本候选可能只是它们的一种普通实例。first-hit bins未必对应source-space modes，同一个bin内部可能比跨bin更复杂；硬监督可能反而限制density。history-to-proposal amortization也可能在未见季节失效。最小protocol的少量contexts不能证明跨系统泛化。

还有一个不能混淆的上限：若`pθ(A|history)`本身错误，精确计算它也不会修正真实事件概率。真实future上的Brier/log score、coverage和event calibration必须另立数据充足的外部实验；这里的所有无偏性只针对固定Gθ。HEPA-style direct event models可能以更低成本给出更好的真实风险；若那才是用户最终任务，不能为保住trajectory机制而改评价目标。

## 8. Answers to the frozen RQs and next action

- **RQ1:** 已核验近期framework支持covariate-conditioned future states、horizon-conditioned event prediction及joint trajectory generation等不同合同；其证据不支持给“world model”强加physical/causal/action/zero-shot门槛。PathFM是当前最贴近的recent interface，但未确认其公开执行artifact。
- **RQ2:** 值得测量的是已有路径模型在小ensemble、多历史、多事件queries下的计算效率，而非从marginal scores或一次差score推出新的state缺陷。稀有事件MC困难本身是已知事实；尚待证明普通conditional/MIS/SMC控制之后有残差。
- **RQ3:** 一个具体可杀掉的lead是“first-hit labels监督的history-conditioned proposal mixture”。Sundial公开checkpoint提供精确G/noise合同；两个小型FEV来源提供独立既有数值序列。最小决定性比较是B0的 `Lmix` 对 `Ltime`，并同时给ARIS/CE-MPPCA/SIS/RQMC公平预算。先source replay，再跑固定8contextscreen；当前只完成来源/接口/协议定义，尚无效果、novelty或final-direction结论。

**Current status: one constructive minimum viable question; mechanism evidence missing; accepted directions remain0.**

## 9. Targeted supplement: first-hit-label responsibility supervision is insufficient as the differentiator

**2026-10-17，append-only reassessment；本节更新§8的去留判断。结论：撤回“first-hit labels监督conditional proposal mixture”作为新方法的立项依据。** 未找到一篇逐字实现“Sundial Gaussian source＋history-conditioned低秩mixture＋16步first-hit标签”的论文；但这不等于有可成立的机制空白。更近的原始文献分别明确覆盖时间失效分解、first-passage-time block分解、已知channel的神经density fitting，以及跨输出条件的amortized source proposal。下面给出目标函数层面的对应，而不是仅说ExoM具有足够表达能力。

### 9.1 Verified primary-source overlap and its limits

下列2001–2024工作均标为**foundational，早于主要检索窗口**；没有把旧机制包装为2026新结果。每行区分原文事实与本候选的对应推断。

| Primary source and identity | Actual mechanism verified in the original source | Consequence for this candidate; boundary |
|---|---|---|
| **Au & Beck (2001)**, *First Excursion Probabilities for Linear Systems by Very Efficient Importance Sampling*, Probabilistic Engineering Mechanics16(3),193–207, DOI10.1016/S0266-8920(01)00002-9. [Author-institution record and abstract](https://authors.library.caltech.edu/records/k8csm-w1659). | 对Gaussian white-noise驱动的linear dynamic system，将failure region分为“特定output在特定time失效”的elementary regions，利用其design points构造IS density；原文指出多个邻近design points都重要。 | **时间输出标签对应多个失效proposal的动机已有直接先例。** 它没有学习first-hit labels、conditional NN或跨forecast histories；不能说它已经实现本候选完整架构。这里的重合是output×time decomposition驱动proposal配置。 |
| **Murthy, Juneja & Blanchet (2014)**, *State-independent Importance Sampling for Random Walks with Regularly Varying Increments*, Stochastic Systems4(2),321–374, DOI10.1287/13-SSY114; preprint2012. [Original paper](https://arxiv.org/pdf/1206.3390), §4 Eq15, printed p15. | 将`P(τ_b<∞)`按`n_(k-1)<τ_b≤n_k`分块，先按`p_k`抽auxiliary block K，再用该block的change of measure构造`Z_K/p_K`无偏估计；原文采用几何time blocks，§4.2继续按失效机理分解。 | **first-hitting-time bins指导不同IS measures本身已存在。** 原问题为heavy-tailed random walk，非learned Gaussian-source forecasting，也非监督神经mixture；几何bins也不是固定16步bins。后两项工程改动并未构成新的统计交互。 |
| **Owen, Maximov & Chertkov (2019)**, *Importance sampling the union of rare events with an application to power systems analysis*, Electronic Journal of Statistics13(1),231–254, DOI10.1214/18-EJS1527; preprint2017. [Original full text](https://arxiv.org/html/1710.06965v2), §2 Eq3–4 and Theorem1. | ALOE用event-indexed conditional densities `q_j(z)=p(z)1_Hj(z)/P_j`形成mixture，再用整体mixture density计权；定理推广至一般event collections，条件是能够相应conditional sampling。 | 将事件换成first-hit strata是一个自然特例/推导，**不是ALOE原文声称做first-hit neural training**。ALOE的exact conditional draws和constituent masses在Sundial中不可免费取得；本候选用learned proposal近似它们。但“按输出event选component＋MIS”不是新机制。 |
| **Heimel et al. (2023; preprint2022)**, *MadNIS—Neural Multi-Channel Importance Sampling*. [Original full text](https://arxiv.org/html/2212.06172v2), §2.1 Eq5–6, §3 and AppendixA Eq76–79. | 通过已知/可学习的local channel decomposition `f_i(x)=α_i(x)f(x)`，为各channel训练神经importance densities；forward-KL部分给出按target/sampling-density比率加权的per-channel negative log likelihood，并支持buffered training减少昂贵integrand调用。 | 取hard channels `α_b(z)=1{B(z)=b}`即得到已知标签的per-bin density fitting。此取法是**本报告的直接specialization**，不是其LHC实验内容。MadNIS的local `α_i(z)`不是候选的history gating `π_i(h,q)`，不可混为同一个网络；但按已知语义分channel再训练neural densities这一机制已明确存在。 |
| **Chen, Du & Tian (2024)**, *Exogenous Matching: Learning Good Proposals for Tractable Counterfactual Estimation*. [Original full text](https://arxiv.org/html/2410.13914v2), §3.2 Eq8 and AppendixA.3 Algorithm1. | 对可计算simulator和可计算/可采样source distribution，抽取状态`s`及source`u`，以simulated outcome `Y_*^(s)(u)`为条件最小化`−log q(u|Y_*^(s)(u))`；其目标包含跨条件复用proposal。 | 显式设`u=z`、`s=(h,τ,K)`、`Y_*^(s)(z)=B_s(Gθ(h,z))`，就是按observed first-hit outcome进行source conditional density fitting。这个映射需要在模型输入中保留`s`，是对其interface的specialization；无需给Sundial赋予causal解释。候选额外的`π_b(s)`只补一个observed categorical label likelihood。 |

窗口内另核验到**Heimel et al., *MadNIS, Ready for Production at LO*，2026-10-12**，[author preprint](https://arxiv.org/abs/2610.17177)。它说明该neural multi-channel计算路线仍在发展；本次关闭候选的依据是上述已核验的旧目标函数，不借此近期title推断first-passage/forecasting结果。也不把2026的stochastic-output MDN surrogate误写为“输入source按first-hit标签训练proposal”的直接相同方法。

### 9.2 Exact loss reduction: this is observed-channel conditional maximum likelihood

**本段为独立推导，不是引用论文声称的Sundial结果。** 固定`s=(h,τ,K)`与生成器G，令event `A_s={T_τ(Gθ(h,Z))≤K}`，并令`B_s(z)=b`表示first-hit落入第b个时间bin。事件内各`B=b`互斥，空质量bins可省略。记

```text
a_s       = P_φ(A_s)
r_b(s)    = P_φ(B_s=b | A_s)
p_b(z|s)  = φ(z) 1{B_s(z)=b} / P_φ(B_s=b)
p*(z|s)   = φ(z) 1{A_s(z)} / a_s = Σ_b r_b(s) p_b(z|s).
```

这是全概率分解。用已观察到的component label拟合`π_b(s)q_b(z|s)`，候选population loss是

```text
E_[Z~φ | A_s] [−log π_(B_s(Z))(s) − log q_(B_s(Z))(Z|s)]
 = H(r(s), π(s))
   + Σ_b r_b(s) { H(p_b(·|s)) + KL[p_b(·|s) || q_b(·|s)] }.
```

因此`Ltime`是**已知channel label的complete-data maximum likelihood**：categorical CE拟合各bin质量，conditional forward KL拟合每bin source。若offline bank从其他已知density抽样，正确importance weights只改变该期望的估计方式。若均匀重采样各bin而不修正质量，则优化目标已经变了，不能再借上式声称拟合event-conditional density。

与ExoM的对应是“simulated outcome改成first-hit-bin后，原本的conditional NLL直接成立”；与MadNIS的对应是“将channel decomposition固定成这些hard indicators后，原本的weighted per-channel KL loss直接成立”。二者不是仅凭神经网络万能逼近作出的类比。再加`π_b`得到普通observed-mixture joint likelihood。更换flow为low-rank Gaussian、按h amortize、复用forecast context cache、把threshold作为输入，分别属于density-family、conditioning和系统缓存选择；现有候选没有额外给出这些选择如何产生一个超过已知outcome-channel方法的有限预算机制。

监督标签仍**可能**改善unlabeled mixture训练的label switching或mode allocation；这是未运行的性能假设，并非新机制证据。它也可能变差：同一first-hit bin在Gaussian-source space内仍可有多个分离区域；固定时间边界可能把易拟合的一个mode拆开。优化forward KL或估计真实bin概率也不保证最小IS variance，尤其在每bin approximation errors不同时。`π_b`不必等于真实`r_b`才能保持正确IS；是否有效必须看完整density ratio下的MSE。

### 9.3 Stronger discriminator and disposition

**原§6的`Ltime`对`Lmix`不再足以辨别方法新颖性。** 即使它达到预设≥25%收益，也只证明observed output channels对这一应用有用。应先加一个“标准observed-channel proposal”对照：同history、query、hard first-hit bins、Gaussian components、所有参数、训练bank和优化器，按上面MadNIS/conditional-MLE specialization实现。它与候选应逐项同构；这比跟不知道标签的`Lmix`比输赢更决定性。

下面是无需生成轨迹或长训练的精确检查；`log_q[n,b]`为各component对第n个source的log density，`b[n]`为已有observed first-hit label，`w[n]`为共同训练weight。两种写法来自同一observed-channel likelihood，CPU随机有限张量即可检查loss及对`logits,log_q`的梯度相等：

```python
log_pi = logits.log_softmax(-1)
row = torch.arange(b.numel(), device=b.device)
candidate = -(w * (log_pi[row, b] + log_q[row, b])).sum()
channel = torch.nn.functional.one_hot(b, num_classes=log_q.shape[-1])
ordinary = -(w[:, None] * channel * (log_pi + log_q)).sum()
torch.testing.assert_close(candidate, ordinary, rtol=1e-6, atol=1e-6)
# Compare torch.autograd.grad of both losses w.r.t. logits and log_q likewise.
```

这段是**可执行但本分支未运行**的discriminator；恒等式已由§9.2给出。它不会评价高斯family是否足够，也不会声称ALOE的exact conditional sampler已经能运行在Sundial上。若候选代码在这个检查中出现差异，应先解释额外信息、loss、参数或采样区别，不能自动把implementation差异当创新。没有差异时，继续大规模训练不能创造机制区别。

主线程另报告已在其server完成Sundial native qualification：**128,329,680 parameters；3个fixtures replay均exact0 difference；finite-difference gradient relative discrepancy0.001255；solar/Jena两公开data files已核验**。这是**主线程执行报告**，本分支没有重跑或访问server；它更新的是§3–4接口/数据可执行性，未提供proposal性能、真实事件calibration或新颖性证据。没有新proposal被训练。

**最终处置：不启动以“监督first-hit mixture responsibilities”为唯一新意的长prototype。** Sundial source-injection、exact weighting和两个公共任务仍可保留为普通rare-event querying研究的可执行基础；如果做小规模应用审计，应将方法命名为既有outcome-conditioned/multi-channel IS的transfer，而非新world-model framework。要重新立项，需要一个材料性的新交互，例如能明确减少既有conditional/channel方法无法同样省去的G evaluations或offline acquisition，并先用最小实验显示该差异；本节没有发现或验证这样的交互。

**Superseding status: first-hit-label-only method lead withdrawn; no retained new mechanism from this perspective; accepted directions remain0.**

