# C58 Mainstream 独立检索记录

日期：2026-10-18。执行者：mainstream58。仅检索、读代码与编写本文件；未访问服务器、未训练、未改环境、未写 GitHub。先读 deep-research/SKILL.md、citation-protocol 与 C58 冻结 brief。本文是供根任务合并的 perspective memo，不是完整综述或已经成立的方法。

## 结论与校准

**本视角接受的新方法 lead 数：0。** 主流方法的覆盖范围比“确定性 JEPA 只能平均未来”这一叙述广。直接把 WorldTS 改成 latent flow、Gaussian mixture、多个 prediction heads、跨 horizon distribution matching、state/context 双 latent，均已有很近的完整框架；其缺少某个 TS 实验不自动构成方法机会。

下文保留一个更窄、尚未通过创新性门槛的可反证问题：**非几何 regime duration 下，predictive-only latent 模型是否需要显式 residual-life belief？** 它最多是排除问题，不是建议启动的方向。其成熟 HSMM / switching LDS 近邻很强，没有证据说明神经 world-model 封装比这些方法增加了不可替代的能力。

## 检索方法与阅读深度

以作者论文、arXiv 当前 HTML、会议 PDF、作者仓库和 raw source 为事实来源。搜索摘要和第三方聚合页仅用于找 primary URL；没有采用 AI review 的方法判断或性能结论。未复现任何论文结果，所有数字若保留均仅是作者报告。特别注意：`2607.05238` 的当前正文已是 **Branch-JEPA v3**，不能把旧 MoP-JEPA 摘要当作当前完整方法。

阅读深度记号：**M** = 阅读正文方法/公式和实验协议指定部分；**A** = 仅 primary abstract/README；**C** = 查看实际源码相关函数；**T** = 仅仓库目录/文件存在性，未运行。下表的阴性代码结论均是“本次未核实链接”，不是断言全网无代码。

| 工作与 primary URL | 已核实的事实 | 本次读深与代码状态 |
|---|---|---|
| [WorldTS, 2609.31162v1](https://arxiv.org/html/2609.31162v1)；[official code](https://github.com/decisionintelligence/WorldTS) | covariate-conditioned future latent block prediction；Stage 1 使用 latent L1/cosine/VICReg；Stage 2 冻结编码器和 predictor，训练同一 decoder 的 prediction/reconstruction losses；21 个真实 forecasting datasets。 | **M+C**：Sec.4.4–4.5、5.1；本地官方 TimeMMD `worldts_model.py` 的 stage freezing / forecast 和 `exp_worldts_forecasting.py` 两阶段 loss / 训练路径。不是“已验证可干预 simulator”。 |
| [LatentTSF, 2602.00297v2](https://arxiv.org/html/2602.00297v2)；[official code](https://github.com/Muyiiiii/LatentTSF) | 先训练 AE 再冻结，以 latent MSE/cosine 训练 forecasting backbone；可选 observation perceptual loss 默认关闭。 | **M+T**：Sec.3 和 5.7、limitations 标题/说明；README、`my_train.py` / `my_AE.py` / pretrained checkpoints 路径已确认。未读完所有 backbone，也未重跑。不能由 temporal locality 指标推得物理 state recovery。 |
| [Pythia, 2610.05240](https://arxiv.org/html/2610.05240) | 正文明确把 world model 定义为 observational forecasting 的 context-conditioned predictive states。context residual 有 stop-gradient numerical-reference supervision；随后冻结 representation，再适配 probabilistic decoder。 | **M**：Sec.3、4 和 corpus/benchmark 描述；未核实可执行作者仓库。无代码不是 gap。其 zero-shot shared pretraining 是该作任务选择，不是本项目强制条件。 |
| [TSWM accuracy/mechanism consistency, 2610.01842v1](https://arxiv.org/html/2610.01842v1)；[official repo](https://github.com/HaochenZhang717/TSWM-Official-Repo) | 比较 prediction spaces、plan fusion/encoding，并用 declared signed action–state relations 评估 response；directional supervision 对错误符号部分加损失。 | **M+T**：Sec.3–5 与 code availability；官方 repository tree 核实，有 `fusion_train.py`、`encode_train.py`、`autonomy_train.py`、`soft_train.py` 等。结果是其 benchmark 范围内的 sign consistency；不等于从 observational clinical data 识别出干预效应。 |
| [COFE, IJCAI 2026](https://www.ijcai.org/proceedings/2026/602)；[full paper](https://www.ijcai.org/proceedings/2026/0602.pdf)；[code](https://github.com/polars8948/COFE) | VAE framework 分 coarse/fine dynamics，使用 adjacent-latent innovation independence 与 sparse innovation constraint。慢 latent 的关键前提是大部分时刻 innovation 为零；这是具体生成假设，并非任意 slowly changing state 的无条件识别。 | **M+C**：Sec.3.2–3.3，正文公式(6)–(11)；读 forecaster `COFE.py` 的 `loss/forecast`，NN `COFE.py` 的 `encode/forward/sample`、ETTh1 YAML。存在真实 NLL、reconstruction、KL 和 sparsity/contrast constraints，不能当作只有 README 的占位工作。 |
| [KUMA, authors' primary page](https://superbearcafedo.github.io/)；[OpenReview identity](https://openreview.net/forum?id=KIJq2riCoy) | 作者摘要称 input-dependent Koopman separation + U-shaped multilevel attention，评估12 datasets。 | **A only**：OpenReview 正文被 browser challenge 阻断；只用作者主页 abstract 核实存在与粗粒度机制。不据此声称其定理/对照足够或代码缺失已被证明。 |
| [Meta-iLaD, author repo](https://github.com/GoForIt-YYB/Meta-iLaD) | 作者摘要提出从 few-shot examples 提取 environment variable，联合讨论 state/environment/dynamics identifiability 与 unseen-environment forecasting。 | **A+T only**：README，未成功打开其 `paper.pdf`，所以未审计 theorem assumptions。repository tree 目前只见 README、paper、poster；README 标明 code will be released soon。不能把它忽略后重提“加一个 environment latent”。 |
| [OSSM, 2609.33566v1](https://arxiv.org/html/2609.33566v1) | 同一 latent propagation 横跨 context/forecast；observations 通过 observer 校正，明确将传播与测量同化分开。 | **M**：Sec.3、4 与 Appendix C；正文称代码在 supplementary，基于 TFB，仅提供 OSSM folder。本次未取得该 supplementary，不冒充已有公开可执行镜像。 |
| [EpicWorldModel, 2610.05996](https://arxiv.org/html/2610.05996) | conditional flow predicts future latent blocks from history/actions；joint representation learning；采样 predictive variance 进入 CEM exploration bonus，正文也明确该量作为实用 bonus，而非精确 entropy estimate。 | **M**：Sec.3.1–3.3、实验协议与 compute-matched appendix 标题/说明；未核实作者源码链接。不能把 stochastic latent flow 或 uncertainty-guided planning 本身再当新机制。 |
| [Branch-JEPA, 2607.05238v3](https://arxiv.org/html/2607.05238v3)；[current abstract/version history](https://arxiv.org/abs/2607.05238) | 不融合的 weighted latent branches 经共享 decoder 分别解码；除 hard specialization 外，当前版本有 latent/decoded full-set Energy Score 和 parameter-exact output-only branching control。 | **M**：当前正文 architecture、losses、population target、AV2/OGBench protocols。未核实作者 code。旧名 MoP-JEPA 的“只做 hard assignment”概括已经不完整。 |
| [MDN-VJEPA, 2610.13956](https://arxiv.org/html/2610.13956) | Gaussian-mixture successor law；同一 categorical variable 耦合完整 direct trajectory；以 samples/energy distance 比较 direct 与 recursively composed laws。Sec.4.4 明确它**没有** persistent `p(M[t+1]|M[t])`，其 continuous predictive state 仍是核心状态。 | **M**：Sec.2.2–4.5，重点 Eq.(15)、(20)–(22) 及模式语义边界；未核实作者源码。正文页头日期与 arXiv ID 月份不一致，本文不利用该日期排列先后。其明确覆盖概率多未来、trajectory coupling、CK consistency。 |
| [VJEPA primary paper](https://arxiv.org/html/2601.14354)；[author code](https://github.com/YongchaoHuang/VJEPA) | 实际仓库有 Gaussian probabilistic prediction / Bayesian soft fusion 的实现及 Kalman、RSSM 对照 notebook。 | **C+T**：读 `vjepa_Yongchao.py` 的 class/loss/fusion 相关实现，查看实验目录。论文仅作为身份与范围指针，未读完其长正文，本文不借它作定理背书。 |
| [Koopa, primary paper](https://arxiv.org/abs/2305.18803)；[official repo](https://github.com/thuml/Koopa) | time-variant/time-invariant Koopman predictors；官方明确提供接收新观测后进行 DMD operator adaptation 的 rolling-forecast notebook。 | **A+T**：论文 abstract、完整 README 和 notebook 存在性；不是本轮全面复核。已足够否定“只有固定 operator，没有 incoming-observation adaptation”这一笼统动机。 |
| [TACTiS-2 primary paper](https://arxiv.org/abs/2310.01327)；[official code](https://github.com/ServiceNow/TACTiS) | joint probabilistic forecasting/interpolation，并支持 unaligned / unevenly sampled series。 | **A+T**：primary abstract、README，`train.py` 与 random-walk demo 存在性。未运行。任意 missingness/query 的“统一接口”不能只对独立 quantiles 比较；须加该类 joint conditional control。 |
| [Deep State Space Models, NeurIPS 2018](https://papers.neurips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf) | RNN 根据 covariates 参数化每条序列的 linear-Gaussian SSM；共享网络学习，Kalman inference。预测区 covariates 被假定给定；条件于 covariates 与参数时，各条序列独立。 | **M**：Sec.3–4 的模型与 assumptions；未核实当前可执行实现。因此只作成熟结构近邻，不作为5090即用代码推荐。 |
| [PlaNet primary paper](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)；[official code](https://github.com/google-research/planet) | deterministic + stochastic latent transitions；latent overshooting；latent CEM/replanning。 | **A+T**：primary abstract、official README；旧 TensorFlow1.13 repository 已 archived。历史方法必须比较，但这个旧实现不是当前5090无需迁移的可执行基础。 |
| [RC-Mamba primary paper](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1903497/full) | Student-t emission HMM 后验调制 selective SSM；其摘要明确 uncertainty gate 的 matched controls 没有识别出超越 posterior-independent scaling 的收益。 | **A**：只读 primary abstract/范围；未核实作者代码。作为新发现近邻，不借其金融结果证明跨领域能力。 |
| [SSM author code](https://github.com/lindermanlab/ssm) | 官方现成支持 HMM、ARHMM、IOHMM、HSMM、LDS、SLDS、rSLDS 与 missing data。 | **A+T**：README、模型列表与训练接口。Autograd/Cython 实现，CPU controls 足够；本轮未安装、未验证在现环境运行。 |

## 已确认的代码底座，不等于已通过运行资格

GitHub read-only API 在本轮返回以下 HEAD SHA；随后逐仓用 `commits/HEAD` 验证了它们确为 commit SHA：

| 仓库 | 当时 HEAD commit SHA | 已实际定位的关键文件 |
|---|---|---|
| `polars8948/COFE` | `a53409aa6d97c65b5b36b7f6b27fc5fbdb5e3df1` | [`probts/model/forecaster/prob_forecaster/COFE.py`](https://github.com/polars8948/COFE/blob/a53409aa6d97c65b5b36b7f6b27fc5fbdb5e3df1/probts/model/forecaster/prob_forecaster/COFE.py)、`probts/model/nn/prob/COFE.py`、`config/ltsf/etth1/COFE.yaml`；同仓库含 CSDI/TimeGrad/GRU-MAF/GRU-NVP configs。 |
| `YongchaoHuang/VJEPA` | `88a90d2c0af970edd46d44d53d4864d10aa6d117` | [`vjepa_Yongchao.py`](https://github.com/YongchaoHuang/VJEPA/blob/88a90d2c0af970edd46d44d53d4864d10aa6d117/vjepa_Yongchao.py)、`exp4_gmm_vjepa`、`exp5_kf_comparison`、`exp7_rssm`。 |
| `HaochenZhang717/TSWM-Official-Repo` | `2e5e48540ff0da48e8192163969a6c58df1d5a3e` | `RQ3_fusion_benchmark/fusion_train.py`、`RQ4_action_encoding/encode_train.py`、`RQ5_autonomy_split/autonomy_train.py`、`RQ7_causal_fusion/soft_train.py`。这里只核实路径，未审计全仓。 |
| `GoForIt-YYB/Meta-iLaD` | `0bc016a51b7ddb5f23b143a3fc01d2e34a1f92cd` | `README.md`、paper/poster；未发现实现源码。 |

正式归档仍应锁定 raw sources、执行 import/shape/sampling 资格检验。COFE `loss` 的 `weight_alpha` threshold、noise sparsity、sampling schedule 等应保留其原生行为，并单独记录必要兼容改动；不能只复制类名后宣称完整复现。

## Source-supported facts 与推断的边界

### RQ1：哪些能力确实存在

1. **观察预测 + covariate 使用已有广覆盖。** WorldTS 和 Pythia 都直接学习未来 predictive states，再接 observation readout；Pythia 正文还明确其 observational scope。不能要求这些论文完成未声称的 causal identification，也不能把它们改成共享 checkpoint 才算 world model。[WorldTS](https://arxiv.org/html/2609.31162v1)、[Pythia](https://arxiv.org/html/2610.05240)。
2. **概率 latent 不是空白。** Epic 的 sampled block laws、Branch 的 weighted atomic support、MDN-VJEPA 的 continuous mixture 是三种不同实现。Branch 当前还对 latent branching 与 output-only branching 做直接控制，所以“把多未来放进 latent 而非 decoder”也不能直接认领。[Epic](https://arxiv.org/html/2610.05996)、[Branch](https://arxiv.org/html/2607.05238v3)、[MDN](https://arxiv.org/html/2610.13956)。
3. **跨步 coherent uncertainty 已有直接方法。** MDN-VJEPA 不是只为每个 horizon 独立预测 GMM：它区分 shared trajectory mode 和 recursively composed kernel，明确用 distributional consistency 约束两条路径。component labels 并不保证全局语义，更不保证恢复真实 regime。[MDN Sec.4](https://arxiv.org/html/2610.13956)。
4. **“纠正状态再外推”和“慢环境/快状态”分别已有专门近邻。** OSSM 显式传播/观测校正分离；COFE 的 coarse/fine split 有创新噪声稀疏与独立性机制。不能只以新命名组合它们。[OSSM](https://arxiv.org/html/2609.33566)、[COFE](https://www.ijcai.org/proceedings/2026/0602.pdf)。

### RQ2：从上述事实能推到哪里

以下是检索者的**推断**，不是论文已经证明的负面结论。

| 待提动机 | 本次判定 | 要排除的最接近解释 |
|---|---|---|
| deterministic mean 不能描述分叉，所以改 flow/MDN | 已被直接覆盖；不成新 lead | Epic / Branch / MDN-VJEPA；经典 MDN latent dynamics。 |
| horizon marginals 好但整条轨迹乱，所以共享 mode 或加 CK loss | 已被直接覆盖 | MDN-VJEPA Level II/III；不能只以换 discrepancy 或换 TS 数据集再提。 |
| observed-history correction 和 autonomous rollout 应分开 | 已被直接覆盖 | OSSM；Kalman/SSM 与 RSSM。 |
| 在多尺度 latent 中分慢 regime / 快 dynamics | 仅此描述不够 | COFE、Koopa、Meta-iLaD，另需 switching LDS/HSMM。 |
| joint probabilistic query / missing-data 接口未统一 | 不能由 point forecast papers 的 omission 推出 gap | TACTiS-2，SSM filtering，CSDI 等；先证明不同 conditioning routes 出现真实不可被 joint model 修复的失败。 |
| 用 uncertainty 选择下一次测量 | 不能直接接受 | Epic 已做 exploration；SSM/POMDP information gain 更成熟。另检到 [Observable quotient world models](https://www.sciencedirect.com/science/article/pii/S0950705126014942) 的 primary publisher 摘要，已涉及 belief filtering 与 QIG；全文这次未成功读取，故不对其完整方法下结论。 |

组件已知不是自动拒绝。例如一个在非线性观测、未知 regimes、任意测量顺序下同时给出可更新 belief、coherent predictive law 和高效 rollout 的完整框架可能有价值。但**本次没有证据证明该完整组合超出了现有 probabilistic SSM / HSMM 能力，也没有看到明确不能由容量或 proper likelihood 修复的剩余失败。**

## RQ3：仅保留一个未晋级的具体排除问题

### Q-Duration：partial observability 下的 residual-life belief 是否是独立瓶颈？

**来源支持的切口：** MDN-VJEPA Sec.4.4 明确没有 persistent discrete-mode chain；其 direct trajectory mode 与 local transition mixture indicator 是不同对象。[原文](https://arxiv.org/html/2610.13956)。这不意味着它不能通过 continuous state/history 学会持续时间。因此不能声称“MDN-VJEPA 结构上不可能建模 dwell time”。

**候选机制（推断，未接受）：** 维护 `q(regime, elapsed_age, continuous_state | history)`，由 age-dependent hazard 控制 renewal；共享 encoder 给多种观测提供 likelihood messages；在 direct horizon law 和递归 rollout 间保留 residual-life uncertainty，避免每次 horizon 查询隐式重置 episode age。这比“再加一个 context latent”更具体，但本质上很接近 neural HSMM / semi-Markov switching SSM。

**所需完整增量：** 必须证明该机制在非线性、部分可观测、多模态 measurements 下，以少得多的有效 context/compute 维持 duration law、跨 observation update 的 age belief 和跨 horizon sampling 一致性；仅在一个 HSMM toy 上打败无 duration 的 GMM 不够。若 oracle latent 或训练强模型已经足够，剩余贡献最多是普通结构先验，而不是新 world-model framework。

**精确假设：** regimes 确实具有非几何 dwell hazards；早期观测含有关进入时刻的信息；历史/采样 masks 完全相同；无 regime-age labels 提供给候选而不提供给 baselines；不假设恢复的 regime 等于物理机制；train/test 的 sensor noise、duration ranges 与 heldout 参数分区预先冻结。若历史根本不含 age 信息，所有方法都应保持相应 posterior uncertainty，而非“恢复真年龄”。

**最强近邻/必需 controls：** oracle-parameter HSMM filtering（数值上限）、learned HSMM/SLDS、同 latent dimension 的 RSSM、history-conditioned mixture with shared trajectory modes、output-only GMM/copula，以及相同 window 的 ordinary AR/NARX。新增 state 数量必须做等容量 control。非线性 sensor 情况必须给 HSMM 同 encoder 或给出应有 observation likelihood，不能只用 misspecified raw Gaussian HMM 垫底。RC-Mamba 已把 soft HMM posteriors 用于 neural recurrence；须加此类软 regime-conditioning control，不能只有 hard labels。

**实际可读代码基础：** author `lindermanlab/ssm` 的 HSMM/SLDS 作 classical control；`YongchaoHuang/VJEPA` 的 probabilistic/Bayesian-fusion/RSSM 示例作最小神经起点；COFE 的 ProbTS loader/forecaster 训练与多个生成 baseline 作更广 TS 底座。上述都只是已核实源码/接口，不宣称现环境已能运行。无需任何大型 pretraining 或 LLM。

**5090 上的一次决定性反证，建议先做 CPU/小神经排除而非扩训：** 用 3 regimes、2D continuous dynamics、geometric vs two non-geometric duration families 生成多 episodes；相同 32/128 history，各自保留 state dimension、参数数与 validation budget 对照。训练仅用部分 duration parameters，独立 heldout 使用未见参数和 nonlinear sensor。主要 endpoint 是未来 renewal event 的 Brier/log score 与 residual-life predictive CDF；辅助是 trajectory energy score。把同一 history 分成不同 chunk 进入模型，重复同一 future-horizon query，检查是否改变 predictive law。以 learned HSMM、matched RSSM、direct block mixture 的最佳 validation-selected 方法为基线。**若 learned HSMM/RSSM 已在误差容许范围内匹配候选，或优势只来自显式 age/supervised regime / 更长 history / 更多容量，就直接判死。** 不以修正 point MSE 作为成功判据。

**为什么目前不推荐它：** 这一设计只是把问题暴露得更清楚；它没有排除成熟 semi-Markov modeling。尚无值得把整个项目押上去的 distinct method evidence。根任务若需要压缩后续行动，应优先吸收本轮排除边界，而不是为了方向数把此问题升为正面 lead。

## 给根任务的实用结果

- 必须更新近邻表：MoP-JEPA -> 当前 Branch-JEPA v3，增加 full-set proper-score regime 与 output-only branching 对照。
- 若未来另提 stochastic/probabilistic world model，MDN-VJEPA 的 trajectory coupling 与 CK objective 是最接近直接先例之一，不能只列 deterministic JEPA。
- COFE 有真实实现和多概率 baseline configs；可作为证据丰富的底座，但尚未运行资格检验。
- 本视角无通过门槛的新框架；不建议基于此笔记直接新增 GPU 训练任务。
