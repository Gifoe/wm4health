<!-- cycle31-current-start -->
# 最新 cycle31：按用户纠正研究范围

UTC 2026-10-16T19:51:23.870858+00:00。通用时间序列 world-model 框架，以 WorldTS 为研究层次参照；不强制零样本共享 checkpoint、因果/物理解释或某种结构。NormWear-2 原生预测执行、三处条件接口诊断和普通修复完成，独立复核通过；不是新方法或论文性能复现。接受方向0，整体 ACTIVE_UNACHIEVED。

见[范围纠正](cycle31/SCOPE_CORRECTION_20261017.md)、[实际证据](cycle31/experimental_results.md)与[当前研究判断](cycle31/final_research_direction.md)。以下各轮历史保留，其“当前”仅指当时。
<!-- cycle31-current-end -->

<!-- cycle29-index-start -->
# 最新证据cycle29

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/literature_review.md](cycle29/literature_review.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/literature_review.md](cycle28/literature_review.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

# 文献、代码与创新边界（2026-10-16）

检索重点为 2026-04-16 至 2026-10-16；保留更早的不可识别性、系统辨识、可达性基础工作。以下是定向检索，不是系统综述或“没有找到即不存在”的证明。公开代码快照见 `outputs/source_registry.json`。论文证据、代码证据与本轮实验分别标明。

## 起点与排除范围

| 工作 | 已核查的内容 | 对当前判断的约束 |
|---|---|---|
| [LatentTSF](https://arxiv.org/abs/2602.00297) / [代码](https://github.com/Muyiiiii/LatentTSF) | 论文、README、AE/latent predictor 代码；旧复现报告 | AE 重建/潜空间预测本身不是新研究问题；本轮未重训 |
| [WorldTS](https://arxiv.org/abs/2609.31162) / [代码](https://github.com/decisionintelligence/WorldTS) | 数值 core、训练阶段、真实旧 checkpoint；本轮执行 | 本版本未来条件只经局部 encoder 与 decoder；该事实不能推广到所有 world model |
| [Accuracy and Mechanism Consistency](https://arxiv.org/abs/2610.01842) / [代码](https://github.com/HaochenZhang717/TSWM-Official-Repo) | 论文全文/实现；标题现为 On the Divergence of Accuracy and Mechanism Consistency in Time Series World Models | 已研究准确性与机制方向脱钩、方向监督、不同 plan encoder；不能把这一大命题重新命名为创新。代码默认 rho 与论文设置须显式对齐 |
| [RP-GSSM](https://arxiv.org/abs/2505.23569) / [代码](https://github.com/gatsby-sahani/rp-gssm) | 论文/README/JAX 实现 | 已研究不依赖重建的 latent dynamics；本轮未执行 JAX 训练 |
| [PRO-DYN](https://arxiv.org/abs/2507.15774) / [代码](https://github.com/ARBrachet/PRO-DYN) | 论文与代码结构 | 动态结构本身已有路线；未执行新复现 |
| [DoTime](https://arxiv.org/abs/2607.27263) / [代码](https://github.com/thummd/dotime) | prior、pairing、evaluation、PFN masking、suite 版本 | 默认 `generate_pair` 为 independent-noise interventional；显式 counterfactual 才共享噪声。新版 suite 已区分 level/effect sign，不能声称其未区分 |
| [EpiCF-Bench](https://github.com/complex-ai-lab/epi-cf-benchmark) | README/tree/数据协议 | calibrated ABM 的模拟反事实有明确结构真值；本轮未下载大型数据，未作排行榜比较 |
| [TIME](https://arxiv.org/abs/2602.12147) / [代码](https://github.com/zqiao11/TIME) | 论文/README/tree | zero-shot forecasting 不是干预或个体反事实真值 |

## 已直接淘汰的宽泛 Idea

| 先前可能的命题 | 直接近邻 | 为什么不能作创新主张 |
|---|---|---|
| 给条件预测器增加动作记忆 | [BeliefGraph-JEPA](https://arxiv.org/abs/2610.05409)；[Attenuated in-context identification](https://arxiv.org/abs/2610.08118) | 前者有持续的 latent action effect；后者已做动态响应诊断、激励对照、forced-system 微调及 ARX/N4SID/NARX 对照。本轮 FIR/GRU/ARX 也已修复 |
| 动作激励不足导致识别失败 | [On Identifiability of Controlled World Models](https://arxiv.org/abs/2607.22430) | 已有 excitation/identifiability 理论。白噪声神经网络 factorial 本轮未跑，不能把它写成验证结果 |
| 因果未来动作 mask | [Parallel Predictive World Models](https://arxiv.org/abs/2610.08627) | action-prefix parallel prediction 已有 |
| belief 而不是确定性 latent | [UWM-JEPA](https://arxiv.org/abs/2605.25313)、RP-GSSM、[MDN-VJEPA](https://arxiv.org/abs/2610.13956) | hidden-state uncertainty、multimodal/coherent rollouts 已有；后者提交日期应以原页面为准，本轮不依赖搜索摘要的异常日期 |
| 分离观测校正与自主传播 | [OSSM / Correct then Forecast](https://arxiv.org/abs/2609.33566) | observer / simulator 解耦已有 |
| 物理加残差潜状态 | [Phys-JEPA](https://arxiv.org/abs/2606.16076) | 不应重新包装为新结构 |
| 非平稳 Koopman / coarse-fine 潜动态 | [KUMA](https://proceedings.mlr.press/v306/xiong26j.html)、[COFE](https://www.ijcai.org/proceedings/2026/602) | 必须区分已有非平稳与多尺度建模 |
| 预测分布随新观测形成 martingale | [Calibrated Probability Forecast Sequences](https://arxiv.org/abs/2606.31621)、[MPNN](https://arxiv.org/abs/2506.11898) | forecast-revision consistency 已有，不是新原则 |

[Pythia](https://arxiv.org/abs/2610.05240) 与 [Meta-iLaD](https://github.com/GoForIt-YYB/Meta-iLaD) 也已筛查；后者所查公开仓库只有论文/海报而无可运行训练代码，不能写作已复现。综述条目深度不同：只有参与决策的近邻作了机制级阅读；不是所有论文都逐式复推。

## 当前反事实区间问题的最近竞争者

1. [CoMP, ICML 2022](https://proceedings.mlr.press/v162/foster22a.html) 已明确用条件相关 Gaussian latent rotation 展示相同拟合分布、不同反事实。**本轮旋转反例不是新发现**。代码固定快照已保存。
2. [Counterfactual Analysis in Dynamic Latent State Models, ICML 2023](https://proceedings.mlr.press/v202/haugh23a.html) 已作动态隐藏状态部分识别与优化界；不能宣称“首次动态反事实界”。
3. [CSM/APID, NeurIPS 2023](https://arxiv.org/abs/2306.01424) / [代码](https://github.com/Valentyn1997/CSM-APID) 已对连续结果部分识别，利用 level-set curvature 与生成模型拟合。本文主要处理二元 treatment/连续 outcome 的静态查询；代码 curvature 实现断言 dim_u=2。零曲率也不能随意解释成唯一点识别。它不是现成的冻结序列模型置信区间接口。
4. [Robust Counterfactual Inference in MDPs](https://arxiv.org/abs/2502.13731) 已对 compatible SCM 的离散动态反事实给界。
5. [Lally et al., 2026-10-14](https://arxiv.org/abs/2610.20210) / [代码](https://github.com/jessicalally/robust-cf-inference-beyond-mdps) 已联合处理 transition 统计不确定性与 causal ambiguity，构造 IMDP 并作悲观 value iteration。**联合处理两类不确定性不能作为我们的首创**。论文定义离散 state space；本轮源码矩阵/Julia IMDP 路线也吻合。未执行其完整实验，未声称超越其结果。
6. [Twin Rollouts](https://arxiv.org/abs/2608.08982) 已有共享噪声的视频分支；共享 seed 本身不是创新。
7. [Flow*](https://github.com/chenxin415/flowstar) 已有 Taylor-model flowpipe、symbolic remainder 与参数依赖的可达性计算；本轮 first-order + remainder 是这些标准思想的特例实现，**不是新验证理论**。尚未完成与 Flow*/CORA 之类成熟工具的预算匹配比较。
8. [Non-rectangular robust MDPs](https://arxiv.org/abs/2305.19004)、[其 Lp 对偶方法](https://arxiv.org/abs/2502.09432) 已研究共享模型不确定性与 rectangular approximation 的困难；参数跨时共享本身也不是创新。

## 剩余问题与创新置信度

尚值得检查的具体交集是：**对冻结的连续、非线性动态模型，在明确的噪声耦合敏感性类中，同时保留 factual abduction、跨分支/跨时间共享参数，给出既有覆盖保证又能支持非平凡效应判断的区间。**

相对所核查公开实现，尚未找到能直接完成该接口的现成方法。但“没有找到”不等于证明不存在；通用可达性工具可能已经足够。当前问题证据强于方法创新证据。不得宣称首次、不应据此直接立项写顶会论文。下一轮必须把通用可达性/静态 APID 的动态组合当作竞争解释，而不是只和松的 Lipschitz 界比较。


新增强近邻：[auto_LiRPA / CROWN](https://github.com/Verified-Intelligence/auto_LiRPA) 已支持一般计算图、双变量乘法与sin/cos。代码固定后执行了同一共享L2置信球和连续角度集合的计算图验证；它不是只供文字讨论的近邻。当前专用Taylor原型不能凭参数共享或微小区间改善宣称新颖。Flow*/CORA仍未执行，但当前原型不需要等待更多对照才能停止推进。


附加创新排除：同一Gaussian kernel的旋转改变pathwise-gradient方差并非新原理。[ICML2018](https://proceedings.mlr.press/v80/jankowiak18a.html)和[AISTATS2019](https://proceedings.mlr.press/v89/jankowiak19a.html)已研究transport-equation的旋转自由度、控制变量与方差优化。本轮只作文献筛选，没有虚报planning-gradient新实验。
