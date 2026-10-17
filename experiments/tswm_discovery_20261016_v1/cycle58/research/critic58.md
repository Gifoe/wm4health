# C58 独立 critic：旧反证的有效范围与近期 JEPA 证据

核查日期：2026-10-18。视角：反方与方法学。仅只读本地协议/报告和公开一手论文、作者仓库；未访问 SSH、服务器结果或 C57 heldout，未训练、改环境或继续委派。检索不是覆盖整个领域的系统综述。

## 摘要

这次小样本审计**没有产生可以晋升的新方法路线**。旧记录有值得纠正的解释边界，尤其 C41 的 known-law Bayesian permutation control 不能当作未知模型学习已经解决的证据；但后续 C46 的同输入、train-only 普通对齐/LGSSM 已真正学会了该 carrier，故纠正这一点并不足以复活原路线。C20 的成熟压缩器共享候选的输出选择成分，也不能证明目标选择早被完整解决；不过 C21 独立机制上的不稳定结果仍然有效。

近期一手文献同时阻止两个错误方向：不能把“不能恢复物理状态/没有因果保证”用作所有 world model 的淘汰标准；也不能把随机 latent、逆动力学、轨迹生成或 observation/dynamics 分离的存在直接包装为新框架。C57 的有限 reference 优化问题可以继续按已冻结合同判别，但目前不能把它扩写成一般 world-model gap。

## 1. 问题与核查方法

本笔记回答三个问题：旧淘汰是否使用了超出任务的信息或要求？最新相关工作实际解决了什么？是否有剩余问题能给出公平、便宜、决定性的反证？

先读 C58 brief 与 C31 scope correction，再对 C20–22、C33–35、C41、C54–57 的指定报告作针对性审计。只有发现 C41 的推理边界可能影响结论时，才补读 C46 的直接后续结果；未遍历旧 experiment outputs。公开检索先核验题名/作者/版本，再看原文方法、限制及官方实现。搜索返回的 Pith/AI review 未用作证据。

## 2. 旧反证审计

| 记录 | 有效负证据 | 必须限制的解释 | 本轮处置 |
|---|---|---|---|
| C31 `SCOPE_CORRECTION_20261017.md` | 用户要求可跨系统复用的建模贡献，不能用单数据集调参/实现修复完成任务。 | 用户没有要求统一零样本 checkpoint、物理状态可解释、严格因果反事实或必含 action。缺少这些性质不构成自动失败。 | 此范围纠正继续有效，不额外收紧。 |
| C20 `ACTIVE_SUBSPACE_CONTROL.md`、`negative_evidence.md` | 同数据 teacher + 普通 sensitivity compression 能修复紧状态上的 gain/响应信息。更宽 latent 与 raw-observation predictor 也能修复，故“所有模型必丢状态”不成立。 | action-response output span 是共享候选成分，并非一个完全独立的现成完整算法；结果只能否定新神经 compressor 的必要性，不能单独否定 target selection 的价值。physical readout 是诊断特权；它不等于可部署无标签解码。 | 不恢复旧架构主张；保留 target-selection novelty 当时未被 classical control 单独判决这一边界。 |
| C21 `negative_evidence.md`、`ACTIVE_SUBSPACE_CONTROL.md` | projected-bank 跨独立机制未稳定保留 gain；response-only 在 queueing 被普通 obs predictor 超过。没有跨机制统一收益。 | “动作无关”不等于对完整观察预测无关。若目标是全部未来观测，AR 信号是合法信息；特权物理输出不能替代任务。成熟压缩器仍共享选择步骤。 | 实际负结果保留；不能从其中归纳通用 state information 不可能性。 |
| C22 `negative_evidence.md` | 原生 WorldTS 确实使用 future-state prediction；ridge、条件 GRU 已是可行 ordinary 修复。 | future truth latent readout 是 oracle；关 RevIN/降 token 维同时改变难度和信息合同。未收敛训练不能当结构下界。 | 关闭该数据集上的旧 target-information 动机有依据；不据此认证所有 latent 都充分。 |
| C33 `negative_evidence.md` | 冻结 stage1 后普通 global MLP 修复一部分残差；原始“latent 已丢信息”归因被削弱。 | global 同时改变参数量、非局部聚合和 RevIN statistics 使用；不是纯 mechanism ablation。NP/PJM 不能外推全部系统。 | 保留关闭旧 readout 线索的开发决定。没有理由重新在这两个任务调该故事。 |
| C34、C35 `negative_evidence.md` | C34 的具体 proposal 等价于 ordinary conditional MLE；C35 具体保护组合未过预列双任务 screen。 | “组件已存在”不是所有组合框架无价值的证明；但 C34 的具体等价足以否定当时唯一差异。C35 单 seed、小预算只判当前实现，不判所有 latent innovations。 | 保留具体否决；不把 finite screen 当普遍不可行定理。 |
| C41 `CONTROLLED_PLAN.md`、`experimental_results.md` | 无锚的跨 view 对齐可不可辨识；给足信息时 ordinary Bayesian marginalization 可解决相应有限 alignment 问题。 | control 明写 known source transitions、noiseless named symbols 是 privileged。它不证明未知 emission/transition 的有限数据联合学习已解决；exact model feasibility 与 learned algorithm quality 不能混用。 | 纠正任何更强的“C41 已完全解决学习”的表述。 |
| 对 C41 的必要后查：C46 `experimental_results.md`、`negative_evidence.md` | 八个 fresh neural ordinary-composition fits 后，train-only PCA + 同样 anchors 对齐 + fitted VAR/LGSSM 的 held-out ES/MSE 优于这些 neural arms。该 control 不用真 state 拟合。 | O1 同时限制 recognition 与 emission，不能孤立归因；这个 carrier 也不是任意 nonlinear sensor 的结论。 | 不能因 C41 的 oracle 边界而重启同一个组合 toy。已有真实 learned ordinary repair，尚缺新的必要交互。 |
| C54–57 指定报告 | ordinary corrected EM 在 C54 的线性 carrier 接近 oracle；C55 理想 scalar Gaussian population objective 在真参数可达 tight optimum。 | full-reference bank 是成本较高的 train-data control，并非 hidden-state oracle，也不是低成本方法。C55 不证明 finite-reference/nonlinear optimization 无问题。C57 unshifted running adaptation 的失败不能写成原始 SogCLR 失败。 | 等 C57 冻结比较结果。现阶段不能宣称不可避免的 population inconsistency，也不能提前判已有 normalizer 成败。 |

这些记录大多已在自己的 limitation 中写出上述边界。主要风险是把局部开发决定汇总成一个比原报告更强的“该问题全解决/所有相关方向都死了”结论。反向错误同样不可取：记录保留 caveat，不等于没有有效 negative。

## 3. 当前一手文献：它们允许和禁止什么结论

### 3.1 Controlled identifiability 不是通用门槛

[Zhang et al., *On the Identifiability of Controlled World Models*, 2607.22430v2](https://arxiv.org/html/2607.22430v2)，§3.1–3.4：其充分条件限定 stationary linear-Gaussian latent dynamics、invertible observations、exact standard-Gaussian encoder constraint、足够丰富的 predictor。representation spectral separation 与 conditional action excitation 是不同条件。Theorem 3 给出可达到的 inverse-excitation error amplification 构造，不是所有 predictor 的统一 counterfactual error upper bound。

所以不能要求任何 observational time-series forecaster 都满足此理论的 state recovery；也不能因论文已有 joint identifiability theorem 就宣称 partial observation、非 Gaussian 或有限样本问题已解决。去掉其假设会得到大量问题，但“放宽 Gaussian 假设”本身还不是有方法、有竞争解、有可证伪实验的路线。

### 3.2 Inverse-action objective 的盲区已被作者明确看到

[Boylan & Hokamp, *No Gaussian Required*, 2608.17542v1](https://arxiv.org/html/2608.17542v1)，§4.5、§7、Appendix B/D：training-only inverse-action contrast 可代替固定 latent marginal matching，但作者明确承认弱受控而任务重要的变量可能被低估；PushT 是失败例子。Scene 的匹配 trajectory-goal 协议不能冒充 official fixed-goal leaderboard；作者也报告后者两法均失败。其 [官方仓库](https://github.com/jackboyla/action-contrastive-jepa) 提供约 15M 参数、单 GPU 的执行底座。

这给 C20 的“inverse 可成功而遗漏响应相关环境因子”提供了新的实际近邻，却不是自动的新 gap：SIGReg、full-observation learning、更多容量与 C20 的普通 response-target compressor 都是必须正面排除的补救。不能只比较一个容易丢弱受控变量的 inverse baseline。

### 3.3 一步指标与 planning 的不一致不能被当作新机制

[Cheung & He, *One-Step Prediction Fidelity Is a Poor Proxy for Planning Performance in JEPA World Models*, 2610.19375v1](https://arxiv.org/html/2610.19375v1)，§3、§8–9：frozen-decoder one-step physical error 与 planning 可产生不稳定关系；matched continued-training 吸收了被归因于 auxiliary objective 的收益。作者明确限定单 architecture、单 planner、三个模拟环境、warm-start 邻域；没有识别导致不一致的机制，rollout-level 验证仍未做。

该结果强化 C33–35 的普通继续训练/读出控制要求。它不证明预测 fidelity 无用，不允许把非显著 planning 差异写成严格零效应，也不单独支持一个新 training loss。paper 的 physical decoder 仍是诊断工具，不是无标签部署能力。

### 3.4 “让 JEPA 变随机”已经有多个完整近邻

| 一手来源 | 已有能力 | 不能由此代替的证据 |
|---|---|---|
| [Huang, VJEPA, ICML 2026](https://proceedings.mlr.press/v306/huang26ba.html) | variational predictive latent distributions；与 predictive-state/filtering 的联系；modular prior experts。 | 不能再以随机 latent 或 Product of Experts 本身宣称创新；也不能由摘要断言已解决所有 task-sufficient state learning。 |
| [Song et al., MoP-JEPA, 2607.05238v1](https://arxiv.org/html/2607.05238v1)，Method、Verification Protocol、Limitations | hard-assigned multi-head successors、router、context-shuffle/codebook/transition-precision controls；明认其 loss 来自经典 multiple-choice learning。 | mode support/graph reachability 不是 complete conditional trajectory law 的认证。但作者任务是 mode enumeration，不能强加 calibrated-law 要求后宣布其失败；若改任务，必须引入强概率对照。 |
| [Huo et al., Flow-JEPA, 2608.29029v1](https://arxiv.org/html/2608.29029v1)，Method、Appendix inference analysis | action-conditioned joint future-latent trajectory flow；报告噪声 robustness，并明确披露 iterative inference 更贵。 | “联合生成未来轨迹”不是空白；当前论文结果也不自动覆盖所有时间序列系统的 proper-score calibration。 |

这里存在能力边界，而非一个自然获准的新方法。把模式遗漏、概率校准、planning exploitation、因果 counterfactual 混成一个失败再加 stochastic head，会混淆四个任务合同。

## 4. 条件性判别与 multi-horizon control 的 go/no-go

**当前不晋升路线。** 首先完成的具体问题是已经在跑的 C57：在该模型族中，低成本、competently centered 的 ordinary normalizer estimator 是否消除 native-to-proper 的 changed-plan mean-risk 差距？这属于现有机制的必要性审计，不是新增候选。C57 的最终结果由 root 核查，本 perspective 不推断结果。

其最强成熟竞争者是两类：一是 [SogCLR 的 global contrastive optimization](https://proceedings.mlr.press/v162/yuan22b.html) 原则及更大/独立/full reference；二是 proper likelihood 的 [Normalizing Kalman Filter](https://papers.nips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html) 类方案。后者是当 invertible sensor + Gaussian dynamics 假设成立时的完整强解。RP-GSSM 本身也已是 [maximum-likelihood latent dynamics without reconstruction](https://proceedings.mlr.press/v306/hromadka26a.html) 框架，不能把“免重建地学习动力学”另取名字。

**现有假设**：已覆盖的 linear controlled latent、指定 linear/nonlinear sensor family、相同可用 prefix/plan、固定优化和 reference 预算。Gaussian pair score 的 unbounded 性质不自动满足 image SogCLR 的 bounded-similarity 条件，故需要 competent adaptation；其失败也不能独立证明新估计器必要。

**确切缺证**：C57 frozen heldout 风险（本笔记未访问）；reference bias 与 tracking/optimization 的可分辨归因；不是仅固定 checkpoint 梯度，而是 trained observable multi-step task 的 residual；随后才可能谈非可逆/高维 sensor 上相同问题是否仍在。C55 已排除的 scalar 理想 population-bias 故事不能回来充当动机。

**最小公平 falsifier**：直接执行已冻结 `DECISION_RULE_V1.md`，不新增试验点、loss 或阈值。任何已有小批 normalizer control 按该规则修复 nonlinear task 且不损 linear task，即关闭“需要新 correction”的必要性。若只有 full-reference 修复，只留下一个 cost-limited optimization 问题；若所有 controls 都失败，也先检查 tracking 与收敛，再决定是否有残余。proper control 若也无 actionable gap，就不再计算有误导性的 closure ratio。

这关通过或失败都不能自动晋升为一般框架贡献。若下一步没有新机制，只是调 running mean、kernel shift、batch size 或同一 toy 的训练长度，应停止该包装。

### 4.1 Multi-horizon control residual：只值得一个有前提的诊断，不值得立即新训练

**严格判断**：仅凭 2610.19375 和旧 toy 的负结果，**不值得启动一个新方法训练 campaign**。“一步误差小、长期控制差”至少可由训练/部署 action distribution 不同、累积误差、读出外插、成本几何与 planner 搜索失败分别产生。当前没有隔离出其中一个仍未被 ordinary 方法覆盖的 residual。多步 objective 本身至少已有 [PlaNet 的 latent overshooting](https://proceedings.mlr.press/v97/hafner19a.html)，近期还有上表的 Flow-JEPA；不能把 horizon loss 或 joint rollout 当新机制。

若已有原生 checkpoint 与 evaluator 可直接复用，可以批准**一次 frozen-checkpoint、无新训练的 go/no-go diagnostic**。它的目的只是决定是否值得设计方法：

1. 固定两个原生 benchmark 环境及其未用来挑选假设的 evaluation episodes，例如公开 LeWM/AC-MTM 的 PushT 与 TwoRoom；不得挑事后出现最大反差的 horizon。预先固定 h={1,5,10}，所有方法使用同 prefix、同受支持 action plans、同可用历史；观测和动作支持不足的 arbitrary off-policy plans 不算算法失败。
2. 用冻结模型 open-loop 展开同一 action plan，比较每个 horizon 的可观测任务误差。physical probes 只准 train-only 拟合、只作诊断；同时报告未来 encoder embedding 误差及多个合理 readout 的敏感性，避免把一个 decoder 的 off-manifold 外插误差认作 dynamics error。允许强 readout 修复；不保护旧解释。
3. 对同一批 plans，使用 simulator true dynamics 作**特权诊断**，运行完全相同 solver/budget。它只回答这个 planner 在此预算下是否本来能完成任务，不能当 learned baseline。如果 oracle planner 也失败，先关闭“模型多步误差造成控制失败”的当前归因。再用预列的更充分 solver budget 检查结论是否只是搜索预算问题。
4. 若模型在 h>1 有真实 observable residual，且真 dynamics 能消除控制损失，才获得“值得一个 learning experiment”的资格。下一实验的首要竞争解是同输入、同预算的 ordinary multi-step training/overshooting 或强直接 trajectory predictor，并保留原 loss 的 matched continued-training arm。不是先创造新 loss 再找弱 one-step baseline。选择 Flow-JEPA 时必须计入 flow steps 与 end-to-end planning cost，不能仅匹配参数数。

**立即停止条件**：真实观测任务上没有 h>1 residual；改合理 readout 即消失；oracle 同 solver 也失败；支持不足解释主要差异；更充分 solver 消除 control 差；或者后续 ordinary multi-step/trajectory control 吸收全部收益。反过来，这些项都没有解释掉差异，才留下“representation × rollout × planner interaction”的有限问题；它仍不是新方法或普遍框架 gap。

此诊断未执行，所需 checkpoint/evaluator/有效分割也未在本 perspective 做资格认证。若无法低成本复用，当前证据不足以仅为这个 broad residual 开始下载/重训。它是方法选择前的 stop/go，不是另一项已成立的候选路线。

### 4.2 同一 toy 的 classical repair 不能替代的框架结论

classic control 成功确实反驳“该 toy 的 deficit 必需新架构/状态信息不可恢复”。但它不能代替以下证据：未知高维或非可逆 sensor 上的学习效率；跨系统保持同一训练规则的稳定性；无 privileged state 的实际部署；模型不确定性的 joint trajectory calibration；同等训练/推断成本下的优势；或某个具体新 learning interaction 相对完整普通组合的必要性。C54 linear EM、C57 invertible-sensor likelihood 以及 C46 PCA 对齐的适用假设也都必须保留。

不能从“一个 toy 被 classical 修复”推出“world-model framework 层面再无问题”；同样不能因为它没有解决任意高维系统，就宣称已经找到了缺失机制。下一步必须给出 classical 假设在哪个真实可复用场景失效、哪种强现成非线性/概率方法仍不足，以及一个具体机制为何利用了现有解没有利用的结构。当前 critic 没有找到完成这三项的证据。

## 5. 对三个 research questions 的回答

**RQ1**：当前框架已分别覆盖条件未来表示、联合状态推断、随机 latent、模式枚举和联合轨迹生成；没有证据允许把未声称因果/物理恢复能力的 observational framework 因此判败。各框架任务不同，也不能把这些能力的并集写成某单一已完成通用解。

**RQ2**：审计确实找到 privilege 与归因边界，但没有找到足以撤销关键 negative 的依据。C41 known-law oracle 不是 learned closure；C46 的 learned ordinary repair 却仍否定同 carrier 上缺一个全新模块化机制的必要性。C20 shared target-selection control 不是完整先例；C21 独立负结果却继续压低原选择规则。

**RQ3**：本 critic perspective 没有剩余证据充分的 salvage/new-method route。建议 root 把其他视角的新 lead 与本笔记的强普通解逐项相交；没有 surviving residual 就明确保留接受方向数 0，而非把 scope correction 当作重新开放所有旧 toy 的许可证。C57 完成后只能按其实际 residual 决定后续，不先写一个框架故事。

## 一手来源核验说明

上述 controlled-identifiability、AC-MTM、one-step-fidelity 三篇均核对 arXiv abstract metadata 与 HTML 原文；stochastic JEPA 的三篇分别核对 PMLR/原始 arXiv。AC-MTM 检查作者链接仓库，仅确认代码基础存在，未执行。SogCLR、NKF、NCDSSM、RP-GSSM 核对官方 proceedings；其中 NCDSSM 支持 recognition/dynamics 分离已是既有机制，未据 abstract 擅自推断未检验泛化能力。未确认的第三方 Var-JEPA 命名/作者混淆未用于结论。此次无任何新的效果测量、数值接受概率或候选训练结果。
