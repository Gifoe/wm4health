# C41 主流证据：learned predictive state 究竟增加了什么

检索与源代码核对：2026-10-17。独立主流视角；只读取共享 research brief，未读取任何 sibling note。本文是研究判断，不是论文性能复现。

## 摘要

当前完整框架支持三种不同的收益：改变预测任务的归纳偏置；将未来条件分布组织成可递推的摘要；在部分观测下摊销过滤、预测和补全。前两者不能仅凭 latent 可视化、MSE 改善或直接／组合预测一致性相互推导。最具体的证据缺口是：现代 probabilistic JEPA 的未来分布建模收益，是否仍成立于**同时学习历史状态与推断器**之后，以及改善来自状态保留的信息、转移模型还是推断近似。本文提出一个可精确分解这些量的小型检验；它通过之前，不应宣布新的 state-learning 框架。

## 问题与检索方法

冻结问题沿用 brief：RQ1，哪些任务和条件下完整方法证明状态有用，证据缺在哪里；RQ2，如何分开 state information、transition approximation 和 inference error；RQ3，面对最近完整方法，有什么值得单卡验证的具体交互，或应改用什么研究表述。

使用公开关键词检索 WorldTS、latent time-series forecasting、predictive-state inference、probabilistic JEPA，并沿主论文追踪到 MDN-VJEPA。以下保留六个决定性完整框架。标题／作者通过出版社或 arXiv 元数据验证；机制与实验限制核对正文。事实标为 **[来源]**，分析标为 **[推断]**，待检验建议标为 **[假设]**。未执行训练、未下载数据或权重、未访问或修改服务器。

## 按模型实际解决的问题比较完整框架

| 完整方法 | 实际训练和使用链条 [来源] | 对“状态有用”的证据及边界 |
|---|---|---|
| [WorldTS](https://arxiv.org/html/2609.31162v1) | causal local patches 编码历史与真实未来；协变量参与未来 latent trajectory 预测；latent／cosine／VICReg 联合训练后冻结，训练观测 decoder | 多数据集预测及 matched one-stage、pointwise encoder 等消融支持整条训练链。它不是已经证明充分性的递归 filter；Appendix C 明说协变量信息、状态质量与预测误差的理论联系仍开放。 |
| [LatentTSF](https://proceedings.mlr.press/v306/yang26y.html) | 先训练 pointwise AE，冻结后让通用 backbone 预测未来 latent，再解码；目标为 latent regression + alignment | 多 backbone 的预测改善支持可移植训练范式。单时刻 AE 并未直接恢复历史 belief；其扩维本身也不是新增信息。 |
| [PSIM](https://proceedings.mlr.press/v48/sun16.pdf) | 用未来观测特征的条件期望表示状态，递归接受新观测；DAgger 在模型自己遇到的状态上训练更新器 | 在 k-observability、特征足够表达未来 law 等条件下有过滤保证；实验比较 AR／谱方法，并检查未来窗口和二阶特征。它已经覆盖“future statistics + learned recurrent update”的完整组合。 |
| [VJEPA / BJEPA](https://proceedings.mlr.press/v306/huang26ba.html) | latent-space variational prediction；可将动态 expert 与 prior expert 组合，并用于预测／规划 | nuisance robustness 与概率预测构成正面证据；但 predictive sufficiency 是有条件的要求。它的充分性结论不等于一般训练过程自动找到了充分状态。 |
| [MDN-VJEPA](https://arxiv.org/html/2610.13956v1) | latent MDN、共享 trajectory modes、direct/composed distribution consistency，加上表示学习目标的完整公式 | 控制实验支持未来分布、轨迹耦合和组合一致性的各自作用。§8.6 明确：Exp1/2/4 直接使用 predictive latent states，Exp3 使用固定表示；这些实验未验证 learned representation 的收益。 |
| [LIFTS](https://arxiv.org/html/2610.03916v1) | latent process 联合生成 measurement 与 mask；filter/forecast + smoother/imputation 联合训练，masked energy + selfmask | 覆盖多种部分观测推断任务；full-law identification 只在明确结构子类及假设下成立。§6.1 的 synthetic encoder calibration 未带来系统改善，后续设 lambda_h=0，故不能把增加 latent calibration 直接当作有效机制。 |

上述六项不是 ingredient 清单：WorldTS／LatentTSF 改变目标与训练顺序；PSIM 学习可递推未来摘要；VJEPA／MDN-VJEPA 学习并使用未来 latent law；LIFTS 联合学习生成机制和多种条件推断。比较新方案时必须对照这些完整训练与推断链。

### 源代码核对：LatentTSF 的“扩维状态”具体是什么

**[来源]** 官方源码当前 commit 为 `7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2`。[my_AE.py:17–48](https://github.com/Muyiiiii/LatentTSF/blob/7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2/my_AE.py#L17-L48) 的主 MLP AE 仅沿 feature dimension 做两层 Linear/ReLU；[my_train.py:312–347](https://github.com/Muyiiiii/LatentTSF/blob/7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2/my_train.py#L312-L347) 实现冻结 AE 和 latent dimension 的 backbone 接口。只读源码，未运行。

**[推断]** 对固定已学 encoder，Z=E(X) 是当前观测的确定函数；扩到更高维不会增加关于隐藏过程的观测信息。这并不否定该方法：backbone 仍能从历史序列学习预测；非线性坐标、监督尺度、预训练和优化都可能改善有限资源预测。它只否定“扩维直接补回未观测状态信息”这一额外解释。

## RQ1：哪些“增加”有依据

**[推断]** 如果 ordinary forecasting 已表示完整的 p(Y_future | history)，predictive state 不会在信息意义上超过这个条件 law。它的潜在价值是用较小、可重复使用、易学习的摘要承载同一项预测信息，或使多个查询共用状态与转移。这是有限数据、内存、计算下的可检验优势，无须证明无限预算模型失败。

因此目前应分别检验：

1. **预测收益**：保持信息集、模型预算和输出族后，latent training 是否改善 observation-space proper score。WorldTS 的消融比 latent 可视化更接近这个问题，但没有完全隔离信息与优化。
2. **状态信息收益**：相同摘要预算下，是否保留更多关于多个未来查询的条件信息。预测均值更准、latent 更平滑、线性 probe 更高，都不是完整条件 law 信息的充分证据。
3. **递推／摊销收益**：已经保留的信息能否由有限更新器和转移器准确使用，且具有可测质量／耗时优势。PSIM、LIFTS 属于这一路完整方法，不能从论文叙述中删除。

**[推断]** MDN-VJEPA 尤其适合作为判别边界：它证明“具有好状态时，正确组织未来 law 会有用”，却刻意没有把该结论扩成“其目标已使 encoder 学到更有信息的状态”。VJEPA 的 sufficiency 条件也不能填补这一实证空位。这个缺口值得测试，但不是 ordinary generative inference 不可能完成的任务。

## RQ2：一个能精确隔离三个误差的小型检验

### 可枚举 HMM 作为证据载体

**[建议，未执行]** 用 3 个 hidden states、3 个数值观测符号（例如 -1/0/1）、历史长度 8、未来块长 4。状态只用于计算 oracle，不向方法提供 hidden labels。一个预注册真值可取：

```text
T = [[.85,.10,.05], [.05,.85,.10], [.15,.05,.80]]
E = [[.75,.20,.05], [.20,.60,.20], [.05,.20,.75]]
```

所有转移／发射概率为正，观测 law 有重叠但非退化。3^8=6561 个历史、3^4=81 个未来块可以枚举。forward recursion 给出 b*(s | h) 和 p*(y | h)。这不是要求实际方法恢复物理坐标；已知状态只是一把实验量尺。

为精确测量信息损失，诊断版所有方法都使用同样的有限 code Z=f(H)，预注册 code 数 8/16/32；同时保留不压缩的 GRU ordinary control。有限 code 只服务于精确误差归因，不能据此宣称所有 continuous-state 方法存在同样缺陷。

对已冻结的 f，可直接按真值历史概率求和：

```text
b*_f(s | z) = sum_{h: f(h)=z} p*(h) b*(s | h) / p*(z)
p*(y | z)   = sum_s b*_f(s | z) p*_{T,E}(y | s)
```

于是 log-risk 的状态信息项可以**精确**计算：

```text
Delta_info(f) = E_H KL[p*(Y | H) || p*(Y | f(H))].
```

不用 t-SNE、latent Euclidean distance 或有限 probe 失败来宣布信息丢失。连续状态上的 probe 失败尤其不能区分“信息没有保存”与“probe 无法读出”。

### 做三个维度的 oracle 替换，不强行写成独立相加

使用同一个真值 emission E，分别替换：

| 维度 | oracle 版本 | learned 版本 | 隔离什么 |
|---|---|---|---|
| 信息接口 | 完整 H | 已冻结 code f(H) | 压缩损失；用上述精确项衡量 |
| inference | b*(s|H) 或 b*_f(s|Z) | 固定信息接口上的 amortized posterior q(s|·) | 给定信息后的推断近似 |
| transition | 真值 T | 用训练观测拟合的 T_hat | 给定 posterior 后的动态拟合误差 |

对 2×2×2 cells 计算同一个未来块 joint log score；再检查 oracle filter + learned transition 和 learned filter + oracle transition 两个关键互换。这里 transition 的 learned HMM 是统一诊断接口，不冒充 native LIFTS／WorldTS 复现。完整 native 方法另在相同观测任务上评估。

这种设计可以报告主效应和交互，不能把结果硬解释为三个互相独立的加数。原始 generative map 和 native encoder 的坐标通常不一致，直接交换任意模型的 latent/decoder 会制造坐标失配；统一已知接口正是为避免该问题。

**ordinary controls 必须充分**：同历史和训练样本的 GRU categorical autoregressive forecaster；同预算直接多步 block categorical predictor；最大似然 HMM；准确 oracle forward filter。所有观测模型给同样的 future block 概率，不用 MSE-only baseline 证明分布状态重要。oracle 的 hidden labels／真值 transition 均不可混进实际训练组。

两个预注册反例：T 各行相同的 iid 条件下，历史状态不应获得信息收益；E 接近单位阵的近完全观测条件下，复杂 state inference 不应产生稳定大优势。如果只在刻意弱控制下出现优势，否决实验主张。

## RQ3：可试的具体交互，以及停止条件

### H41-M：conditional-law target × 有限可复用状态

**[假设，未证明新颖]** 把单次随机未来的 point latent target 换成同历史下的 predictive-law target，可能在有限 memory 和有限样本下更稳定地分配状态容量；其实际意义应体现在 Delta_info 降低，而非只降低 latent matching loss。完整实现是：从独立训练折上的强 ordinary forecaster 得到未来块 soft distribution；压缩 causal state 同时接受真实观测 proper loss 和 law distillation；同一状态供多个 horizon 的 decoder 使用。测试不需要额外动作、物理结构、零样本或新数学算子。

**最小 2×2**：sample observation CE vs cross-fitted law target；finite code vs continuous state。共享 backbone、future output family、训练数据和调参额度；至少三个独立训练种子。总训练成本必须包含 teacher；oracle law target 只作 positive control。主测未来 joint log-risk、精确 Delta_info、质量／内存／推断成本；保留 GRU ordinary law-distillation control，避免把蒸馏收益命名为 world-model 特有收益。

交互的可证伪预测：law target 的主要收益应在有限 code 条件下增加，而且通过 oracle readout 后仍能体现为 Delta_info 减少；如果只让 q 更易拟合，信息项不动，说明收益主要是 inference/optimization。如果 strong ordinary distillation 以相同总成本得到相同效果，也应撤回“新的状态机制”解释。

**最近完整方法的约束**：PSIM 已有未来统计量状态监督和递归训练；VJEPA 已有概率 latent target；MDN-VJEPA 已有 GMM、轨迹耦合、CK；LIFTS 已联合训练生成与过滤，且其额外 calibration 未系统奏效。因此 H41-M 现在只是一个便宜的交互筛查，不是足以发表的新模块组合。若能找到跨方法、跨 carrier 的稳定质量／内存优势，才值得围绕“把强预测器编译成可复用状态”继续构建框架；需继续核对 predictive compression／distillation 完整先例。

### 当前应保留的替代表述

若 H41-M 被 ordinary controls 解释，改以**“预测目标与计算预算如何决定可复用状态的质量”**为研究问题，把目标选择、状态成本、推断近似和动态拟合放在统一、可复核的评估接口。它可以产出可迁移的训练／选择框架，但“多一个 latent loss”或“普通预测器不是 world model”都不能充当贡献。该替代表述没有承诺未检验的性能，也没有要求所有 classical 方法必然失败。

## 对冻结 RQ 的最终回答

- **RQ1**：有条件的证据支持预测归纳偏置、可递推未来摘要、概率分布使用和多任务推断；缺少对 learned state 本身的信息收益与转移／推断收益的清晰隔离。不存在已建立的“同信息完整 ordinary conditional law 仍必然不足”结论。
- **RQ2**：小型可枚举 HMM + 共同 finite code + oracle posterior/transition swaps 可以精确隔离；它是机制量尺，不是原论文完整复现，也不要求实际 latent 物理可辨识。
- **RQ3**：H41-M 有明确对象、交互与 falsifier，适合小成本筛查，但尚未通过完整先例的新颖性审查。若状态信息项和质量／成本均无独立收益，应关闭该 state-learning 主张，保留预算化训练目标／推断框架的替代方向。

## 核验记录与限制

六个核心条目均通过标题存在性和上述机制／范围核验；LatentTSF 另读固定 commit 源码。本文没有复制任何论文的大段原文，没有声称复现论文结果。VJEPA／MDN-VJEPA 正文范围限制用于校准结论，未把 sufficiency 假设当作学习保证。MDN-VJEPA 页面当前显示的编号月份与 submission date 不一致；本文不据此推断发表先后或优先权。

独立主流视角不替代其他视角的覆盖；有限可枚举 fixture 的机制结论必须在至少两个具有不同观测难度的真实 carrier 和强 ordinary controls 上进一步检验，才可能支持用户要求的可复用框架。
