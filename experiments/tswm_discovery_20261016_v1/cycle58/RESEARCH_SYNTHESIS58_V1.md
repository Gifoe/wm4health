# C58：机会核查与后续实验边界

核查日期：2026-10-18，Asia/Shanghai。状态：探索性文献和代码审计已完成；没有执行新的模型训练；接受的新研究方向数为 0。整体研究目标仍 active / unachieved。C57 的 heldout 尚未打开时编写本文件，不能把其未来结果当成本笔记的证据。

## 当前判断

现有证据不足以推荐一个具有独立方法贡献的最终方向。三个独立视角均未找到通过最强近邻核查的正面 lead。这不证明时间序列 world model 已被解决；它说明“加随机 latent、多个分支、长期记忆、环境变量或一致性损失”这样的提法已经不足以构成研究问题。应先定位同输入、同信息、足够容量和成熟控制之后仍存在的实际能力缺口。

本次是定向机会核查，不是对近六个月所有论文的完整系统综述。各笔记明确区分正文、摘要、目录、源码阅读与实际执行。代码存在不能写成已经复现；作者未报告的能力也不能写成经实验证明的失败。

## 三类能力的交叉判断

| 能力合同 | 已有直接机制 | 本项目仍缺的证据 |
|---|---|---|
| 给定历史和外部条件预测未来观测 | WorldTS/Pythia 的预测状态和分阶段 readout，COFE 的 coarse/fine 随机动力学 | 不能因为它们未声称因果识别就判败；需要同任务 observable error 和充分训练控制。 |
| 描述多个未来及整条路径的概率 | Epic 的 latent flow、当前 Branch-JEPA v3 的分支和 proper-score、MDN-VJEPA 的 shared trajectory mode 与 direct/composed law consistency | 泛 latent stochasticity 和 CK loss 已有直接先例；需要超出这些机制的具体、可归因剩余失败。 |
| 观测到达后的状态更新及不同时间尺度记忆 | OSSM、概率 SSM/HSMM、COFE、GLE memory、Preisach/hybrid 与 nonlinear state-space identification | 一个旧 toy 上的普通修复不证明任意框架都充分；但新候选必须正面比较这些成熟结构，不能只打败弱 GRU。 |

WorldTS 与 Pythia 的观察预测合同不同于受控模拟；[WorldTS](https://arxiv.org/html/2609.31162v1) 与 [Pythia](https://arxiv.org/html/2610.05240) 均不能被强加未声明的物理状态恢复要求。另一方面，[Branch-JEPA 当前 v3](https://arxiv.org/html/2607.05238v3)、[EpicWorldModel](https://arxiv.org/html/2610.05996) 与 [MDN-VJEPA](https://arxiv.org/html/2610.13956) 已直接覆盖概率多未来的不同实现，因此仅将确定性预测头换成 flow 或 mixture 不是新的问题定位。

[一步 fidelity/planning 审计](https://arxiv.org/html/2610.19375v1) 在一个架构及指定 warm-start/planner 协议下展示 proxy 可错序，并发现 matched continued training 吸收 auxiliary objective 收益；这与旧 C33–35 对公平继续训练控制的要求相符。它没有定位一个新的失败机制，更不证明预测精度无用。直接把这个 dissociation 换到时间序列上，再加 multi-step loss，不足以提出方法贡献。

## 旧负证据的校正

Critic 复查发现 C41 的 known-law Bayesian permutation control 有特权信息，只证明可辨识信息足够时存在解，不能替代未知模型的有限数据学习实验。但后续 C46 已有 train-only 的普通 PCA/对齐/LGSSM 学习控制，因此不能借此重新启动相同 carrier 的模块化故事。

C20 的成熟压缩控制共享了 response-target 选择成分，故它不独立否定 target selection 的所有价值；C21 独立系统上的不稳定结果却仍有效。组件已知不是框架无价值的证明，局部小实验也不是全领域不可能性定理。相反，保留这些限制不等于可以忽略已有真实负结果。

## 两个仅保留为排除问题的假设

1. **持续时间与 residual-life belief。** MDN-VJEPA 明确没有独立 persistent discrete-mode chain，但其连续状态仍可能学习 duration。若考察非几何 dwell time，必须比较 learned HSMM/SLDS、同 encoder 的 observation likelihood、RSSM 和 history-conditioned joint mixture；不把显式 age 标签、更多历史或容量只给候选。只有这些控制之后仍存在可复现的未来 renewal-risk 误差，才进入方法设计。当前没有该证据。
2. **非衰减 hysteresis 与衰减 relaxation 的跨系统复用。** GLE 工作已提供递推 fading memory 和随机 flow，Preisach/PNLSS/神经 hysteresis 已处理相关记忆。标准 context-conditioned hysteresis+relaxation hybrid 本身就是强对照。目标是未见系统和时序安排下的条件响应，不要求真实 latent 恢复；整体混合响应在 dwell 中允许变化。只有标准 hybrid 和充分阶 state-space 在公平信息下仍留下实际残差，才有后续价值。当前没有该证据。

这两项不是推荐方向，不是两个已经验证的 research gaps。不得因为模型未报告某个设置，或者普通对照需要实现工作，就将其升级。

## 实施发现与风险

Adjacent 视角静态检查 HystRNN `Exp1/HystRNN.py`，发现输入轴/循环轴、跨调用 hidden reset 和 split 前全数组 scaling 风险。这个结论仅限读到的脚本，尚未执行复现，不能外推到整篇论文。若后续采用，先修正并做 state-carry 与 train-only scaling 资格检验；直接使用有问题的实现会制造弱基线。这是工程审计结果，不是新方法。

当前 Branch-JEPA 已到 v3。Critic 笔记用明确的旧 v1 检查 hard-assignment 的边界；当前创新核查应采用 Mainstream 笔记对 v3 的 full-set proper-score 和 output-only branching 对照，不能引用旧名 MoP-JEPA 后漏掉新版本机制。

## 后续决策

先按冻结规则关闭 C57，保留所有 seed、失败尝试、训练预算和可视化，不为保留路线新增 gamma 或同 toy 调参。C57 的结果只判该模型族中现有修正是否足够，不决定整个 world-model 领域是否有机会。

若下一轮选择记忆问题，首先核查公开真实 complete-run 数据的输入方向、采样时钟、reset 信息、材料/设备身份、划分与 measurement repeatability，并验证标准 hybrid 能否在自身生成族达到数值基准；此时不启动新框架的长训练。若选择 duration 问题，先资格验证 learned HSMM/RSSM，不能把已知 HSMM 玩具当成神经方法必需的新研究缺口。任何成熟控制已充分解决预定能力格，就停止对应方法叙事，不能事后只保留胜出的一格。

最终推荐方向：无。当前交付的是更新的创新边界、旧负证据解释修正和两项严格受限的排除问题。具体训练结果以服务器上的独立实验 receipt 为准。

## 证据索引

- `research/mainstream58.md`：当前版本、真实代码 commit、能力合同、duration 反证。
- `research/critic58.md`：旧反证边界与公平输入/容量/任务语义。
- `research/adjacent58.md`：记忆近邻、可执行来源、HystRNN 风险和 2×2 排除实验。
- `RESEARCH_BRIEF_V1.md`：冻结研究问题、权限与非目标。
