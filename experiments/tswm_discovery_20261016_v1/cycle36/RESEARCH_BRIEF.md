# C36：外部未来信息的不确定性如何进入状态与转移学习

2026-10-17，检索前冻结。整体目标ACTIVE_UNACHIEVED，接受最终方向0。上一轮C35=PROGRESS：实际两任务12fits和独立复核关闭了当前protected-latent分支方案，不代表world models或条件表示学习无价值。

研究层次以WorldTS为参照：通用的时间序列状态学习、条件演化与观测生成机制。按数据集训练允许，跨数据任务验证同一机制；不强制大模型、Loop、LLM、物理/因果识别、动作、通用checkpoint或在线适应。

## 三个冻结问题

1. 近期已发表/预印本数值world-model与covariate-aware forecasting框架，如何区分真正future-known条件和在预测时已经发布、但仍有误差的外部未来预测？有哪些原生实验、可执行代码和输入合同直接评价这种区别？不能将作者未声称的因果语义或未来truth可见性强加给它。
2. 在相同发布时间可用信息下，现有direct forecast-as-input、预测协变量的联合生成、ensemble marginalization、普通calibration/forecast combination和Bayesian uncertain-input机制，已解决哪些问题？是否存在一个具体、重要、预算有限且可检验的学习残差，值得设计state/transition框架，而不只是availability审计、Jensen偏差、output adapter或旧IS wrapper？
3. 哪个最小可复用机制、最近的完整方法对照和公开数据协议能证伪该残差？必须给出issued_at≤forecast_origin的可核查数据版本，及单5090的最小决定性实验。若普通直接条件模型已胜任，明确关闭该具体命题；不要强行留下Idea。

## 三个独立clean-context视角

- 主流：实际已发布模型/输入合同/实验与数据发布时间。关注状态与条件演化，不只列forecasting榜单。
- 批评与方法学：最强普通对照、训练/部署目标一致性、可获得信息、可辨识性与负例，具体指出能推翻何种主张。
- 建设性相邻机制：寻找一个可实现的有限预算interaction，给出完整相近先例和能失败的最小协议；已知组件不自动否定组合，但换名字/换输入模态不构成机制。

主要检索窗口2026-04-17..2026-10-17，必要早期基础工作单独标识。仅依赖原论文、作者code/card、官方proceedings、原始数据提供方。身份与原文主张分别核查；未核实不使用。每个视角独立先宽后窄，合并后补盲区，不重复堆同一reference池。

## 现有边界与不重复事项

旧NESO persistent在线updater、NP/PJM Loop/readout/epoch1调参路线关闭。C16因为未来forecast columns发布时间不明使用calendar-only；这说明当时不能用该信息，不等于本问题已有新颖性。若复用旧数据，必须查明实际来源与发行版本，不能把旧列重命名成真实forecast vintage。

C34 first-hit-source proposal因conditional-channel MLE逐项等价撤回。C35 common E/D、target-only P加historical-condition residual共6arms×solar/Jena：PZ比普通JZ_N更差1.627%，Jena全选step0，单seedTRAIN/VAL。该配置关闭；不继续调这两任务loss/字段/seed挑收益。C35未涉及future external forecast或未来条件分布。

更早的通用missingness/observer/uncertainty/covariance/系统辨识/输入积分/semigroup toy已被强普通方法解释。它们不证明全部有限预算组合已解决，但普通Kalman、joint probabilistic forecasting、已知预测器不确定性与augmentation必须认真纳入，不可人为削弱。

本轮intake不跑训练、不大批下载数据或模型、不公开发布、不删除旧成果。never read/execute cycle6 secret/W&B launcher。只写各自命名note；主线程合并并独立复核后才冻结实验协议。
