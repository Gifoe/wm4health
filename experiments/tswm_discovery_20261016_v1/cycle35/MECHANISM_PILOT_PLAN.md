# C35：上下文与数值分支的学习责任，TRAIN/VAL pilot

2026-10-17。用户要求可复用的时间序列world-model框架，以WorldTS为研究层次参照。接受最终方向0。此方案在任何拟合前冻结，承接C34 critic的可证伪线索；不宣称残差、stop-gradient或latent loss新颖，不是Pythia原生复现。

## 问题与已有边界

共同训练的外部条件分支能否带来独立预测信息，还是补偿数值分支的退化？在有限预算下，保护数值自身预测任务的latent innovation是否优于相同保护的普通output residual？Pythia reference、REST-TS、Gradient-Blending、普通残差/stack已构成强重叠，阳性结果最多支持继续检验具体interaction，不能自动成立新框架。C33旧NP/PJM读出线索关闭，本轮不重跑。ETTh1在C5已使用，因此不冒充新独立数据。

## 数据与信息

使用C34已下载且只用过历史fixture的FEV solar_with_weather/15T与jena_weather/10T，固定revision f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8。目标分别预列target、target_0；其余9/20字段仅作为历史数值covariates。不猜Jena物理通道单位。不是原FEV leaderboard协议。

按完整时间轴70/10/20划分TRAIN/VAL/TEST。只在TRAIN的有限值上逐列拟合均值/总体标准差。history168，horizon96。跨任何字段缺失的窗口不可用于拟合/评估，但不删行、不压缩时间。训练target区间完全在TRAIN；VAL每96点取一个不重叠target块并允许合法过去历史。TEST future labels不进入runner或scorer，正式TEST暂不执行。raw parquet读取本身不是“密封测试标签”的加密隔离，本轮通过明确索引访问约束保证不评分/选型。

随机seed2026351，一个初始化和两真实任务，只作为预算有限机制筛查。无seed稳定性或广泛框架结论。若通过再冻结独立确认与至少3seeds，不能按TEST改结构。

## 共同表示与teacher

从已核查WorldTS revision269426ea6217d7152a244da0e209d922acd46d4f直接导入native target encoder、PatchTST transition、causal patch decoder与LatentStateLoss。修改任务配置：D32，patch-state/decoder3，transition patch48/stride24/d_model64/heads4/layer1/ff256/dropout0，history168/horizon96，target-only输入、无future conditions。该小配置是component transfer，并非论文表格复现。

Teacher Stage1只用target历史/未来TRAIN：原生10×latent MAE +15×trajectory cosine +0.1×legacyVC，Adam lr0.001，1000updates；每100updates用VAL latent loss选E/P。Teacher Stage2冻结E/P，以forecast MAE训练native decoder，Adam lr0.001，1000updates；每100updates按VAL forecast MAE选。B32；梯度clip1；float32、TF32关闭。容量/预算可能不足，记录学习曲线，不因封顶仍改善就宣称模型类缺陷。

随后所有arm固定同一E和D、复制同一P初始化。P可更新。n=P(E(y_hist))，t=sg(E(y_future))，decoder接收同一历史prefix和RevIN统计。g是同样的global MLP：输入n、历史prefix与完整训练标度历史covariates，hidden256、GELU、outputH×D；末层初始化0，使所有fused预测从同一teacher开始。PY额外32维线性投影无bias、固定随机初始化后可学习，将g映射成标准化target输出残差；保证g=0时残差0且梯度可达。PY有略多参数，完整记录；不能归为严格相等FLOPs。

## Arms、损失与梯度

每arm相同2000updates，Adam lr0.0003、B32、clip1，独立相同batch索引计划。每100updates及step0评估，按VAL fused MAE选；N按数值MAE。相同E/D不参与优化。

| arm | 公式 | 用途 |
|---|---|---|
|N| MAE(D(n),y) | 相同额外数值updates |
|JZ| MAE(D(n+g(n,c)),y)+MAE(n+g,t) | 共同训练，无自身数值objective |
|JZ_N| 上式+MAE(D(n),y) | 额外数值objective控制 |
|PZ| MAE(D(sg(n)+g(sg(n),c)),y)+MAE(sg(n)+g,t)+MAE(D(n),y) | 保护数值分支，仅自身预测loss更新P |
|PY| MAE(sg(D(n))+linear(g(sg(n),c))*hist_std,y)+MAE(D(n),y) | 普通保护output residual，目标等价于残差监督 |
|PZ_NC| PZ的历史covariates置0，训练与评估都如此；同模块容量/输入维度 | 区分条件信息与额外容量 |

所有MAE forecast在相同TRAIN标度上计算，latent MAE权重1，固定不网格搜索。因PY无latent辅助loss、映射不同，PZ胜它还不能独立归因于所有设计因素；若阳性需额外同辅助控制、强readout refit与充分预算。JZ/PZ梯度路径必须自动微分实际核验；共享初始化和逐update样本计划必须hash。

拟合前实现审计补充：P与context/projection分别clip1，防止共同clip系数间接改变受保护P的更新。N/PZ/PY/PZ_NC在每个相同update的P状态hash必须完全相同（最优checkpoint可能选择不同step，故不要求各自best-P相同）。预检查E/D参数冻结但decoder对输入保留梯度，future target更改不能改变forecast。JZ_N在固定TRAIN batch的step100/500/1000/1500/2000记录fusion+latent与自身数值loss对P的梯度内积/余弦，避免仅由隔离收益就断言梯度冲突。Teacher Stage1选模的latent几何变化不证明teacher forecast充分；PZ_NC只测相对所学表示的条件增益，不能证明给定全部raw target历史仍有条件独立新信息。

## 结果与去留

记录fused/numerical MAE/MSE、common-space latent MAE、step曲线、best step/cap、训练参数量、时间、显存、data/source/config/initialstate/schedule/checkpoint hashes。VAL目标按96非重叠块汇总，不把同一任务时间点或seed当独立系统。

只有PZ在两任务VAL fused MAE均优于PY与JZ_N至少2%，并且优于PZ_NC，才开展正式确认；这是探索go/no-go，不是显著性或最终投稿标准。阈值在结果前预列。若不满足，停止当前具体机制组合，不通过改权重/选有利字段/追加测试调参挽救。若teacher/arm仍明显改善，负结论仅限本预算/配置，不证明所有latent innovations无效。

初步通过仍必须测native historical-only WorldTS、TimeXer、合理验证stack/gate、同预算强output残差、共同strong readout和独立cohort；否则不接受为最终framework。未来无条件缺失能力、因果可辨识性和物理解释不作为本轮人为门槛。
