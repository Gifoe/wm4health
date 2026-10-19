# C36：外部预报条件与世界状态的研究边界

更新 UTC 2026-10-16T22:15:52.350331+00:00。C36 是已完成的文献/源码及数据资格审计，整体目标 ACTIVE_UNACHIEVED，接受最终方向 0。本轮没有模型训练、预测误差或方法效果实验。

## 摘要与研究问题

目标为可复用时间序列 world-model 的状态学习、条件转移与观测生成框架。数据集只是证据载体；允许分别训练，不强制共享零样本 checkpoint、action、因果/物理识别、Loop 或 LLM。 本轮三个冻结问题是：现有模型是否已经使用带误差的未来预报；同信息的直接预测、联合概率建模和校准积分覆盖了哪些机制；是否还存在可检验的有限预算学习残差。文献不支持“future covariates 被一律当成未来真值”的宽泛论据。数据资格有所推进，尚未证明新状态机制有必要或有效。

## 方法与证据分组

三份独立 clean-context 原文检索见 [mainstream](mainstream_conditions36.md)、[critic](critic_conditions36.md)、[constructive](constructive_conditions36.md)。主线程进一步核验代码/卡片和原始接口；来源及整文件 SHA 见 `public_metadata/receipts.json`、`source_audit/manifest.json` 和 `outputs/elexon_probe_v2/receipts.json`。 检索主窗口 2026-04-17 至 2026-10-17；更早基础工作单独标识。先宽检索再按命中方法和数据缩窄；身份与具体主张分开核验。三视角都没有执行模型或科学实验，其建议的 24-fit 设计不是完成记录。以下按机制分组，不把不同输入任务的分数混成排行榜。

## 一、已经读取预报的条件模型

[WorldTS](https://arxiv.org/html/2609.31162v1) Appendix A.1 的 NP/PJM/BE/FR/DE 使用负荷、风电或发电预报；[Chronos-2](https://arxiv.org/html/2510.15821v1) §5.3 的 EPF-DE 也含 day-ahead forecasts。它们的接口足以表达 forecast-as-input；不能由字段名 future_covariates 推导“物理未来精确已知”。反过来，读入预报也不等于已评估 ensemble/vintage 的联合概率语义。

[Cross-Unet](https://www.nature.com/articles/s41467-026-73817-3) 已针对历史与 forward weather 的异质作用设计交互；[KITE 的 ICML 2026 正式版本](https://proceedings.mlr.press/v306/cheng26t.html) 已有历史条件中心/尺度、低秩噪声流形和 conditional generation。前者是直接模型的强控制，后者是概率生成的强控制；没有证据允许把任一说成已完整解决真实 forecast-vintage uncertainty。

主流 note 中仅由作者主页未检得 WorldTS code 所产生的可用性疑问在整合时纠正：[公开 WorldTS 仓库](https://github.com/decisionintelligence/WorldTS) 早已固定为 `269426ea6217d7152a244da0e209d922acd46d4f`，此前多轮实际执行；本轮不重复下载或伪报新复现。ICML 的 KITE 也不是名字相同的 event 模型。

## 二、不确定输入、联合分布与后处理的完整先例

较早 [DSSMF，IJCAI 2019](https://www.ijcai.org/Proceedings/2019/0402.pdf) §2.4 已将未来外生变量分布经 MC 送入深状态空间模型；其误差实验为合成，不能当真 vintage 证据。[MMFE 状态空间研究，2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8182152/) §4–6 已把 forecast/revision 信息扩展为状态，但其线性理论不直接决定神经模型的有限数据效果。[UKF-L，2012](https://mlg.eng.cam.ac.uk/pub/pdf/TurRas12.pdf) §4 已按预测 marginal likelihood 学习有限 sigma points，限制了“少数任务相关情景首次可学习”的主张。

普通 joint forecasting 的 [TACTiS-2](https://arxiv.org/html/2310.01327v2) 与 [CircuITS](https://arxiv.org/html/2604.27814v1) 可表达联合变量的条件/边缘分布；它们并未在本轮被迁移到实际发行数据。[PBC](https://arxiv.org/html/2604.16238) 展示 forecast/lagged observations/climatology 的概率偏差修正与组合。已有积分、联合建模和校准机制的存在不证明所有有效组合已完成，但也不允许把这些原理改名为新世界模型。

## 三、真实误差与现有实验证据的边界

[2026-10-05 load covariate uncertainty benchmark](https://arxiv.org/html/2610.07232v1) §II-C 用 train ARIMA、validation residual scale 和独立高斯扰动生成不同噪声强度；现有 day-ahead load forecast 并非同样被扰动。作者承认 bias、temporal dependence 和 non-Gaussianity 限制。它支持输入误差值得评估，却不能证明同信息、在真实历史预报上健康训练的 strong direct predictor 仍失败。

因此公平目标首先是 p(Y_future | H_origin, F_origin)。直接模型允许学到预报不准、年龄/可靠性和非线性响应；它不必先恢复未来真实天气。新模型若使用 spread、revision 或成熟误差历史，普通基线也须获得同样信息。不能 truth-train/forecast-test 后把不匹配当创新动机。

## 四、可用数据合同而非无限强化的审计门槛

[GEFCom2014 组织者](https://www.pierrepinson.com/docs/Hongetal2016.pdf) §6.1/§7.1 给出训练和评测均提供 daily midnight、至 +24h 的 ECMWF forecast 合同，足以支持声明清楚的 benchmark 协议；本轮未下载完整数据，也未核 timezone，不添加 UTC 或独立网络接收保证。它不提供本候选所需的全部天气实况、ensemble 或 revision。

[OpenSTEF Liander2024 卡片](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark) 有 versioned available_at，并明确部分 availability/delay 为模拟。固定的消费代码证明 as-of 选择接口存在，不能证明 weather construction 链。[Open-Meteo Single Runs](https://open-meteo.com/en/docs/single-runs-api) 区分初始化与实际可用时刻，并注明 2024 IFS 为 hindcast。hindcast 可以用于声明清楚的协议，不能冒充当年实际发行。

Elexon 一日公开原始 JSON 已在服务器验证：发布时间字段可用于 as-of 选择；原生预报为小时，实测为半小时。资格结果及范围见 [实际结果](experimental_results.md)。不要求每行历史网络抓取日志才能做研究，也不以暂缺元数据推导方法新颖性。

## 跨组综合与逐问回答

RQ1：预报条件模型已存在，且部分已原生使用实际外部预测；宽泛空白被反证。RQ2：冻结目标模型、只校准外部未来条件再积分的当前变体与普通两阶段方法在指定因子分解下目标等价，关闭其新方法主张；这不否定所有有限预算新组合。RQ3：一日发行 fixture 合格，但没有模型学习残差或新机制收益。只保留未批准的“小预算、目标相关不确定性表示与共享转移的交互”问题；压力、成本与近邻差异均待证。当前不推荐最终方向。

## 自我反审

没有把所有模型都说成泄漏，也没有把已知组件等同于所有组合都已解决。没有强加 causal/action/zero-shot 门槛。最具体提案被关闭的依据是可执行目标等价，而非单纯“先有人做过积分”；条件性压缩问题被保留为待证伪问题，不升级为创新。数据审计既没有证明方法有效，也没有成为停止研究整个问题的理由。
