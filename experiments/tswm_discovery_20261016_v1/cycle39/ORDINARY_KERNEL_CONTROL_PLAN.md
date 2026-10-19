# C39 后续普通控制：一致性本身是否只是便宜wrapper？

2026-10-17，原V2自查询作业仍在运行、root未读任何effect rows时冻结。来源为独立mainstream视角提出的反证，不是根据native指标挑方案。本控制另脚本/输出，不改原运行器。

同固定8histories/同两rep、N256、未来H192，保留原Sundial参数与完整增长history。四arms：原joint720截192；canonical96每次只取head前96、append后重算再取96；canonical8同样重复24次；普通AR8(含intercept，ridge1e-3、训练history标准化、残差Gaussian)，每context拟合一次，固定其参数跨未来重复。canonical8在8-grid、canonical96在96-grid定义自身block AR law，与原720joint是不同law。缓存原sample suffix只解决同一个已选trajectory继续返回的问题，不等于新真实prefix的独立条件采样。

原720单头始终生成720，不能称canonical调用完全等于8或96维高效专用kernel。前两canonical都共享原native第一步noise；后续为独立noise，不复用可能依赖prefix的原noise剩余维度。不使用sampling均值喂回。

原V2完整独立统计核查通过后才执行。源模型不再拟合；AR仅8个history-only解析fits，没有训练SGD、验证HPO、未来信息或TEST选择。full-growing重编码使用原generate的native mean/std规则。C34source资格与6个samenoise native/adapter资格已经覆盖这一规则。

真实未来192点只用于**本补充**有限质量诊断，加载对应parquet但不作为模型输入；若未来含NaN则报告并整context排除评分，不能插值或挑有利区间。origins是早期固定数据片段，已在先前探索语料中，不能称独立TEST/无预训练污染。目标/预测按各自history mean/std归一以报告维度无关分数。

评分固定fair marginal CRPS、whole192 mean CRPS、terminal CRPS、max absolute ramp CRPS。全context/rep/arm逐行报告，另描述平均而非以8重叠origins冒充显著泛化。wall-time单独同步CUDA，含每个arm必要history编码+全sampling；统一已warm模型，AR CPU拟合和采样分开记录。rep是MC重复，不是训练seed。

若canonical96便宜且真实分数未明显损失，关闭“跨查询一致性需要新机制”的当前动机。若有代价，仅记录finite quality/cost tension，仍需与FlowNP、NDP、SFM、ForcingDAS、SC/CK训练等同合同强控制比较。不能只赢原wrapper就声称新framework。两rep/8contexts既不提供投稿确认，也不允许挑单个metric讲故事。
