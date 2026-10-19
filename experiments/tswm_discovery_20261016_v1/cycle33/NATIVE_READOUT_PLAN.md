# C33 Native WorldTS frozen-readout attribution

冻结于训练前。任务是通用时间序列 world model 的证据发现；本实验是已有模型的机制诊断，不是新方法、未解决gap或论文复现数量目标。C32固定编码反例只说明一种兼容性可能，不足以证明原生learned模型失败。本轮不再扩展该toy。

## 数据与被冻结的模型

NP/PJM、H24、各自公开recipe的T168和latent_dim128/224，source revision269426ea6217d7152a244da0e209d922acd46d4f。使用已完成baseline_reproduction_seed2021_v1的原stage1 checkpoint，严格检查其非decoder张量与原stage2相等；旧文件/环境只读。

保留named target/covariates与官方future-known numerical-covariate合同，不称为因果干预或实际部署未来可获得性证明。CSV按具名列pivot，52416小时、3通道；原train70%/val10%/test20%，先tv=floor(.8*N)，再tr=floor(.875*tv)。只用TRAIN统计拟合StandardScaler。Train目标区间完全在train，val目标完全在val，test目标完全在test；history可回看此前已观测区间。

这些数据和原checkpoint曾用于本对话研究。本轮test是原数据的固定诊断，不是独立新cohort，也不将滚动窗口当独立世界。完成所有24次fit与checkpoint选择前不计算候选test分数。

## 四个普通readout

| arm | encoder/predictor | decoder训练 | 用途 |
|---|---|---|---|
| native_mix | 同一冻结stage1 | 原causal-patch decoder，forecast MAE+.3×target-encoding reconstruction MAE | 原监督混合机制 |
| native_pred | 同上 | 同一decoder，forecast MAE | native现有prediction-only普通修复；与mix逐seed完全同初始化 |
| wide_local_pred | 同上 | 同一causal patch逻辑，FFN hidden×4，forecast MAE | 普通局部容量控制；参数/计算另列 |
| global_pred | 同上 | 标准MLP，读取全部future predicted tokens、实际history prefix及已提供的RevIN mean/logscale，forecast MAE | 完整decoder信息集的普通非局部读出；不是新架构 |

global MLP为256→256→H，GELU；输入每坐标mean/std仅从TRAIN的冻结特征估计，std floor1e−4。输出仍按原RevIN stats反标准化，两个stats显式进入MLP只重新使用已有decoder信息，不增观测。Local两臂保留原生输入/恢复方式。目标编码只给mix重构损失或事后diagnostic，不给预测支路；记录改变未来target而保持past/futurecov时h_pred不变的资格检查。

原完整stage2模型是reference，不与新heads的不同初始化或预算混成论文复现排名。新增均为frozen-readout refit或普通capacity/protocol extension。

## 固定训练与选模

Seeds2026331/2026332/2026333，各dataset4arms，共24fits。Adam原生stage2_lr（NP1e−4，PJM2e−4）、batch32、20完整epochs、无early stop；type3按官方epoch-end调用等价LR（前4个实际训练epoch为base，之后×.9）。不做额外LR/width/test搜索。

同dataset/seed四臂共享20条完整样本permutation，保存order文件及SHA；不丢尾batch。相同更新次数不等于相同FLOPs或wall-clock，mixed/更宽/global成本另列。所有臂以固定完整validation forecast元素等权MAE选checkpoint，epoch0合格；这统一了选择目标，区别于官方mixed总目标选择，故不声称原训练全流程复现。另保存/记录原生objective选出的checkpoint作为预列secondary选择控制。

冻结stage1推断可缓存为float32，保存h_pred/h_target/decoder所需history prefix/RevIN stats/y，禁止float16近似。首先用原state_dict与原forward/decode核对缓存等价，再正式fit；所有缓存和旧checkpoint SHA锁定。只缓存train/val直到fits完成；test推断随后单独执行。正好相同encoder/predictor不必反复执行，报告缓存预处理和decoder优化耗时。

## 指标、归因与停止

Primary：test MAE；secondary MSE、逐horizon误差、只含同一允许信息的target-code重构diagnostic、每seed相对mix/pred的gain、参数/更新次数/时间/显存。保存全预测、origin与target，不只汇总。

mix→pred可重复改善只证明普通辅助重构的负迁移；native已有修复，不形成新框架。wide/global消除差距，则优先解释为容量/读出限制；不能证明潜态丢信息。若所有readout仍差，须与同输入/预算的observation predictor、优化/权重等普通envelope比较，不能直接命名信息不足。只在两个同类电力任务的结果不足以声称跨领域普适。

若最后若干epoch仍明显下降，报告固定预算尚未收敛；不把cap当principle limit。描述paired seeds和按连续168-origin区块的敏感性，不用重叠窗口数或两个dataset作大样本显著性。

独立验证覆盖：原数据/命名/划分/统计、冻结hash与缓存重放、每fit同样本序列、selection minima、实际state_dict与test预测重放、全部MAE/MSE/差分重算。验证通过前不使用训练/测试结果作科学结论。任何实现失败保留原脚本/目录，另起明确修复版本，不放宽证据门槛。
