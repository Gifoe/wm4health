# 当前研究状态 — 2026-10-23 V13

总体研究目标ACTIVE/UNACHIEVED，接受新颖方向0。停止以不正确时钟制造的失败支撑新架构。

新增原生模型rolling-origin审计：同一Meta-iLaD物理checkpoint，全部75个ID环境、300条query、五个起点，全部评分在原训练0..3.9秒时域。中间相位错误重置clock的MSE约22–34；正确绝对clock约0.23–0.40，普通prefix-only初态拟合约0.18–0.20。该负对照不能算作原论文失败，原生模型已有绝对时间输入，无需新架构。

7500项完整query指标独立复算一致；native support/query RNG、全部排除规则和prefix argmin均验证。持续latent正确时钟重启像素最大差3.55e-6，五次原模型新鲜重放最大差3.10e-6。0次网络训练，300次state优化调用。全部原始数组和日志留G，源码/配置/指标/回执上传GitHub。

普通state拟合的均值在五个起点均改善，但个别query出现prefix更好、future更差；全部1500行及最差case完整保留，未据此调参，尚不足以认定新问题。[完整结果与单例退化](meta_ilad_native_20261023_v1/ROLLING_ORIGIN_FINDINGS_V1.md) · [七份交付](meta_ilad_native_20261023_v1/rolling_origin_audit_v1/final_research_direction.md)。

上一轮context/机制移植残差已被同信息普通latent推断大幅解释，该新架构动机继续停止。WorldTS/LatentTSF/Meta-iLaD保持原生任务分组，不冒称同任务三基线排行。新候选仍需要独立能力、可辨识信息合同和强普通控制之后的实际剩余缺口。[前版状态](CURRENT_RESEARCH_STATUS_20261023_V12.md)。
