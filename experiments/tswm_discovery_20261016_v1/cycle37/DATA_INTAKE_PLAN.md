# C37：真实历史预报任务的学习压力筛查

2026-10-17，读取新信号前冻结。C36=PROGRESS：三视角原文/源码与服务器一日发行资格审计完成，observer-only提案关闭；整体目标ACTIVE_UNACHIEVED，接受最终方向0。

目标仍是可复用时间序列world-model的状态与条件演化框架。此次数据为机制证据载体，不把GEFCom分数优化视为研究目标。不启动旧NP/PJM/NESO/C35的调参。

## 本轮先解决的事实问题

1. 是否可从组织者/正式附件取得GEFCom2014 wind及solar原始CSV，并保留完整文件版本和原benchmark条件预报合同？
2. 两个任务里，健康的同信息直接预测与普通校准/联合轨迹模型之间，是否存在对路径预测而非仅点误差重要的差异？这是failure discovery，不是新方法验证；尚无新方法被批准。
3. 这些数据实际能证伪哪一条解释？GEFCom的未来NWP为forecast，不是未来真实weather。没有真实weather标签就不能计算physical covariate residual、训练有监督weather calibration q(U|F)，或伪称完成C36所有pressure条件。

## 数据取得与保留

优先组织者论文附件及IEA官方测试数据页指向的GEFCom2014原ZIP。候选精确地址：
- https://ars.els-cdn.com/content/image/1-s2.0-S0169207016000133-mmc1.zip
- https://dl.dropboxusercontent.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip

正常TLS，原始ZIP/URL/UTC/bytes/SHA256保存。下载上限256MiB防止意外大文件，不是科学预算。若服务器访问失败，可本地公开下载后整SHA上传，透明报告传输路径。保留失败日志，不修改旧成果。不extractall，不运行档案内脚本，只在新目录读取明确列名CSV。

资料：Hong etal2016原文 https://www.pierrepinson.com/docs/Hongetal2016.pdf §6.1 wind与§7.1 solar说明每日midnight发行、hourly+24h预报在train/eval提供。仅按其原benchmark合同，不添加未核的UTC/历史网络接收保证。IEA摘要表有7farms/旧年份，与原文10farms不同；以实际原ZIP及组织者原文为准并记录差异。

优先Task15最终完整训练文件；Task1–14不可拼成重复样本。每个任务固定最小zoneID（预期1），不遍历后挑最好站点。先只审计schema、时间连续性、重复、缺失和数据范围，不读取信号分数。

## 学习实验前的协议边界

在模型训练前另行冻结明确arms、horizon、history、split、loss、预算、seed、选模与主评价。按origin时间切分、排除跨split目标重叠；拟合/校准只用TRAIN；VAL用于一次选模；TEST暂不评分。所有模型得到完全相同在origin可用的历史target、已发行未来forecast和日历。

目标轨迹的残差相关性可以检查，但不等于外部weather forecast误差。普通误差path bootstrap/copula、强direct和现有conditional generative/state模型是必要对照，不通过弱化它们制造latent需求。有限预算组合可以有价值；只发现普通已知修复或元数据问题时不立项。

没有已批准新机制，不执行原notes的24fits。never read/execute cycle6 secret/W&B launcher。旧数据/权重/日志只读保留。
