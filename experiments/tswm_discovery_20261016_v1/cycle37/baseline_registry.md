# C37 已执行基线登记

UTC 2026-10-16T23:04:09.714426+00:00。C37原作业完成，12个neural fits、2个HGB fits、48000个formal updates；独立复核通过。整体目标仍ACTIVE_UNACHIEVED，最终接受方向0。TEST没有评分。本轮是failure discovery与普通解释控制，不是新方法。

目标是可复用的时间序列world-model框架；两个站点仅为机制证据载体。不能把单数据集涨点、实现修正或已知copula/校准改名为最终研究方向。

quantile MLP是自建直接控制，19个quantile、96×96 hidden；KITE原组件pin `3140ee824cbd80c5ec7fdf2b54666210519d0b39`，64dim/2depth/4head/rank8，原train/inference实现迁移。缩小模型、新daily loader、概率选模，明确不是原论文排行榜复现。L2 arm仅将原velocity L1换MSE；同initial hash/schedule/per-step RNG，不是本项目新方法。

两个任务×两个seed×三个neural families共12fits，各4000updates/batch64/Adam lr.001。全部同输入history target/history NWP/future NWP/calendar。每个neural都有同CAL逐lead逐q加性校准；quantile额外IID/CAL-PIT rank/32-nearest analogue rank无需另拟合，exact marginal multiset恒等。HGB两CPU fits固定200trees/15leaves/lr.05，CAL residual IID/joint/analogue；无VAL HPO，但不称FLOPs匹配。KITE50-step敏感性使用同10-step所选checkpoint，不重选。

| task | fit | selected step | params | fit seconds* | 128-path VAL seconds | cap flag |
|---|---|---:|---:|---:|---:|---|
| wind | quantile_mlp_2026371 | 500 | 254088 | 16.52 | 0.0113 | False |
| wind | kite_l1_2026371 | 500 | 361477 | 155.42 | 0.6349 | False |
| wind | kite_l2_2026371 | 500 | 361477 | 155.83 | 0.6499 | False |
| wind | quantile_mlp_2026372 | 500 | 254088 | 16.11 | 0.0255 | False |
| wind | kite_l1_2026372 | 500 | 361477 | 155.95 | 0.6438 | False |
| wind | kite_l2_2026372 | 500 | 361477 | 155.67 | 0.6513 | False |
| solar | quantile_mlp_2026371 | 500 | 401544 | 17.75 | 0.0275 | False |
| solar | kite_l1_2026371 | 1500 | 364037 | 162.03 | 1.0287 | False |
| solar | kite_l2_2026371 | 1500 | 364037 | 162.45 | 1.0329 | False |
| solar | quantile_mlp_2026372 | 500 | 401544 | 17.53 | 0.0228 | False |
| solar | kite_l1_2026372 | 2500 | 364037 | 162.44 | 1.0351 | False |
| solar | kite_l2_2026372 | 2500 | 364037 | 162.11 | 1.0096 | False |


原selected.pt、CAL/VAL路径NPZ、schedule、canonical/raw数据留服务器。源码、固定配置、日志和JSON已按SHA本地镜像；不对外发布。独立raw↔canonical、window/split/scaler、schedule、initial/selected checkpoint、VAL/CAL native路径及selected64-path评分重放通过；12个neural fits和2个HGB point模型。对全部56个已存路径arms重新计算9个score，共504个score cells；最大差1.21156307209e-10。native prediction/CAL重放最大差0（容差2e-4），不是声称逐bit为0。quantile/HGB同marginal排列恒等性通过。50-step和CAL/rank arms的score已独立重算，但不是各转换实现的全套独立重写；HGB点值已重放，不把随机residual路径生成称为第二套独立实现。

两个seed、两个固定zone、一个小时级日窗口协议，全部是探索VAL结果，不是独立TEST确认。wind VAL原105候选仅66完整case，39因缺失被排除；TEST76未评分。solar VAL118/TEST119。预算cap标记0/12，初始step0被选0/12；无cap标记不等于已证明收敛。参数量/训练目标/CPU树与GPU网络FLOPs并未严格匹配，不能仅凭排名断言特定state机制缺失。bootstrap按7个日历日而非连续row号，2000次；CI探索性、未校正多重比较；两个seed的score平均不是forecast ensemble，也不是大量独立随机重复。
