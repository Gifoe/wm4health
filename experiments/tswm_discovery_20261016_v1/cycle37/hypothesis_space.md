# C37 竞争解释的完整证据索引

UTC 2026-10-16T23:04:09.714426+00:00。C37原作业完成，12个neural fits、2个HGB fits、48000个formal updates；独立复核通过。整体目标仍ACTIVE_UNACHIEVED，最终接受方向0。TEST没有评分。本轮是failure discovery与普通解释控制，不是新方法。

预列比较已全部完成，详细方向、效应和探索CI见[结果表](experimental_results.md)，不得只报告其中显著行。

| 解释 | 已执行区分 | 能支持的结论边界 |
|---|---|---|
| 直接边际加经典rank coupling足够 | exact同marginal IID/shuffle/analogue；HGB residual joint/IID | 可归因对应路径变化，不能泛化所有任务 |
| 原flow loss而非state能力解释差异 | L1/L2匹配初始化、schedule、RNG和预算 | 标准目标控制；不是新方法 |
| 只是marginal calibration | 所有neural raw与同CAL校准 | 校准本身不产生机制创新 |
| 只是积分离散误差 | 同weights 10/50steps | 不重选checkpoint；不能等同外部forcing K |
| undertraining/overfit/容量 | 所有step0到4000曲线、cap标记与参数/成本 | cap无改善也不证明全局收敛 |
| 额外状态/转移机制必要 | 尚无新方法，尚无独立TEST | 本轮不能批准最终框架 |

两个seed、两个固定zone、一个小时级日窗口协议，全部是探索VAL结果，不是独立TEST确认。wind VAL原105候选仅66完整case，39因缺失被排除；TEST76未评分。solar VAL118/TEST119。预算cap标记0/12，初始step0被选0/12；无cap标记不等于已证明收敛。参数量/训练目标/CPU树与GPU网络FLOPs并未严格匹配，不能仅凭排名断言特定state机制缺失。bootstrap按7个日历日而非连续row号，2000次；CI探索性、未校正多重比较；两个seed的score平均不是forecast ensemble，也不是大量独立随机重复。
