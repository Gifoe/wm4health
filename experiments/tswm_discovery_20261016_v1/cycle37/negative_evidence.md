# C37 负证据与未排除解释

UTC 2026-10-16T23:04:09.714426+00:00。C37原作业完成，12个neural fits、2个HGB fits、48000个formal updates；独立复核通过。整体目标仍ACTIVE_UNACHIEVED，最终接受方向0。TEST没有评分。本轮是failure discovery与普通解释控制，不是新方法。

数据/reader失败全部版本保留：hour0..23修正、wind11NaN保留并按固定完整窗口排除、pandas numpy.str_显式str转换。正式原训练没有被这些preflight修正重复启动。solar未来POWER没有进入features。

两个seed、两个固定zone、一个小时级日窗口协议，全部是探索VAL结果，不是独立TEST确认。wind VAL原105候选仅66完整case，39因缺失被排除；TEST76未评分。solar VAL118/TEST119。预算cap标记0/12，初始step0被选0/12；无cap标记不等于已证明收敛。参数量/训练目标/CPU树与GPU网络FLOPs并未严格匹配，不能仅凭排名断言特定state机制缺失。bootstrap按7个日历日而非连续row号，2000次；CI探索性、未校正多重比较；两个seed的score平均不是forecast ensemble，也不是大量独立随机重复。

同marginal rank重排可改变daily/ramp/joint score，不能当本项目新发现；标准L2、普通校准、10/50-step变化若解释差异，就否定相应“新状态机制缺失”的归因。反过来，某个普通控制未胜也不自动证明新框架必要，仍须排除有限预算/容量/有效样本差异。

C36 observer-only提案仍因普通目标等价关闭；C35具体protected residual配置未过screen仍关闭。未重新调zone/seed/rank或TEST以复活旧方向。没有weather truth，C36 q(U|F)条件假设仍未建立；target residual结构只是目标条件分布的诊断。

独立raw↔canonical、window/split/scaler、schedule、initial/selected checkpoint、VAL/CAL native路径及selected64-path评分重放通过；12个neural fits和2个HGB point模型。对全部56个已存路径arms重新计算9个score，共504个score cells；最大差1.21156307209e-10。native prediction/CAL重放最大差0（容差2e-4），不是声称逐bit为0。quantile/HGB同marginal排列恒等性通过。50-step和CAL/rank arms的score已独立重算，但不是各转换实现的全套独立重写；HGB点值已重放，不把随机residual路径生成称为第二套独立实现。
