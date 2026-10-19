# C42 — RUNNING

目标仍为可复用的时间序列 world model。当前接受最终方向0，整体ACTIVE_UNACHIEVED。

已冻结PROTOCOL.md并通过精确likelihood枚举、anchor独立枚举、有限差分梯度、未来不可用值污染检查。定时资格显示该小型精确HMM在服务器CPU约8ms/update，GPU约18ms/update，因此选CPU；两种设备的资格权重均丢弃。未改服务器全局环境，其他GPU任务继续运行。

当前运行32个未知参数普通conditional HMM fits：两个独立world、两个训练数据量、两个初始化、观测/动态共享2×2。控制了相同数据、选模信息、loss和exact inference。完整分离11本身就是ordinary structured HMM，无独立“新方法”。结果尚未完成、未形成创新结论。此前真实数据TEST未评分。

后续先独立重算全部raw scores与选模，再决定是否还值得开展神经/真实域候选开发；禁止从人工构造中已知的factorization直接宣称全球未解决问题。
