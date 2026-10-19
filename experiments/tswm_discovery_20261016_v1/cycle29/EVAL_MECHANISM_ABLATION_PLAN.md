# 评估预声明：机制代码是否实际被使用

在任何模型测试性能被读取前，补充以下无需新训练的诊断。不修改训练协议、数据、超参数或 checkpoint 选择。数据物理重算不属于模型性能查看。

对每个训练 seed、两个家族及所有128个测试系统：

- `compiled64_zero_operator`：保留该系统编码出的当前 hidden state，将生成的低秩 recurrent update 置零。
- `compiled64_swap_operator`：保留当前 hidden state，将该系统 recurrent update 替换为同家族下一个系统（循环顺序）的 update。四个未来计划共享同一交换。
- `conditional128_swap_code`：保留当前 hidden state，以同样规则交换普通 context code。

如果 compiled 模型的生成算子被置零/交换仍几乎不变，不能把其效果归因于机制编译；它可能通过当前 hidden state 存储系统身份。若交换造成损害，只证明模型依赖该代码，不证明它恢复了真实物理参数或已获得创新性。固定交换无需选择测试超参数。

所有诊断保留全量系统指标，不据诊断结果选择正式模型。不把 out-of-distribution latent 扰动自动解释成系统辨识失败。
