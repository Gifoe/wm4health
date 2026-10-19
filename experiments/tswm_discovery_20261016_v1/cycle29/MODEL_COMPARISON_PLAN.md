# 跨系统动态机制复用：执行协议 v1

本实验检验一种通用模型组织方式，不针对某条真实记录优化。上一轮范围校正没有新增实验结果，本轮将其分类为 no progress，并启动新的可复现实验。总体目标未完成。

## 问题与竞争解释

在相同短历史、跨系统训练数据和监督目标下，将系统差异写入可复用的转移算子，是否比历史作为普通条件输入、或直接 attention 预测未来更有利于多步模拟？

- H1：普通条件 GRU 已足够；生成动态算子没有准确率/复用成本优势。
- H2：算子条件化具有有限预算优势，但宽泛机制已由 CLAW/超网络/元系统识别提出；只支持它作为强基线，不自动接受创新性。
- H3：直接多步模型的长时退化由查询/rollout方式解释，普通分块调用已修复；不据此提出新架构。
- H4：小模型训练不足/容量不足是主要原因；验证曲线与更宽条件 GRU 是必要对照。未收敛结果不能证明研究缺口。

本轮**没有宣称全新的方法**。先建立模型级证据，避免把旧机制重新命名。

## 数据与任务

源生成器：PuraFilippo/sysid_benchmarks，revision `4c20d94ccab51abf4ebf68aa8810f4596381dd41`；复用其 LTI 参数采样、线性仿真与 WH 静态非线性，保持源码不变。WH 使用 cycle23 已验证的 independent fixed-reference 定义，不使用包含未来的全轨迹归一化。

两个不同结构的系统家族：1–5 阶 LTI；两个 1–5 阶 LTI 块与随机 tanh 网络组成的 Wiener–Hammerstein。每个家族 512 train、64 validation、128 test 个独立系统；种子分别 202629101、202629102、202629103（每家族派生独立 SeedSequence）。训练世界的完整序列长度 512；每条轨迹先 burn-in 200。内部/输出固定单位来自各自独立 4096-step reference，不依赖查询未来。保留全部有限样本，不按测试性能删样本。

训练输入含 white / colored / PRBS / sinusoid+steps，幅度覆盖 0.5–2。测试 context 是 white；同一物理状态下比较 factual white、zero、constant step、alternating pulse 四个未来计划。计划从 context 截止处开始分叉；共享观测噪声，真实效应可精确抵消该噪声。所有测试系统未进入训练和模型选择。

上下文 C=128；训练预测 H=32；测试 H=32/64/128。模型输入按当前 context 的 u/y 均值和 std 变换，尺度在所有未来分支上固定；模型从不接收未来 y。所有模型都能看到相同 128 步历史。

## 模型

1. `conditional96`：GRU 编码历史为 context code 和当前 state；共享 GRU 转移在每步接收 code 与控制。
2. `conditional128`：更大 state 的同类控制，排除低秩转移收益仅由参数数目解释。记录实际有效参数，不预先虚称完全等量。
3. `compiled64`：同一历史 encoder 生成当前 state 与 rank-4 recurrent-weight update；生成一次后固定转移、对多个控制计划复用。它是 CLAW 启发的自主数值部分观测迁移，**不是 CLAW 原论文复现，也不是本项目已接受的新方法**。
4. `native_attention64`：使用上述 pinned 仓库的 TSTransformer 源实现，2 layers、4 heads、embedding64；用相同数据/MSE 从头训练。context128，query 前16步复制已观测末16步用于初始化，其后才预测；总观测信息不增加。原生概率 head 不用于本实验均值目标，记录 inactive parameters。测试直接查询与 H32 分块重查询均保留，不按测试选择。
5. 同信息、只在 context 内划分验证的 ARX；现有发布的概率 ICL 检查点仅作外部预训练参考，不与本轮训练宣称预算匹配。

## 训练与选择

训练 seeds 2026291/2026292/2026293。相同批次索引/起点序列；batch32，AdamW lr3e-4，weight decay1e-4，grad clip1，FP32，MSE。最多8192 steps；每512 steps 在独立 validation 系统固定窗口上评价；最少2048 steps，连续4次无 0.2% 相对改善即停止。仅按 validation MSE 选择 checkpoint。验证曲线仍明显下降时标为 underconverged，不能当新机制失败。任务/种子串行运行，原始日志、最后状态与最佳权重保留。

单个关键路径检查包括：自定义低秩 GRU 在零 update 时与 GRUCell 一致；数据分叉前缀严格相同；目标 placeholder 不影响原生推断。检查失败只修实现后继续同一科学协议，不用 smoke 结果代替实验。

## 指标与解释

每个独立系统记录各计划/时域的 clean 和 noisy RMSE、相对 factual 的 effect RMSE、动态响应回归斜率。按系统聚合，再按 seed；不把时间点当独立样本。保存所有预测和模型选择记录。重复64计划测量 context 编译、已缓存 context 下模拟时间、显存；延时只作本机工作负载证据。

若算子条件化无优势，淘汰其宽泛新方法动机；若有优势，仍须独立真实系统验证、最近原生方法及训练/容量对照后再判断是否有新贡献。不把合成双家族试验直接当作通用性证明。下一轮不根据测试选择层数、rank或学习率。

## 文献边界

[CLAW](https://arxiv.org/abs/2609.12278) 已做 context→低秩 world-model adaptation；[DynaDiff](https://arxiv.org/abs/2505.13919v2) 已做观测条件的模型生成；[Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html) 已有共享上下文系统分布和多计划模拟。精确的有限数据/观测假设和实测差异才可能提供后续研究价值。当前接受方向 0。
