# C39 执行登记

UTC 2026-10-16T23:31:05.545952+00:00。原生query诊断与普通kernel控制均完成且独立复核；最终接受方向0，整体ACTIVE_UNACHIEVED。新neural updates0；只有8个history-only AR8解析fits。

| 对象 | 本轮执行 | 范围 |
|---|---|---|
| 固定Sundial128M | 原生成head及新增history重编码；6same-noise资格 | 组件诊断，不是原论文score复现 |
| fixed-reference | 同weights和inner noise，原坐标控制 | 普通RevIN解释对照 |
| canonical96/8 | 同weights，分别2/24完整head调用 | 已知wrapper，自身block law一致；不是原joint条件sampler |
| AR8 | 8个history-only ridge解析fits、16次MC | 固定系数普通law；非新neural训练 |
| FlowNP/SFM/ForcingDAS/MissPath-FM | 原文及部分固定代码接口审计 | 未实训，不能称已被本实验击败 |
| PathFM | 近期完整原文 | code/weights待发布，未执行 |

第一项独立检查8raw histories、48NPZ SHA、864moment rows，max probe差7.1054273576e-15、summary差7.77156117238e-16。第二项独立pairwise CRPS替代生产排序公式，16NPZ/256score cells、8AR拟合及递推，max score差4.4408920985e-16，AR replay差1.63709046319e-11，native及canonical第一prefix重放差0。两项model state同一SHA。6个same-noise API资格（长度672/768/1032 × native/reference）max relative差0；此API资格由worker执行，独立verifier没有再跑GPU API，不能夸为双实现模型全重放。

只有每task4个早期history、2个MC repetitions；rep不是训练seed，重叠origin不是独立样本。MC CI只描述同history抽样误差，不是现实泛化区间；864个probes未校正多重比较。质量分数使用真实未来192点，但这些是先前探索过的早期载体、非独立TEST，预训练覆盖未知。未测试新增真实prefix后的固定tail条件质量，未执行FlowNP/SFM/ForcingDAS的完整训练；coherence≠accuracy。
