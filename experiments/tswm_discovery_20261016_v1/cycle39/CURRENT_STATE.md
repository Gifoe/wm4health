# C39 两项诊断完成，目标未完成

UTC 2026-10-16T23:31:05.545952+00:00。原生query诊断与普通kernel控制均完成且独立复核；最终接受方向0，整体ACTIVE_UNACHIEVED。新neural updates0；只有8个history-only AR8解析fits。

第一项独立检查8raw histories、48NPZ SHA、864moment rows，max probe差7.1054273576e-15、summary差7.77156117238e-16。第二项独立pairwise CRPS替代生产排序公式，16NPZ/256score cells、8AR拟合及递推，max score差4.4408920985e-16，AR replay差1.63709046319e-11，native及canonical第一prefix重放差0。两项model state同一SHA。6个same-noise API资格（长度672/768/1032 × native/reference）max relative差0；此API资格由worker执行，独立verifier没有再跑GPU API，不能夸为双实现模型全重放。

普通canonical kernels无需训练就定义一致的自身block law，但改变原joint law。canonical96约2.6倍原single-head采样耗时；本小样本上solar mean CRPS变差、terminal却改善，Jena主分数变化小；canonical8更慢且主分数更差，AR8便宜但质量混合。不能只挑一个metric断言新框架必要；也不能称普通一致性修复没有实际代价。当前保留的是固定future-tail真实reveal下的quality/cost问题，尚非被证明未解决的机制。

原V1失败、V2成功、kernel控制各独立目录保留。所有raw samples/weights/source在服务器；紧凑JSON/报告/plots按SHA镜像。只有每task4个早期history、2个MC repetitions；rep不是训练seed，重叠origin不是独立样本。MC CI只描述同history抽样误差，不是现实泛化区间；864个probes未校正多重比较。质量分数使用真实未来192点，但这些是先前探索过的早期载体、非独立TEST，预训练覆盖未知。未测试新增真实prefix后的固定tail条件质量，未执行FlowNP/SFM/ForcingDAS的完整训练；coherence≠accuracy。
