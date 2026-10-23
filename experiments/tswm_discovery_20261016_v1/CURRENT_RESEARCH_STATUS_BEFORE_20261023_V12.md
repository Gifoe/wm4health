# 当前研究状态 — 2026-10-23 V11

总体研究目标ACTIVE/UNACHIEVED，最终接受新颖方向0。当前没有合格的新架构动机。

## 新增实跑证据

原生Meta-iLaD物理配方在现有5090上运行：原CUDA内核经NVRTC绑定、原相机/渲染Python及限定原OBJ解析，19,000条完整序列约4.7GB留G。参数/标签匹配，state/image参考SHA未逐位匹配；完整legacy栈未安装。六次原版/refined完整评估均完成，只有同一颗seed1000训练权重。36指标独立复算最大差5.92e-8，18个refined源参考均在1%内，factual门槛全部通过。原有失败/源码轴问题/数值对照全部保留。

新设计的20对环境、160个相同物理起点审计中，naive context移植MSE0.3542，额外目标prefix诊断0.2977，比1.1896；响应误差仅为zero-effect参考的0.0574，已过预注册整体标准。普通prefix-only状态拟合降至0.2479，0次权重训练。1,610项指标独立核验最大差1.57e-13。

因此停止把“现有模型缺少固定机制语义”作为宽泛新方向。逐对描述性筛查仍有三个残差环境对，必须保留并先排除context估计等普通解释；不改总体判据，不称所有工作负载无失败。[完整执行与失败归因](meta_ilad_native_20261023_v1/PHYSICAL_NATIVE_AND_TRANSPLANT_FINDINGS_V1.md) · [全部pair](meta_ilad_native_20261023_v1/context_transplant_audit_v1/PAIR_RESULTS_V1.csv)。

## 下一项最小反证

固定每次11条context重复重采样，禁止按未来误差挑context；更多context另列extra-information诊断。先定位三个残差是否来自context估计或动力学，而不是立刻设计新模型。该后续诊断尚未执行。

WorldTS、LatentTSF、Meta-iLaD既有原生执行继续分组，尚未完成同一任务三种完整SOTA公平比较。DALI不足factual能力的迁移路线不再追加，MoCAP的已解释随机分差不作为新动机。[前版状态](CURRENT_RESEARCH_STATUS_20261023_V10.md)。所有原始数据/模型/环境/数组留服务器G，仅紧凑代码与证据上传GitHub，旧文件保留。
