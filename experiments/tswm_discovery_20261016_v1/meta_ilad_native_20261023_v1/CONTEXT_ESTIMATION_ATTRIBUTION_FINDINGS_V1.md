# 普通推断已解释大部分局部机制移植残差

结论：停止把这组Pendulum残差作为新动力学架构的依据。固定原模型与已有观测，仅做普通环境码/初态估计，future反事实MSE从0.354192降到0.174291。该实验是归因诊断，不是新方法或新SOTA；另外两对略超描述性阈值的残差完整保留，不继续调预算把它们压线。

## 这次到底运行了什么

沿用已核验的Meta-iLaD原生seed1000 Pendulum checkpoint、原版物理输入、20对环境和160个相同物理起点。涉及38个不同环境，环境对存在复用；160行不等于160个独立系统。全部分析使用posterior-mean点预测，strict future为3..39，图像MSE为逐帧像素求和再平均。不要与原生MC评估或像素平均MSE混合。

1. **固定context数量重采样。** 原11条、额外信息32条、64组固定11条，共66组，四条预测路径。所有组完整保留；不按future误差选context，不做新的state或weight训练。保留约6.92GB实际预测图像在服务器G盘，独立复算全部73,920项图像指标，最大差4.49e-14；10,560个仿射头分量最大差1.16e-6。embedding/cache与原调用、旧完整预测重放也通过预先限定容差。
2. **同观测联合latent推断。** 每个环境仍是原先同样11条、各40帧context；只拟合11个初态（Control A），对照联合拟合这些初态和一个二维环境码（Control B），各100次Adam更新。随后在事实query前三帧做相同60次初态拟合。参数选择只使用各环境context或query prefix损失，绝不使用query future或目标twin。原网络权重哈希不变。200次context优化调用加60次query优化调用，0次网络权重训练，不是200个独立训练模型。

所有四个执行/核验进程均终止且exit=0，原始stdout日志SHA与回执匹配。联合推断另有355项独立评分检查（最大差4.85e-14）、全部argmin/排除规则核验和5次新鲜原模型前向重放（最大像素差2.93e-6）。独立核验完成状态以新VERIFICATION为准；不可修改原RESULTS中当时的pending字段来抹去执行时间线。

## 失败现象如何被拆解

| 原先残差对 | 64组固定11条中旧拟合状态仍超阈值 | 32条额外信息控制 | 联合推断后的CF/目标prefix诊断比 | 联合推断后的归一化响应误差 |
| --- | ---: | --- | ---: | ---: |
| 5：49→11 | 28/64 | 比1.2490，进入描述性范围 | 1.076661 | 0.024277 |
| 6：72→34 | 59/64 | 比1.3425，仍超范围 | 1.282638 | 0.024231 |
| 8：56→18 | 64/64 | 响应比0.3421，仍超范围 | 0.886594 | 0.256803 |

阈值为CF/target-prefix诊断<=1.25且响应/zero-effect<=0.25。这里的逐对筛查从一开始就是posthoc描述性筛查，不能倒改原先已通过的整体判据。32条改变了信息量，不能声称与11条公平的方法提升。重采样中fitted初态固定来自原source context，因此对应响应只解释target-context敏感度；没有伪装成每组重新拟合的source初态。

第5对既有context-set敏感性，也被普通推断解释。第6对在多数抽样下仍有残差，但联合推断使其ratio从1.3574降到1.2826，只比描述性边界高约2.6%。第8对的反事实MSE比本来合格，问题在小effect下的响应相对误差；联合推断后0.2568，只比0.25高约2.7%。保留绝对误差和zero-effect分母，不把两种残差说成同一种动力学失效。

## 全部样本的结果与普通控制

| Query推断路径 | 反事实future MSE | 事实future MSE | 响应误差/zero-effect |
| --- | ---: | ---: | ---: |
| 原amortized c/z | 0.354192 | 0.357366 | 0.057435 |
| 原c，只拟合query初态 | 0.247927 | 0.285823 | 0.046889 |
| context联合拟合c/z，query初态仍amortized | 0.293725 | 0.261097 | 0.043021 |
| context联合拟合，再拟合query初态 | 0.174291 | 0.175925 | 0.031127 |

表中的后两行使用同样已有输入但增加了推断计算；这是归因对照，不是公平计算量的新方法比赛。精确机器值以[独立核验JSON](joint_context_refinement_verification_v1/VERIFICATION_V1.json)为准。target-prefix诊断从0.297731变为0.240718；它含额外目标观测且未经最优状态求解，既不是可部署公平对手，也不是误差下界。

已观测context的MSE：初始0.349695，只拟合初态0.179478，联合拟合0.129552。两种100步context控制分别约42.16秒和41.27秒，但只适用于本次GPU实现。环境码最大位移0.4242，按原encoder对角标准差归一化最大4.5688；这不是后验校准证据，也不证明环境码等于真实物理参数。

## 新颖性与停止决定

[T2-z](https://arxiv.org/html/2610.22364v1)已研究冻结world model、从观测交互测试时优化环境latent；我们没有执行其代码或权重，不能把它列为本轮复现基线。这次把普通latent拟合原则用于已有Meta-iLaD做归因，不构成新机制。另见[既有上下文/状态辨识边界](../cycle69/LITERATURE_DELTA69_V1.md)。

不追加该Pendulum工作负载的优化步数、学习率或种子来消除两个边界残差。不把“有限context有误差”“latent test-time optimization有用”包装成创新。当前仅一个训练checkpoint、一个合成物理系统、回顾式目标机制从time0生效；不是现实在线干预，也没有跨系统新颖性证据。整体研究目标ACTIVE/UNACHIEVED，接受的新颖方向仍为0。

后续新候选必须先通过历史负证据与现有方法筛查，并在独立系统/预先冻结的能力测试中证明普通辨识控制仍不足；这两个残差不承担该证明。新增网络应晚于这项门槛，而不是为了凑第三种模型或提升本数据集分数。

## 可复现入口

[重采样协议](CONTEXT_RESAMPLING_PROTOCOL_V1.md) · [重采样完整结果](context_resampling_audit_v1/RESULTS_V1.json) · [独立核验](context_resampling_verification_v1/VERIFICATION_V1.json) · [联合推断协议](JOINT_CONTEXT_REFINEMENT_PROTOCOL_V1.md) · [联合结果](joint_context_refinement_v1/RESULTS_V1.json) · [联合独立核验](joint_context_refinement_verification_v1/VERIFICATION_V1.json) · [全部20对重采样CSV](context_attribution_closure_v1/CONTEXT_RESAMPLING_PAIR_RESULTS_V1.csv) · [全部20对联合推断CSV](context_attribution_closure_v1/JOINT_REFINEMENT_PAIR_RESULTS_V1.csv) · [七份研究交付索引](context_attribution_closure_v1/final_research_direction.md)。

数据、图像、模型、运行环境和完整日志留服务器，GitHub仅上传紧凑源码、配置、指标、来源、哈希和执行回执；无本地payload同步。旧失败与旧结论按时间保留。
