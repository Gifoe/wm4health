更新 UTC 2026-10-16T19:51:23.870858+00:00。整体目标 ACTIVE_UNACHIEVED，已接受最终方向 0。此为过程报告，不能冒充最终已验证创新。

# 已执行实验与证据

身份：原生 API 执行资格 + 自主接口算术诊断。输入为固定 seed20263101 的生成张量，没有报告科学 benchmark MAE/MSE 或模型排名。

| 检查 | 实测结果 | 解释 |
|---|---|---|
| 原 card predict，B2/C128/H64/3 channels | shape [2,64,3]，finite；新进程重放 maxabs 0 | 原生预测可执行 |
| simulate，C128/H64，action=2 给定，target=0/1 | 后三次 encoder 的已知 action 偏差 [0.4835415482521057, 0.3149281144142151, 1.1713517904281616] | 原函数 decode 后改写已给定 action |
| ordinary target-only clamp | 四次 encoder 偏差最多 2.38418579e-07；输出 action 偏差 1.1920929e-07 | 保留 action 的普通修复；全通道预测时与原函数逐元素相同 |
| symbolic latent flatten，action [2]/[0]/[0,2] | 原末尾切片均不等于实际 action 特征；逆 reshape 精确 | embedding-major 交错布局不等于末尾 channel block |
| 实际 768 维 centroid 布局 | 原切片与显式 action 特征最大差 17.164328 | 数值确认同一索引偏差；不是性能分数 |
| 原 action Bayesian，C128/H64 | IndexError，索引4但仅4个 future patches | use_cls=False 时多采一个 patch |
| 官方 demo 长度 C2048/H2048，10 states，生成输入 | 同样 IndexError，索引128但仅128 patches | 越界不只出现于短上下文 |
| ordinary ceil(H/patch) 修复 | 短长度 [2,64]，demo 长度 [2, 2048]，finite | 仅已核查无 CLS、context patch 对齐合同；非通用实现声明 |
| 修复后改变全部 future target、保持 past/actions/seed | 输出差 0 | 此调用路径没有使用未来 target 真值 |
| 原 no-action Bayesian，仅 past 输入 | [3,64]，finite | 不给它任何未来目标 |

simulate 原接口返回 patch-pad 后 [1,208,3]，输入长192；记录该行为，没有静默改成原生正确长度。未在此判断 API 是否承诺裁剪。

独立 verifier 通过全部 29 个保存数组 finite 检查、NumPy action normalization/逐步输入重建、独立 centroid 轴还原、已观测 transition 计数重建（maxabs 0）及原 card 新进程重放。见 [qualification](outputs/normwear_qualification_v3/qualification.json)、[proof](outputs/normwear_qualification_v3/verification.json)、[demo 长度检查](outputs/normwear_demo_extent_v1/result.json)。

demo 长度的普通 length 修复调用用时 0.516s，CUDA peak allocated 794,075,136 bytes。这只是单次执行测量，不能与冷启动 card 调用或其他模型做延迟排名；reserved 包含 allocator 历史。

没有新训练、真实 counterfactual 真值、跨数据集科学比较或新模型。C29/C30 历史结论不被此次接口审计覆盖。
