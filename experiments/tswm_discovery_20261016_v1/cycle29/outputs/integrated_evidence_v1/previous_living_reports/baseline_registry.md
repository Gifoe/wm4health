更新时间 UTC 2026-10-16T18:40:34.562241+00:00。状态：完整测试已独立复核。最终接受研究方向0；总体目标 ACTIVE、UNACHIEVED。

# 实际模型与数据登记

数据来自 pinned sysid_benchmarks，LTI 与 WH 两个结构家族；train1024/val128/test256 个独立系统。数据、源码、参数与种子见 outputs/world_data/manifest.json、systems.json；SciPy 独立验证通过。

| 模型 | 有效参数 | 身份 |
|---|---:|---|
| conditional96 | 91361 | 自主普通条件GRU |
| conditional128 | 122433 | 更宽容量控制 |
| compiled64 | 120705 | 已知低秩超网络思想的自主数值迁移 |
| native_attention64 | 230721 | 作者 TSTransformer 源架构、新数据/MSE训练 |
| ARX | 按context验证选择 | 测试时仅用128步context拟合 |
| pretrained_native_icl | 外部发布权重 | 预训练预算不同，单列参考 |

v1 已因 train-only 目标尺度问题停止并完整保留。v2 不增加数据、不删窗口，不使用未来y；训练与评估采用固定参考单位，外部 ICL 的原 context-only 变换单列。v2 完成训练 12/12，实际验证选择如下：

| 模型 | seed | steps | best validation MSE | cap 下仍下降 |
|---|---:|---:|---:|---|
| compiled64 | 2026291 | 32768 | 0.451683 | False |
| compiled64 | 2026292 | 9728 | 0.708333 | False |
| compiled64 | 2026293 | 29696 | 0.554497 | False |
| conditional128 | 2026291 | 32768 | 0.467781 | False |
| conditional128 | 2026292 | 13312 | 0.728333 | False |
| conditional128 | 2026293 | 14848 | 0.726674 | False |
| conditional96 | 2026291 | 22528 | 0.629552 | False |
| conditional96 | 2026292 | 12800 | 0.755987 | False |
| conditional96 | 2026293 | 15872 | 0.701618 | False |
| native_attention64 | 2026291 | 10752 | 1.031586 | False |
| native_attention64 | 2026292 | 20992 | 0.849585 | False |
| native_attention64 | 2026293 | 17408 | 0.932446 | False |
