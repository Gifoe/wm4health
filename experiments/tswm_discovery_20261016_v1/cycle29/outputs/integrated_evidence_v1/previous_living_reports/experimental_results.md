更新时间 UTC 2026-10-16T18:40:34.562241+00:00。状态：完整测试已独立复核。最终接受研究方向0；总体目标 ACTIVE、UNACHIEVED。

# 完整实验记录

数据独立验证：{'train': 1024, 'val': 128, 'test': 256}；全部256个测试世界的分叉前缀最大差0；SciPy+NumPy独立重算64条分支通过。

v1 训练目标审计按实际262144窗口计算：最高0.1%窗口贡献归一化目标能量的95.7408%。这是目标能量，不能冒充已训练模型误差。修订前后验证损失单位不同，不能直接比较数值作为提升。

v2 训练完成 12/12，模型选择只看独立验证系统。

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

## 测试

| 家族 | 方法 | factual RMSE | changed-plan effect RMSE | 训练种子数 |
|---|---|---:|---:|---:|
| lti | conditional96 | 0.518685 | 1.184998 | 3 |
| lti | compiled64 | 0.432509 | 0.927933 | 3 |
| lti | compiled64_zero_operator | 1.076453 | 2.718607 | 3 |
| lti | compiled64_swap_operator | 1.217127 | 3.114653 | 3 |
| lti | conditional128 | 0.478670 | 1.161889 | 3 |
| lti | conditional128_swap_code | 1.205915 | 2.952769 | 3 |
| lti | native_attention64 | 0.934855 | 2.213042 | 3 |
| lti | native_attention64_chunk32 | 0.773906 | 1.978284 | 3 |
| lti | arx_context_selected | 0.006396 | 0.014222 | 1 |
| lti | pretrained_native_icl | 0.837783 | 2.379410 | 1 |
| wh | conditional96 | 0.742050 | 1.552423 | 3 |
| wh | compiled64 | 0.708764 | 1.557241 | 3 |
| wh | compiled64_zero_operator | 1.101849 | 2.348906 | 3 |
| wh | compiled64_swap_operator | 1.195290 | 2.448825 | 3 |
| wh | conditional128 | 0.734700 | 1.524709 | 3 |
| wh | conditional128_swap_code | 1.186203 | 2.334392 | 3 |
| wh | native_attention64 | 0.993575 | 1.802091 | 3 |
| wh | native_attention64_chunk32 | 0.878406 | 1.688486 | 3 |
| wh | arx_context_selected | 0.676759 | 1.953751 | 1 |
| wh | pretrained_native_icl | 0.985830 | 1.936135 | 1 |

表中为每系统 RMSE 的均值，效应排除 factual 零差分。外部预训练 ICL 不匹配本轮预训练预算；全部原始系统/seed/计划/时域见 CSV。

生成算子相对近参数量条件 GRU 的效应 RMSE 变化：lti=-20.14%；wh=+2.13%。该比较本身不能成立新颖性或跨真实系统结论。

测试若完成，必须与 outputs/world_verification/model_comparison_v2.json 的全量重算相对应；图是合成跨系统诊断，不是原论文排行榜、真实系统泛化或因果临床证据。
