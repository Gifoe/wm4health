# Baseline与真实执行状态

| 对象 | 固定版本/产物 | 本轮执行 | 不可声称 |
|---|---|---|---|
| 原生WH生成器 | PuraFilippo/sysid_benchmarks 4c20d94ccab51abf4ebf68aa8810f4596381dd41 | 16次原生数组精确重放；128新系统/两分支切点/四管线 | 不是新模型训练，也未复现新论文排行榜 |
| causal / fixed-reference管线 | 同参数/同噪声/独立参考校准；PROTOCOL.md | 配对模拟，512个严格前缀检查 | 不代表原checkpoint在新管线上预测更好 |
| Explore the Context | 修复分支f0baf6262523d9c5abaf8c0381b4db97818f7c34；作者最终预算validation-best | 24个权重，严格state_dict；原生前向；空/非空上下文；控制前缀 | 尚无新belief-update、主动规划或论文最终回报复现 |
| MPNP / Drift-to-Coherence / CNP理论 | primary papers | 文献边界 | 未执行，不能宣称比它们更好 |

发布修正版archive=2210754075bytes，SHA1=f24b54d4c817ac8d32fb80d1f0c8b08ededa4933，与作者发布值一致；SHA256=43b4600f58bb27ac59c3c43421896113c3abe7518a37253a2ffed1785019dc49。Archive在服务器，未下载到本机。仅提取24个final-budget最佳检查点，共57843216bytes；没有提取大型原训练信号。817个pickle候选member包含校准结果，不能将其全部计为模型权重。

主运行时仍Python3.10.20/PyTorch2.8/RTX5090。gym0.17.2纯Python依赖单独放在本cycle；未修改既有环境。可选torch_scatter只对本编码器实际使用的非负max聚合提供进程内适配；native tensor路径与ragged路径24/24完全一致。checkpoint仅允许必要NumPy数值标量全局类型，仍weights_only=True，未开启任意pickle执行。原生数值源码未编辑。完整配置、参数数目、权重hash和作者选中step见outputs/native_context_preflight.json。
