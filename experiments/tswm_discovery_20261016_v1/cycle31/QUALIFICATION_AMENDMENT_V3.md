# 执行资格检查 v3：原生 latent Bayesian action 接口越界

v2 已严格加载 277 个参数，原 model-card predict 成功，simulate 及已知动作保持检查完成。原生 latent Bayesian 在 C128/H64、3 通道、action=[2]、use_cls=False 的合法完整 C+H 输入上发生 IndexError：输入未来动作只有 4 个 patch，采样器请求 5 个。

保留 v2 原脚本与 failure.json。v3 在独立目录先复现并记录原生 IndexError，再单独编译普通 patch-count 修复：在此无 CLS、context 为 patch 整数倍的合同中，未来编码切片与采样步数均使用 ceil(H/patch)，不要求预测区间之外额外未知动作。其余模型、权重、输入、seed 和检查不变。此修复不能宣称覆盖任意未对齐 context。

随后数值执行与未来 target 不变性检查须标记为 ordinary length repair 后的行为。原生 no-action 路径仍独立执行原函数；action feature 索引修复在 length 修复基础上另行记录。原生接口错误、普通修复均不构成新的研究方法或 Research Gap，不推断整篇论文实验结果失效。
