# C41 实际运行结果

UTC 2026-10-17T00:21:25.577862+00:00。本轮研究单位仍是可复用时间序列 world-model 框架；数据不是优化终点。选择一个后续开发候选，不代表已接受新颖机制或完成总目标。

## 原生执行

LIFTS作者minimal例子实际训练、生成样本及接口核验通过，详见[qualification.json](outputs/native_v1/qualification.json)。本轮没有用5loops来比较准确率/收敛，也没有评分旧真实数据封存TEST。

## 控制A：精确belief与坐标规范

4record×4时间点的recursive vs block Gaussian mean/cov，以及4record×3horizon的observable gauge检查全部通过；后者独立dense24-history复算12项，max差 2.66e-15。这建立量尺，不证明任何学习模型状态有问题。

## 控制B：转移错误能伪装成校准收益

错误固定Q时，完美state posterior在h=1有KL regret .0320983，补偿性错误state达到约0；改回普通正确transition后全部h的regret为0。故不能从“校准state后预测变差”推出posterior inference有害。h=4/8的补偿仍不正确，全部9cells在[CSV](outputs/controlled_v2/compensation.csv)。这个经典控制不作为新贡献。

## 控制C：组合复用的信息边界及普通解

在3个训练view×dynamics cells，24个view对齐给出完全相同training law（max0）；held-out组合的transition law最大元素差 .18。普通Bayesian mixture同时使用anchor和held-out实际prefix，未丢弃新观测来制造失败。对128固定模拟：

| paired prefix | held-out prefix | entropy | oracle-relative KL | uniquely anchored fraction |
|---:|---:|---:|---:|---:|
| 0 | 1 | 3.178054 | 0.041490 | 0.0000 |
| 0 | 8 | 2.877678 | 0.045948 | 0.0000 |
| 0 | 32 | 1.930144 | 0.032894 | 0.0000 |
| 1 | 1 | 1.791759 | 0.039028 | 0.0000 |
| 1 | 8 | 1.501358 | 0.038339 | 0.0000 |
| 1 | 32 | 0.837008 | 0.019332 | 0.0000 |
| 2 | 1 | 1.594353 | 0.036916 | 0.0000 |
| 2 | 8 | 1.319208 | 0.035147 | 0.0000 |
| 2 | 32 | 0.736111 | 0.016674 | 0.0000 |
| 4 | 1 | 1.312957 | 0.031793 | 0.0469 |
| 4 | 8 | 1.082280 | 0.029452 | 0.0469 |
| 4 | 32 | 0.612676 | 0.014880 | 0.0469 |
| 8 | 1 | 0.825677 | 0.021021 | 0.2422 |
| 8 | 8 | 0.664901 | 0.016460 | 0.2422 |
| 8 | 32 | 0.333526 | 0.008821 | 0.2422 |
| 16 | 1 | 0.365173 | 0.010013 | 0.6094 |
| 16 | 8 | 0.291053 | 0.008298 | 0.6094 |
| 16 | 32 | 0.158282 | 0.004125 | 0.6094 |


![Ordinary Bayesian control](outputs/analysis_v1/anchor_information.png)

16个连续anchor观测只有60.94%模拟唯一确定对齐，不等于16个不同state；覆盖3个不同state时，四状态双射剩余映射即唯一，ordinary control逐cell预测差为0。不同held-out prefix长度对应不同下一目标；不可把三条曲线当同一target的单调定理。所有2304rows保留，包括有限样本非单调项。

独立permutation-matrix/product-likelihood重算9216指标cells，最大差9.55e-15；closed-form另核9compensation rows。评分的是冻结人工known-law控制，**旧真实数据TEST未评分**；不能笼统说本轮没有计算任何held-out synthetic score。未训练候选2×2，不声称方法收益。
