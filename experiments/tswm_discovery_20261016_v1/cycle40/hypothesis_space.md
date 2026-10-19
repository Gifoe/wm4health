<!-- root-post-result-decision40 -->
本轮对宽泛的观测修正失败假设给出负面证据，而非支持它。下一机制实验应区分学习状态的信息、transition误差和inference误差；只有新交互有证据后再开发模型。SFM/state-space对照按具体残差需要选择，不自动扩表。

# C40 竞争解释与下一反证

研究对象是可复用时间序列世界模型框架；数据集只作机制验证载体。本轮是普通方法筛查，不是提出了一个新框架。当前工作问题：同一学得状态能否同时支持未来推演和真实新观测下的修正。最终新机制仍未成立，接受最终方向0，整体ACTIVE_UNACHIEVED。

| 假设 | 当前可测证据 | 不能混淆的解释 / 下一反证 |
|---|---|---|
| 表达/训练不足 | 完整learning curves、cap和模型容量 | 若cap仍改善，先公平延长/校准普通baseline；不称机制缺口 |
| 逐步生成误差累积 | AR固定B、r揭示下变化 | 联合生成/适当SSM的已有解法仍需比较 |
| 直接conditioning重算有成本 | Native FlowNP1/2/8/32曲线与cache AR计时 | SFM warm source、少步蒸馏及新高效prior可消除，不能泛化 |
| 状态学习与修正存在非平凡冲突 | 本轮没有直接证明 | 先有known-latent可辨识fixture区分信息损失/近似误差/conditioning；再测试新交互 |

两数据、两seed、固定有限预算、VAL反复用于选择；TEST未评分。VAL origins={'ett': 73, 'jena': 23}；达到训练上限时选中 2/8，最后一段VAL仍改善 6/8。不能将训练不足、普通AR误差累积或原生采样器开销称为新机制缺口。SFM、适当state-space和ordinary consistency等强普通控制尚未完成。
