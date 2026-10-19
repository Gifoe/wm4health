<!-- root-post-result-decision40 -->
新增LIFTS源代码缓存：e42430980e51bebf2e1b54d1beddeda7a31dee51，24 files/409834 bytes，未import/训练。独立stdlib原数据重建：时间戳/缺失mask相等，ETTh1数值最大解析差7.1054e-15、Jena0；bitwise verifier原失败保留，数值容差2e-12，不改训练输入。24项未知future污染检查全部预测差0。

# C40 实际代码与数据

研究对象是可复用时间序列世界模型框架；数据集只作机制验证载体。本轮是普通方法筛查，不是提出了一个新框架。当前工作问题：同一学得状态能否同时支持未来推演和真实新观测下的修正。最终新机制仍未成立，接受最终方向0，整体ACTIVE_UNACHIEVED。

- GRU2x256 + 8 Gaussian mixture：631960 parameters，新的共同协议普通baseline，不是官方SSM论文复现。
- Native FlowNP 128x4, ff256：604673 parameters；保留官方target-wise random time、public noisy/augmented sampler及.loc样本语义。架构缩放/数据合同迁移明确，不冒充headline reproduction。
- 每family每数据2seed，共8fresh fits/16000updates，AdamW/batch32/2000step，全部曲线和selected/last checkpoints在服务器保留。
- ETTh1 source commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07`；OT物理单位原README未明确，不猜测。Jena原始字段T (degC)，2016两zip。原字节来源/SHA见outputs/intake.json。Jena74缺失小时槽不插值，完整窗口筛除。
- Runtime依赖仅cycle40/vendor，旧venv/global配置和旧成果保留。

两数据、两seed、固定有限预算、VAL反复用于选择；TEST未评分。VAL origins={'ett': 73, 'jena': 23}；达到训练上限时选中 2/8，最后一段VAL仍改善 6/8。不能将训练不足、普通AR误差累积或原生采样器开销称为新机制缺口。SFM、适当state-space和ordinary consistency等强普通控制尚未完成。
