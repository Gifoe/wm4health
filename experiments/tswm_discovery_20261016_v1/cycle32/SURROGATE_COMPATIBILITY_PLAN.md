# 固定目标编码的 surrogate compatibility 检查

冻结于运行前，seed20263201。身份：自主有限分布数学诊断 + 原生 WorldTS loss 组件调用；不是训练 WorldTS，不是论文基准复现，不是已确认新 gap。目的是区分“原观测编码丢失”与“回归 Bayes 摘要丢失”。C11 的 sample folding 与 C21 的有限 driver bottleneck 不重复运行。

Y∈{-1,0,1}，C∈{A,B} 等概率，P_A=(.45,.40,.15)，P_B=(.60,.10,.30)。各 context 20 条精确计数样本，合计40条；不是随机MC估计。固定 injective 编码 phi(-1)=(100,100)，phi(0)=(103,100)，phi(1)=(100,103)。原始样本可精确线性恢复 Y=(phi1+2phi2−300)/3−1。无维度压缩/未知输入/不识别干预。

用公开 WorldTS revision269426ea6217d7152a244da0e209d922acd46d4f 的 LatentStateLoss/SIGRegLoss/VISRegLoss 原组件，10×MAE +15×flattened cosine；分别 reg0 和原active reg0.1。H1、fixed encoder、两个 context 的任意预测向量是函数类；不能声称结果覆盖自由 joint training 或 nativeH360 时序全模型。原组件与独立 NumPy 公式逐项相符才解释。

两个条件分布的 latent coordinatewise median 均m=(100,100)，而原Y的唯一MAE最优预测分别0与−1；最佳共用解码器风险.70，context可区分的Bayes风险.65。样本编码注入性/精确重构不自动保证其回归摘要的任务兼容。使用margin+cosine全局界证明同m为两个context的唯一最优，避免只以局部优化输出认定global。

精确full-target方差在两坐标均>1，协方差满秩；固定target无samplecollapse。原prediction variance penalty仍单列并纳入fixed-target全局界。先证loss差对||deltaA||1+||deltaB||1有严格正下界，再以固定随机扰动检查算式；随机检查不是全局证明。

强普通修复：1）保护独立raw-Y坐标，用其自身MAE监督、读其median，不冒称未保护cos/reg耦合的anchor有保证；2）平方latent回归得到两个不同的 Ephi，按恢复出的离散概率重新拟合/构造MAE Bayes读出，不把线性mean读出直接当median；3）允许普通 end-to-end observation supervision，尚未训练。

保存源码版本/SHA、完整40行、编码/预测/风险、全局certificate、native对NumPy差与全部扰动。独立 verifier 用分类概率/坐标分解重算。即使witness成立，已有 calibrated surrogate/PSR 理论也是强先例，只有真实learned-target模型在公平控制后出现同类余差才考虑新框架。没有真实failure则只记录理论边界，不推荐新方法。
