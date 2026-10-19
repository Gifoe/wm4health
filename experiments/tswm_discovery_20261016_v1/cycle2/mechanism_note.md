# Gaussian辅助界的罕见条件反例：形式与适用边界

这是本轮固定模型推导，不代表原生模型训练结果或已通过创新性核查的论文贡献。

取 prior p=N(0,1)，两个 full recognition f0=N(0,1)、f1=N(0,V)，reference类别概率p0(1)=eps。F=(1-eps)f0+eps*f1。定义proper RPM联合分布 p(z,x)=p(z)*p0(x)*f_x(z)/F(z)。两个观测类别的likelihood逐点相加等于1。

观测x=1后的精确density与 p*f1/F 成正比。其真实观测概率为 eps*Z，Z=integral p*f1/F；不是未经验证就等于eps。该posterior是严格log-concave、对称、单峰。log density可写成负的log-sum-exp正二次函数，因此不是依赖多峰性制造的反例。

通用Gaussian-shaped辅助函数 g=exp(t*z-h*z^2/2) 的有限Gamma条件是 h>-1/V（V>1且eps>0）。固定g时，内部界的最优q与 sqrt(p*f1*g) 成正比，方差为 2/(1+1/V+h)，严格小于2。非零线性项t不能提高对称构造的最优值：Gamma中每个component的t^2系数严格大于分子中的系数，所以最优t=0。服务器测试另用二维数值优化核对该结论。

令eps趋零，精确rare posterior趋向N(0,V)，普通Gaussian VI的最优KL趋零。V>2时，最优Gaussian辅助界q的方差趋近2；其目标gap趋向 -log(2*sqrt(V-1)/V)>0。这是辅助Gaussian族可积性造成的额外条件gap，不是Gaussian q族本身必然不够。

与此同时：TV(F,p)<=eps；KL(F||p)<=eps*(V-1-log(V))/2；相对Fisher(F||p)<=eps*(V-1)^2/V。最后一个界来自score差 w1(z)*(1-1/V)*z 及F*w1^2<=eps*f1。因此平均marginal可在这些量上接近prior，而没有同样的uniform rare-condition辅助gap保证。数值脚本独立积分实际KL/Fisher并检查这些上界。

这不反驳平均意义上的收敛：x=1本身的概率趋零，平均损失可以同时趋零。它也不反驳原作者完整的可识别性定理。要把平均marginal接近转成每个条件下的可积性/校准结论，需要额外的尾部或事件概率条件。

以上小marginal-divergence构造含V>prior的full recognition，超出当前native正site类。另有native类内构造：prior方差1，common full recognition方差.6，rare方差.999；对应正site precision分别为1/.6-1和1/.999-1。精确rare posterior在eps较小时方差接近3，而原生surrogate方差约.999，普通Gaussian VI几乎无gap。该构造不声称F接近prior。这个例子也说明：**原生proper RPM模型本身可能表达扩张，收缩约束直接作用的是其Kalman surrogate后验。** 不能把surrogate后验容量与整个proper概率模型容量混为一谈。

固定模型的反例不足以说明学习后必然失败。已追加：原生1000/10000步训练、独立告警对照、冻结ratio推断、普通生成式SSM，以及正在执行的已有RPM数值似然重新训练。后者用于区别训练时bound偏置与冻结参数后验误差。

数学修复如普通VI、负相对precision、EP、数值likelihood推断均已有，当前没有把它们重新命名为新方法。
