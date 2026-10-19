# C34 实际执行与结果

类型：已发布生成器的接口资格、独立重放与成本测量。没有 proposal 训练、正式事件误差比较、预测校准结果或新方法验证。

冻结 QUALIFICATION_PLAN.md 后在服务器 RTX5090 / torch2.8+cu128 / float32 / TF32关闭执行。原生50步、720维高斯源均保持不变；H=96、history=672。确定性数值 fixture 仅检查API，不作为科学数据；另两项使用真实公开solar和Jena历史。

三种fixture中，API generate 对缓存原生flow、显式噪声adapter及2/4样本前缀的相对差均为0。独立新进程直接重放原始FlowLoss.sample，12项输出/conditioning检查最大绝对差0。原始源码梯度对中心差分对称相对误差0.0012553656；整个模型state前后相同。生产运行10.529秒，独立复核6.471秒。

| paths / 同一固定history | 三次同步计时中位秒 | 峰值分配字节 |
|---|---:|---:|
|32|0.0799724|585870336|
|64|0.0785686|587240448|
|256|0.0797269|595462144|
|1024|0.0835421|628809728|

计时仅包含缓存history后的采样，不含加载权重、原history编码、proposal训练或真实多context负载。小batch受启动成本影响，不能按32path的单位耗时外推1024path基线。这里只说明批量普通采样相当便宜，未测概率估计MSE或训练摊销的盈亏平衡。

H>720时原生实现追加ensemble median到共享历史，不能宣称独立祖先采样，也没有在本轮更改其定义。当前资格只覆盖H96。数据缺失值没有通过删行压缩时间；最早有限输入的确定性选取由独立实现核验。未建立正式训练/验证/测试事件cohort。

result.json SHA f804c7fe754bd0e8a8d33b37f333e1a57e32e57efad2ddaf1c0b066797ee0c35；verification.json SHA d6e95a68ecc427b2a52c27a8365039582e2cfd6dead4c01ec8a97c14560525a3。
