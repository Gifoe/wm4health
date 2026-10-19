# 已完成的紧凑世界状态机制实验

在三个可重复的合成动力系统中，4D正逆预测表示可以几乎完美恢复动作，却没有保留当前可见、决定物理动作幅度的随机增益。增大表示维度基本修复，因此当前证据指向容量约束下的状态选择偏差，不能表述为所有世界模型均失败。

## 数据和控制

gain g_t在0.4/1.6之间独立采样，当前g随观测提供，下一步g未知。两维随机二值动作覆盖全部四种组合；训练/验证/测试gain和动作支持相同。q,p的更新由当前g调制。九个AR(.98)干扰因子与动作独立。生成因子的训练集标准化后作固定正交混合，模型不能读取物理变量身份。物理标签只用于评估和冻结表示探针。

每族192/64/128独立轨迹，各64个记录步，另64步burn-in；数据随机流与模型初始化分开。全部72个世界模型和探针选择完成后才计算测试指标。三seed结果是初始化变异，不能当作三个真实系统重复实验。

## 主结果（3个初始化的均值）

| Family |4D joint+inverse: gain accuracy | Action accuracy | Physical prediction MSE |4D gain-modulation NRMSE |12D gain accuracy |12D modulation NRMSE |
|---|---:|---:|---:|---:|---:|---:|
|coupled_linear|0.5080|1.00000|0.07040|1.0002|1.0000|0.1181|
|rotating|0.5012|0.99980|0.02247|1.0010|1.0000|0.1962|
|saturating|0.4994|0.99996|0.08600|0.9996|0.9999|0.1498|

Modulation NRMSE比较同一(q,p,nuisance,action)下当前gain取两种支持内值时的下一状态差。1约等于完全忽略gain造成的差，0为精确响应。它有已知生成真值，不是从观测相关性推断出来的因果标签。

4D joint+inverse对当前物理(q,p)的解码MSE均值分别为0.003191/0.001612/0.001884，表示有效秩约4，排除了整个表示完全坍塌这一简单解释。冻结表示再拟合更强successor probe仍接近gain-blind误差水平，问题不只在原先的转移头。

## 强探针与修复证据

27个冻结编码器的ExtraTrees分类/回归检查没有恢复4D joint+inverse的gain：九次分类准确率范围49.19%–51.16%。树探针的successor误差也没有通过验证集选择替代原探针。该结论仅适用于实际测过的读出族，不是严格零互信息证明。

12D joint+inverse恢复gain接近100%，原始观测MLP也达到100%。原始观测模型三个系统物理MSE均值0.00135/0.00112/0.00250，均通过预设验证readiness。双头控制在coupled_linear的一个seed恢复gain，另外两个seed失败；不隐藏这个反例，也不声称所有随机模型均失败。源码兼容的MoP/EMA/路由控制尚待cycle21独立测试。

## 完整目标对照

| Family | Method | Gain accuracy | Native physical MSE | Frozen-encoder successor probe MSE | Native modulation NRMSE |
|---|---|---:|---:|---:|---:|
|coupled_linear|joint|0.50557|0.903970|0.482023|1.00020|
|coupled_linear|joint_idm|0.50798|0.070398|0.069421|1.00016|
|coupled_linear|ema_idm|0.50899|0.082714|0.078966|0.99953|
|coupled_linear|gaussian_idm|0.50562|0.069732|0.070200|1.00020|
|coupled_linear|mixture_idm|0.66907|0.046500|0.046340|0.69047|
|coupled_linear|obs_idm|0.62122|0.064583|0.063865|0.87409|
|coupled_linear|fullobs|1.00000|0.001350|0.000397|0.07331|
|coupled_linear|joint_idm_full|1.00000|0.006954|0.003426|0.11805|
|rotating|joint|0.49418|0.630870|0.563649|1.00155|
|rotating|joint_idm|0.50122|0.022467|0.022753|1.00103|
|rotating|ema_idm|0.50203|0.024897|0.022893|1.00111|
|rotating|gaussian_idm|0.50037|0.021661|0.021897|1.00056|
|rotating|mixture_idm|0.50553|0.022277|0.022746|1.00217|
|rotating|obs_idm|0.57023|0.023994|0.022634|0.97574|
|rotating|fullobs|1.00000|0.001123|0.000277|0.10316|
|rotating|joint_idm_full|1.00000|0.004840|0.002516|0.19622|
|saturating|joint|0.49870|0.823126|0.384135|1.00009|
|saturating|joint_idm|0.49939|0.085999|0.084264|0.99958|
|saturating|ema_idm|0.51518|0.097729|0.090862|0.99170|
|saturating|gaussian_idm|0.50069|0.084351|0.084147|0.99989|
|saturating|mixture_idm|0.49593|0.083731|0.084167|1.00007|
|saturating|obs_idm|0.50631|0.086074|0.085372|0.99792|
|saturating|fullobs|1.00000|0.002500|0.001037|0.06774|
|saturating|joint_idm_full|0.99992|0.011339|0.006518|0.14981|

## 成熟敏感性降维对照（探索性，同一pilot数据）

| Family | Output target | Gain accuracy | Physical MSE | Modulation NRMSE |
|---|---|---:|---:|---:|
|coupled_linear|all_outputs|0.50370|1.015174|0.999998|
|coupled_linear|action_outputs|1.00000|0.000525|0.002748|
|rotating|all_outputs|0.50053|1.016512|1.000012|
|rotating|action_outputs|1.00000|0.001439|0.003324|
|saturating|all_outputs|0.50191|0.984924|1.000004|
|saturating|action_outputs|1.00000|0.000420|0.006819|

18次旧方法控制只改变输出目标选择，压缩器均为输入梯度协方差的前4方向，再用多项式ridge拟合转移。全部输出目标的gain约50%、响应NRMSE约1；动作响应目标九次gain均100%，物理MSE0.00035–0.00174、响应NRMSE0.00140–0.01429。没有用物理变量标签选择输出、输入投影或转移超参数；物理标签只在冻结后拟合诊断读出。

这证明无需新神经压缩器便能修复当前toy。成功控制本身共享“从teacher动作响应选择输出”成分，所以它尚不能否定目标选择问题，也不能证明该成分新颖。18个模型/teacher哈希、72个指标、验证选择和手写monomial完整重放通过；这是同cohort的探索性反证，不是新独立数据确认。

## 可复核性

主训练和探针共797.43s；72个权重、72个预测、9个数据文件哈希验证。504个测试指标独立重算最大差1.84e-7，8个完整batch checkpoint重放误差0，物理更新独立复算最大差2.35e-7。所有训练权重/探针选择文件先于测试预测生成。更强探针27个权重/预测另作哈希检查、54项算术复算和3456样本的逐树独立遍历。

初次preflight在写receipt时因outputs目录不存在退出1；创建目录后完整重跑，8种模型的同seed三步更新重放均为0。失败说明保留，主科学job一次完成exit0。源码中的梯度tensor转float日志警告没有改变训练步骤；没有中途修改科学脚本或依据测试调参。

![机制结果](outputs/analysis/causal_driver_pilot.png)

这些是合成、完整观测、单步、任务迁移证据；尚未证明真实任务频率、多步概率模拟、跨系统零样本能力或新方法优势。
