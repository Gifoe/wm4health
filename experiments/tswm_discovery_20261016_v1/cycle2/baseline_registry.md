# 第二轮底座与复现注册

更新时间：2026-10-15T21:07:41.248013+00:00。这是第二轮工作记录；上一轮目录及结论未改写。研究总目标仍未完成。所有数值只从已保存结果读取。

|底座/实验|版本/配置|状态与身份|
|---|---|---|
|原先WorldTS/LatentTSF等|见上一轮source_registry与旧项目审计|已审计历史结果；本轮没有重复训练|
|PRO-DYN|733d986d349e9e71ded5d9cdeeeed8d3323d83b4|阅读实际DYN projection实现；未新复现，不能称为已训练world simulator|
|DoTime|8f227406a5b17c2040cfdcc680b493729ef47d00|实际调用metric/batching/VAR；新的受控AR1诊断，不是官方suite成绩|
|RP-GSSM|a75225d5edceb972bb872dc9a335608733567dd4|原生正精度映射与smoother已运行；1000步 6/6、10000步 6/6 个新任务训练完成|
|冻结RP + retained-F grid|训练集4096条reference；401/801 grid|初始评测 6/6、长训练评测 6/6；已有数值推断对照，不是新算法|
|普通非线性观测SSM ML|三颗种子×两个数据条件；相同10000更新预算|训练 6/6、评测 6/6；新控制任务迁移，非原论文复现|
|正确观测似然Bayes filter|真实AR/传感器/告警方程|有特权生成机制的归因oracle，不能与未知观测模型放在公平学习排行榜中|
|通用Gaussian VI|固定normalized RPM；160/320阶GH|16个已完成cell；这是标准已有方法|

原生运行使用本轮独立.venv_rpm，JAX/JAXLIB0.4.35 CPU、Flax0.10.2、Optax0.2.4、Dynamax0.1.5、NumPy1.26.4；具体源文件hash见native_rp_precision.json。GPU生成式基线使用既有只读PyTorch运行时；结果保存设备、Torch版本与checkpoint hash。旧环境与旧实验目录未修改。

原生学习固定正确prior是本机制实验的设计选择，不能当成原论文完整训练协议。所有训练参数、数据生成种子、划分、选择准则写在outputs/*plan.json。物理潜状态标签只用于冻结后的诊断读出与指标。没有测试集超参数搜索。


已有RPM数值似然对照：训练已保存 8/8，评测 8/8。配置见rpm_exact_training_plan.json；通用finite-grid forward实现已与joint归一化和有限差分梯度独立核对。该项不是新的概率模型。


本轮共26个训练及其评测cell全部完成（含不同预算对照），不是26套独立数据。新增RPM控制没有未结束的正式训练。
