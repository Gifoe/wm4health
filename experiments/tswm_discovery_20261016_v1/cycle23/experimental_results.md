# Cycle23：实测与验证

128个全新WH系统；原生burn-in200、保留930；只替换cut400或744之后的iid N(0,1)输入；参数、历史原始输入和噪声不变。独立参考激励只用于fixed-reference管线，与审计分支无关。随机种子及假设先写入PROTOCOL.md，SHA256=1ff5fcf0e72f243ba77e92ecc1587263c5e9b7fe0cdcf0dacb4f989b24d60fdb。

| Simulator pipeline | Prefix cut | Median past-output change / original std | Systems above1e-8 |
|---|---:|---:|---:|
| raw_causal | 400 | 0.0000% | 0/128 |
| raw_causal | 744 | 0.0000% | 0/128 |
| native_internal_only | 400 | 9.8838% | 128/128 |
| native_internal_only | 744 | 5.2221% | 128/128 |
| native_complete | 400 | 13.1380% | 128/128 |
| native_complete | 744 | 6.7865% | 128/128 |
| fixed_reference | 400 | 0.0000% | 0/128 |
| fixed_reference | 744 | 0.0000% | 0/128 |


结果支持内部、非线性之前的全轨迹归一化依赖未来输入；关闭最终normalize不能修复此项。外部归一化还改变历史输入坐标。普通固定参考修复了本项模拟契约，但改变了生成协议；没有重训和性能收益结论。

16次未编辑WHDataset重放（8seed×normalize两值）float32输入输出最大误差0。独立NumPy递推核对10系统所有分支，最大输出差1.528e-13；独立参考常数差3.553e-15。128文件hash核对、6192指标重算差0.0、512个因果管线严格前缀检查通过。主模拟8.853s，独立核查3.526s。原始参数、信号和噪声NPZ留在服务器。

另24个老强baseline权重在服务器完成受限加载、作者best标记及预算检查；参数545038/652364/653366，ragged与tensor聚合最大差0，改变后续控制对此前输出影响0。这只是数值/接口验证，不是质量、校准或新机制实验。

工程尝试全部保留：首个文件传输慢，确认所属PID后停止，原任务4294967295；8路HTTP Range复用既有前缀，302.1s完成并通过完整发布SHA1。首个checkpoint预检因未允许NumPy标量而失败；仅加入数值类型后新job通过。没有将这些错误算作科学失败。
