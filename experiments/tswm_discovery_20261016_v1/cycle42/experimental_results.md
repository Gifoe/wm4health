> 完成后的科学裁决：ordinary双侧factorization有人工heldout收益；新框架贡献尚未建立，seen代价与优化限制须同时保留。见[SCIENTIFIC_DECISION](SCIENTIFIC_DECISION.md)。

# C42 Actual experimental results

服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对131072条未来序列score和524288个next-probability cells，max NLL差4.25e-07。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。

24step sequences，prefix8/future16；8seen temporal cells/8held-out cells；每个world另32组四view同步无label静态anchors，system0下给所有模型相同信息。测试目标是有anchor的未见**动态组合**；不能声称完全没见该view–system的任何观测。原理、split和信息合同见[PROTOCOL](PROTOCOL.md)。

|N / seen cell|evaluation group|model|future NLL/symbol|excess vs oracle|next-symbol oracle KL|
|---:|---|---|---:|---:|---:|
|16|heldout|00|1.390066|0.269925|0.265922|
|16|heldout|01|1.300351|0.180209|0.182152|
|16|heldout|10|1.404499|0.284358|0.286939|
|16|heldout|11|1.238231|0.118090|0.121622|
|16|seen|00|1.146358|0.031359|0.034159|
|16|seen|01|1.156609|0.041610|0.045655|
|16|seen|10|1.143909|0.028910|0.031123|
|16|seen|11|1.156263|0.041264|0.046893|
|128|heldout|00|1.385295|0.265153|0.266961|
|128|heldout|01|1.249977|0.129835|0.131167|
|128|heldout|10|1.369497|0.249356|0.255656|
|128|heldout|11|1.219431|0.099290|0.103804|
|128|seen|00|1.119456|0.004457|0.005310|
|128|seen|01|1.137051|0.022052|0.024658|
|128|seen|10|1.118476|0.003477|0.004144|
|128|seen|11|1.139664|0.024665|0.027593|

每行均为4fits均值，不是四个独立数据集。完整raw rows见outputs/analysis_v1/all_results.csv。

交互 I=R10+R01−R11−R00，正值表示分离观测时动态factorization收益更大：

|N|group|mean I|range over 2 worlds × 2 initializations|
|---:|---|---:|---:|
|16|seen|-0.002103|[-0.004738, 0.001014]|
|16|heldout|0.076553|[0.006954, 0.112223]|
|128|seen|-0.003593|[-0.003792, -0.003218]|
|128|heldout|0.014748|[-0.007512, 0.032436]|

只报告descriptive range，不以两个world做可靠总体显著性判断。11最好、正交互或oracle excess下降均不足证明新方法或现实research gap。

![all initializations](outputs/analysis_v1/factorization.png)

精确likelihood枚举max8.88e-16、anchor枚举0、梯度finite-diff约1.41e-11；CPU8.0ms/update快于GPU17.9ms，资格权重丢弃。正式用CPU，其他GPU任务不动。独立验证结果见outputs/verification_v1.json，最终验证覆盖全checkpoint/score而不是复述训练日志。
