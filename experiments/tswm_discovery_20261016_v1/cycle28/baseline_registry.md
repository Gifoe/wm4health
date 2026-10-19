# 基线、数据和复现状态

| 项目 | 固定来源 | 本轮执行 |
|---|---|---|
| PCFM 52,852,000参数 | Git226ddc37786ab0060ab1dbe7b0b0ebcb58b24630；HFce49c06f05efeb50bd307940392af0d8c89a3796 | 211,418,848-byte safetensors，SHA746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755；严格加载；全部5官方示例；另12个预列screen条件 |
| 原生概率 ICL 5,548,418参数 | PuraFilippo/sysid_benchmarks4c20d94ccab51abf4ebf68aa8810f4596381dd41；cycle16已缓存 | 90,193,824-byte checkpoint；仅允许argparse.Namespace的受限weights_only加载；原生noncached模型；两种预列初始化表达 |
| 训练内选参 ARX | 6个阶数×4个ridge值 | 各记录24个验证拟合，先300训练、100步free-run验证，再400refit；没有测试选参 |
| 持久性/训练均值 | 相同信息 | 两条记录全部保留 |
| CED 官方实测 | [Uppsala原档](https://uu.diva-portal.org/smash/get/diva2:1165531/FULLTEXT01.zip)，MAT SHA102c8d8012d6e850d50c13227b8c2280fe69f8bc4a845177f8bc3d761d152c89 | 两条500点记录，400校准+10初始化+90全测试；速度传感器不保留符号。两条记录不等于独立装置总体 |

PCFM 的6个图/glue设置不是6种独立模型；ICL两个初始化约定不是独立训练seed。所有模型可读相同410历史u/y和90未来u，未来y均置零占位。CED方向先验u→y也与普通SISO识别接口一致。PCFM图CP只从允许历史估计；显式lag0不使用隐藏真实延迟。

代码、计划和所有raw数组保存在服务器独立cycle28；safetensors和原始NPZ不被当作本地紧凑镜像。PCFM权重直接从服务器HF镜像正常TLS取得、计算完整SHA；没有改动旧环境或重下已缓存ICL权重。nonlinear_benchmarks1.0.1的官方划分源码经PyPI wheel hash固定，信号从原始MAT读取，不需要全局安装其依赖。
