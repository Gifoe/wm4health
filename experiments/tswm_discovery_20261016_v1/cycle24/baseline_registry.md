# 模型、数据与运行身份

- 原生模型：EmbodiedVision/explorethecontext，修正模型选择分支f0baf6262523d9c5abaf8c0381b4db97818f7c34。六个作者validation-best检查点；action_squash_1/2各3训练seed，原始50k训练预算；无新训练。三种seed不是三份独立测试数据。
- 两变体共享一个两状态plant，只改变actuation map。每变体64个新alpha/初态/控制/噪声实现，共128；源模型训练数据本轮未下载，未进行原训练参数ID逐一重叠核查，不声称严格原参数集合去重。新生成流及新轨迹与现有评测分离，不通过新指标选择checkpoint。
- 原物理核、参数范围、噪声、控制支持不变。连续历史前缀n=0/2/5/9，避免随机漏掉的历史让“精确后验”定义错误。query从独立固定状态0开始，probe也从0开始；区分系统信念与当前状态变化。测量不是临床或真实设备数据。
- 对照：不更新；原生重新编码；同decoder完整一维Bayes积分；单步EKF；MAP+GGN局部Gaussian；defensive proposal importance校正+Gaussian矩投影（256/1024/4096粒子，每条件3MC seed）；已知物理alpha后验。最后一项是privileged reference，不能冒充通用学习模型。
- 主48条件在看结果前固定；普通滤波/粒子对照为事后诊断，计划先于各自执行，但不算独立确认。所有方法接收相同真实probe/query，未使用query结果选择粒子数。

运行环境：服务器RTX5090、PyTorch2.8+cu128、Python3.10、原生float32数值网络，概率积分/聚合float64。source和weight不变，旧环境未修改；复用cycle23隔离的legacy依赖与已验证max适配。详细配置/sha、原始日志、所有失败/修复尝试和检查点来源均保留。大数组在服务器，本机紧凑镜像。
