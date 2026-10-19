# Completed independent-cohort development

四个合成系统，3个初始化，10个学生配置，共120次学生训练和12个完整观测teacher。全部模型/探针选择固定后统一评估测试集。新轨迹、旋转和teacher权重均未复用cycle20；前三族机制相同，queueing是新的简单裁剪机制，仍不是真实数据。

所有学生64-wideMLP，4D状态；wide_budget为12D容量控制。joint/obs/reconstruction/wide/MoP为3000更新；四种teacher-bank与projected_mop为1500学生+1500teacher更新。辅助头48输出、weight1，BANK类有相同结构；teacher、辅助头参数和时间单独记录。这个更新预算对照不等于精确FLOP/墙钟匹配。原型MoP保留归一化、EMA.99和context-only路由，额外加入共同inverse/variance约束，是任务迁移而非原生OGBench复现。

旧方法控制24次：同一批12个teacher，全部输出或动作响应输出目标，前4个输入梯度协方差方向、train-only whitening、多项式ridge，由验证latent误差选阶数/正则。物理标签仅用于冻结后的诊断解码。既有压缩器与待核查目标选择必须分开归因。

- coupled_linear: teacher readiness=True, validation MSE=[0.001616, 0.001698, 0.001316]
- rotating: teacher readiness=True, validation MSE=[0.000986, 0.001024, 0.001037]
- saturating: teacher readiness=True, validation MSE=[0.002637, 0.002565, 0.002726]
- queueing: teacher readiness=True, validation MSE=[0.005275, 0.007198, 0.006087]

数据、120权重/原始预测、12teacher、24旧方法系数/预测和job日志保存在服务器cycle21/outputs。紧凑报告和结果镜像在本地；不声称所有大数组已下载。没有修改此前环境或公开发布。
