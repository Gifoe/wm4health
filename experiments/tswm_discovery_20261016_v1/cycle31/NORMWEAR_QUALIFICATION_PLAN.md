# 发布world model的执行资格与未来条件接口检查

目的：增加一个机制不同的、真实可运行的数值时间序列world model强底座；不把下载、示例通过、源码缺陷或普通修复当作创新。C31 streaming三视角目前无具体新机制，不因此启动切换模型长训练。

固定HF `mosaic-laboratory/normwear2` revision `3d1281c9994140056edfbbdedab57f2e6493dfaa`，9个小源码/配置/说明文件已在服务器下载并逐Git blob+SHA核验、只读检查。选择唯一safetensors 748201812bytes，SHA256 `44ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11`；不同时下载重复bin，不运行安装脚本或远端自动取代码。普通正常TLS优先HF镜像；所有权重留服务器。旧torch/全局环境不改。

先执行：原model card seed20263101的`rand(2,256,3)`，C128/H64，调用原`predict`；读取官方demo的纯代码cells（不执行整本notebook），选其中latent_bayesian_transition及simulate最小I/O合同。保留模型/源码版本、state_dict严格加载、完整调用参数、耗时、峰值显存与原始输出，不声称原论文表格复现。

对未来条件接口额外检查两个关键路径：

1. `simulate`接收完整已知future action channel，非可见target只给过去。源码每次decode后是否仅更新target，或也改写已知future action？用两步以上rollout及hook捕获每次实际encoder输入，核查允许信息，不凭最终RMSE推断。
2. `latent_bayesian.py`对 `(nvar,P,E)` reshape为`(P,E*nvar)`，之后对centroid最后`E*action_nvar`元素取action特征。用唯一整数标签的symbolic张量验证该切片是否真对应声明的action channels，并核查inverse reshape。此为实现路径算术检查，不是训练表现。

原生调用与任何必要ordinary修复分目录/版本保存。若出现实现偏差，先记录最小复现和等价/更正路径，不当作新Research Gap。没有因果干预真值的实物数据不宣称反事实正确；未做科学benchmark的运行只证明执行资格。

后续只有在源码/接口资格通过、强普通修复可执行后，才冻结新系统与机制测试；不使用示例分数选择新方法或假装已完成研究目标。
