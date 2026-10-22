# 原生物理任务资格：模拟器已核验，模型输入尚未就绪

这是原生source模拟器资格检查，未执行学习模型、未新增训练。原MoCAP的18次模型评估统计不变。

固定源码`3d47140f18262628d252858c3efdfe2ad1e21a86`。仅新获取8个generator/renderer/mesh/dependency文件，101,784字节，按Git blob及SHA256核验。CPU原生生成train/test/OOD共16,000条60帧物理轨迹；参数、标签6个指纹完全匹配官方参考，3个state指纹未匹配。

18个固定轨迹与独立DOP853高精度积分最大差4.820987653e-05，小于预注册0.005。随后RTX5090/Torch2.8/CUDA12.8的同源码三次生成对照完成，CPU/GPU完整state数组最大差1.192092896e-06；6个参数/标签指纹仍匹配，但三个GPU state指纹仍未匹配源参考。不能把残差唯一归因于CPU/GPU；没有原始参考数组，aggregate sum近似不能证明逐点一致。

原生图像生成需要Kaolin0.17、gradsim及CUDA extension。执行的CPU与GPU环境均未安装renderer模块；隔离CPU PATH无nvcc/cl，标准CUDA目录和vswhere也未找到。这是有限位置盘点，未做全盘搜索，不能断言机器任何位置都没有编译器。未安装OS/驱动或变更共享环境。全部三项原生合成任务使用该类渲染依赖，替换绘图器不能冒称原模型输入资格通过。

源代码使用`theta_ddot = -2*gamma*theta_dot - omega**2*sin(theta) + 3*cos(pi*t)`、dt0.1、RK4步长0.05。论文附录的damping/forcing写法有差异，当前oracle严格以固定release源码为准，不能套另一条方程去宣称模型物理机制错误。

## 研究判断

[Meta-iLaD Section3.2](https://arxiv.org/html/2610.27180v1#S3.SS2)已在injectivity、环境充分变化、共享emission及适当context等条件下讨论全局环境坐标的可辨识性。这直接覆盖泛化的“学习固定机制坐标”动机。尚未验证这些条件在计划任务中是否满足，也没有合格物理checkpoint的跨root失败，当前不能接受其为新颖方向。原生factual能力合格后，才值得问：一个固定物理变化映射到同一latent编辑，能否跨起点复用；若现有模型或普通校准满足，立即淘汰这个动机。

后续优先资格核验现成原生vector-input物理模型，或在不改数据定义的前提下解决当前renderer；不再从头训练失败的DALI迁移配置，不围绕MoCAP分数开发方法。普通依赖、数值协议和未满足的可辨识性条件不是自动成立的研究gap。总体研究ACTIVE/UNACHIEVED，接受方向0。

[CPU协议](PHYSICS_QUALIFICATION_PROTOCOL_V1.md) · [CPU完整记录](physics_qualification_v1/RESULTS_V1.json) · [能力盘点](physics_qualification_v1/CAPABILITY_INVENTORY_V1.json) · [GPU对照协议](PHYSICS_CPU_GPU_CONTROL_V1.md) · [GPU完整记录](physics_gpu_qualification_v1/RESULTS_V1.json)。数据、向量和runtime均留G，未同步本地。
