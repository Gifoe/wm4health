# Native checkpoint context-transplant results

# 原生物理复现与机制移植：宽泛缺陷假设被否证，局部残差保留

固定Meta-iLaD官方release `3d47140f18262628d252858c3efdfe2ad1e21a86`，同一个原seed1000 checkpoint。没有从头训练，也没有小网络替代。以下区分原生配方执行、绑定适配和我们新设计的机制审计。

## 原生输入与实际执行

原版gradsim CUDA前向device函数通过现有NVRTC/driver运行，原Python相机/投影/lighting/rasterizer默认值保留。Kaolin0.17的原OBJ导入函数在无materials/normals分支不变，SurfaceMesh接收端为identity容器；不是安装了完整旧版依赖。独立三角形alpha最大误差5.94e-6、相机误差4.77e-7、482顶点/960面完全一致、正常批次完全一致。旧source的batch3 cross轴问题原样保留并由显式空间轴对照解释；原生60帧生成没有走该分支。

生成train9000、test3000、OOD4000、trainenv3000，共19,000条序列，约4.7GB。四份完整source格式输入均通过有限值、形状、图像[0,1]、存储读回和参数/标签SHA检查。模拟状态及图像的source参考SHA未逐位匹配；图像sum仅作为sanity，不冒称像素相等。代码/原CUDA/独立几何/完整配方/实际模型数值共同支持functional qualification。

原版与refined两个evaluate.py各执行三个完整数据集：六个完整调用、同一权重。原队列的第四项在完整benchmark落盘后影子检查失败，恢复仅接续两个未启动的case；四个已完成benchmark未重跑。另有四次边界forward和两次decoder-only诊断，0次weight training。所有旧失败、原始数组和重建receipt来源保留。

| Source evaluator | Set | Native MSE independently recomputed | Strict-future MSE | Persistence future MSE |
| --- | --- | ---: | ---: | ---: |
| paper | test | 0.380685 | 0.379127 | 30.723507 |
| paper | ood | 0.608635 | 0.611515 | 30.557771 |
| paper | trainenv | 0.410832 | 0.406368 | 30.474347 |
| refined | test | 0.385332 | 0.379127 | 30.723507 |
| refined | ood | 0.609999 | 0.611515 | 30.557771 |
| refined | trainenv | 0.414889 | 0.406368 | 30.474347 |

36项指标从保存的实际预测图像和latent独立float64复算，最大差5.92e-8。Refined的18个仓库参考指标均在固定1%范围内，最大相对差低于1e-6；六个factual门槛全过。原版MSE按batch不加权平均、refined按序列平均，不能混表。原生MSE含查询已观测前三帧，strict-future列只取3..39。只有一个训练checkpoint，不能宣称多seed训练稳定性。

原始default GPU影子检查测得像素3.576e-7、latent/c均0。临时deterministic cuDNN的两次影子调用全部精确一致，恢复原flag后的同latent decoder重复又出现3.576e-7。旧pixel-exact失败不删除，也不改门槛；该差异有普通数值解释，不能当未来泄漏或新模型动机。

## 机制审计：准确的任务含义

20对ID test环境、160个完全相同物理初始角/速度，涉及38个环境。11条固定同环境context来自其他root；query和目标twin均不在context中。Mean-point z0来自事实前三帧，c来自独立context。目标机制在配对twin的time0生效。这是回顾式初态/机制移植，不是观测前三帧后才发生的在线干预，也不声称任意动作或真实世界因果识别。

| Diagnostic | Future MSE | Response MSE / zero-effect reference |
| --- | ---: | ---: |
| Naive fixed-context transplant |0.354192|0.057435|
| Ordinary prefix-only state fit then transplant |0.247927|0.046889|
| Target-prefix diagnostic, extra target observations |0.297731|not a fair deployable comparator|

Naive/target-prefix比1.1896，小于登记1.25；response ratio0.0574，小于登记0.25。源factual MSE0.3574、目标prefix诊断0.2977，各自persistence约30.6，点估计门槛通过。普通shooting只优化160个二维初态、60次Adam调用，选择仅看事实前三帧；模型权重完全不变。目标prefix诊断也不是最优oracle或误差下界，普通拟合优于它并不矛盾。

1,610项指标独立复算最大差1.57e-13；160对物理初态、全部context排除规则和prefix-only argmin选择均独立重建，prefix重算最大差4.82e-8。预设总体决定：`REJECT_CANDIDATE_NATIVE_CONTEXT_TRANSPLANT_SUFFICES`。这不支持用“现有模型缺少全局固定机制语义”作为新架构动机。

## 不用平均值掩盖失败case

逐对描述性筛查中5对naive数值超出同样阈值；普通状态拟合后仍有pair5、6、8。最差pair6（72→34）naive MSE0.7570、target-prefix0.4161、shooting0.5648，response ratio却仅0.0454/0.0373。pair8的naive/target MSE比1.0802，但小effect下response ratio0.3318，shooting后0.3118。两种指标暴露的残差不同，不得抹去。

这些是posthoc描述性failures，不能反过来改已经冻结的总体判据，也不能把8个root当8个独立新系统。保留[全部20对](PAIR_RESULTS_V1.csv)和[机器可读筛查](PAIR_FAILURE_SCREEN_V1.json)。它们尚未排除有限context估计、初态推断、效果信号大小与普通数值/近似误差。

## 下一项有信息增益的动作

停止宽泛的global-context缺陷路线，不追加DALI迁移训练或MoCAP分数优化。若继续这三个局部残差，先做**固定每组11条context、重复重采样**的诊断，不按未来误差挑最优context；再与固定更多context的extra-information控制分开报告。比较残差是否随context变化而消失、目标prefix预测本身是否同样波动，区分context估计与动力学响应。Bayesian context aggregation、显式状态辨识和联合后验已有直接先例；普通控制解释残差时，不制造新方法。该诊断尚未执行。

总体研究目标ACTIVE/UNACHIEVED，最终接受方向0。本轮实证关闭一个宽泛动机，未证明所有机制/所有初态/所有OOD均已解决，也没有完成三模型同任务排行榜。

[原生独立核验](../pendulum_native_closure_v2/VERIFICATION_V1.json) · [原生协议](../PENDULUM_NATIVE_EVALUATION_PROTOCOL_V1.md) · [数值恢复边界](../PENDULUM_POST_EVAL_NUMERIC_RECOVERY_V1.md) · [机制协议](../CONTEXT_TRANSPLANT_AUDIT_PROTOCOL_V1.md) · [机制实际执行](RESULTS_V1.json) · [机制独立核验](../context_transplant_verification_v1/VERIFICATION_V1.json)。数据/图像/权重/runtime只留G，GitHub仅紧凑来源、源码、配置、指纹、回执、CSV与报告。
