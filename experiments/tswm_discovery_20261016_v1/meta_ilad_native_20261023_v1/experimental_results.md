# Meta-iLaD当前发布版：原生执行、精确数据与偏差归因

固定官方源码3d47140f18262628d252858c3efdfe2ad1e21a86（2026-10-21）。旧搜索摘要的“code soon”已被真实新发布代码替代。本轮MoCAP仅有作者refined版，包含重新调参的训练方案；这是该发布配方/现成checkpoint的执行，不是原论文Table2数值或整篇论文复现。

## 来源与执行

50个选定源码/元数据/权重文件按Git blob SHA及SHA256校验；9个现成checkpoint用weights-only模式加载且有限值。60个CMU原始AMC的SHA和6个split文件内24个张量指纹全部精确匹配源参考，TRAIN-only mean/std另经NumPy重算。原始数据、权重和预测数组留G。

Python3.11.9、torch2.4.1+cpu、NumPy2.0.1、torchdiffeq0.2.5、scipy1.15.2、matplotlib3.10.0的隔离环境运行9个原生评估单元。Meta-iLaD ID/OOD各3个checkpoint，IDF仅ID三颗seed42/1000/1111。观察前75帧、预测后50帧；误差先对50坐标求和再对时间/试次平均。Meta-iLaD使用另3条同subject的75帧context：ID取TRAIN，OOD取其他Test试次且排除query。已知分组用于构造context。IDF接收已见subject标签，未制造OOD标签比较。

| Native CPU recipe | Prediction mean | Sample SD across3 trained checkpoints |
| --- | ---: | ---: |
| meta_ilad / ID | 23.249255 | 4.024686 |
| meta_ilad / OOD | 78.990448 | 3.130609 |
| idf / ID | 26.780074 | 4.214500 |

这是同一发布任务内、按原信息协议分组的原生CPU结果，不与C88或WorldTS异任务拼统一榜单。独立float64复算主表24个指标，最大差4.05e-6。

固定seed原生重放与另一个未来观测篡改资格调用也完成，预测/reconstruction均精确不变；后者使用单独shadow文件，得分不计benchmark。主流程9个独立benchmark单元加2个资格调用共11次，0次新训练。

## 对不上仓库数字的失败如何被定位

CPU主流程Meta-iLaD的18个指标均进入预注册1%参考范围；IDF的6个比较中4个未进范围，最大未来分差约10.73%。保留这个原始结果，不修改门槛、不选择有利seed。

进一步7次冻结诊断在同一IDF权重上区分runtime、随机抽样与算术。记录器先精确重放原CPU2.4输出。CPU2.4与CPU2.8的251组Gaussian噪声和预测都完全一致。相同seed1000的CPU2.8/GPU2.8原生draw不同，MSE相差12.02%。固定整个251组噪声bank后，跨device相对预测RMS仅7.71e-7/7.18e-7，MSE相对差3.23e-7/0。

| Case, seed1000 | Gaussian bank | Future MSE |
| --- | --- | ---: |
| cpu24_native_s1000 | record | 29.487215042 |
| cpu28_native_s1000 | record | 29.487215042 |
| gpu28_native_s1000 | record | 33.032196045 |
| gpu28_cpu_bank_s1000 | replay | 29.487205505 |
| cpu28_gpu_bank_s1000 | replay | 33.032196045 |

GPU2.8的另外两颗固定seed也执行；三个GPU种子的全部6个指标都匹配仓库参考，最大相对差2.63e-7。原CPU主表和GPU诊断分别保留。这支持当前差异来自不同device的随机实现，不能推出原外部运行的完整环境已被唯一恢复。PyTorch[官方说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)本已讨论跨device/版本相同seed的局限；这是普通解释，不是新模型动机。

7次诊断另有14个指标和common-bank数组独立复算，最大标量差1.61e-6。全阶段18次实际评估、9个原始benchmark单元、0次新训练，所有源码与checkpoint未改。协议、输入、seed、日志和失败回执均保留。

## 对研究问题的实际影响

现已具备可执行、来源明确的第三项独立模型工作Meta-iLaD，与WorldTS/LatentTSF形成模型主线；TSWM继续作为benchmark/recipe，不能算第三种网络。各任务/监督/输入不同，不能称同任务三种完整SOTA复现或统一性能领先。

本轮证明发布模型可真实执行，并定位一个普通随机复现差异。MoCAP没有已知物理参数干预真值；这些forecast指标与未来输入资格不证明机械语义、因果反事实或新颖gap。当前不接受新方法。下一项应在有真实参数和可控初始状态的原生物理任务上，检验现有辨识方法是否已经满足固定机制语义，并先排除原生数据生成/信息条件问题。

[冻结原生协议](MOCAP_NATIVE_PROTOCOL_V1.md) · [原生执行与全部单元](native_recipe_v1/RESULTS_V1.json) · [数据/指标独立核验](native_verification_v1/VERIFICATION_V1.json) · [噪声归因协议](IDF_PARITY_ATTRIBUTION_PROTOCOL_V1.md) · [归因执行](inference_parity_v1/RESULTS_V1.json) · [归因独立核验](native_verification_v1/ATTRIBUTION_VERIFICATION_V1.json)。

总体目标ACTIVE/UNACHIEVED，最终接受方向0。原生复现成功不等于研究发现完成。
