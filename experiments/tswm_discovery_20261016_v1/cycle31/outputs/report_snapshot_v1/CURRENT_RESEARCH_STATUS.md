<!-- cycle29-current-start -->
# 当前：通用时间序列world model证据筛查

UTC 2026-10-16T19:21:51.067123+00:00。C29原始12/12训练、主/真实/次级/等价实现复核通过；等更新对照已复核=True。接受方向0，目标ACTIVE_UNACHIEVED。跨家族证据混合，未把已知编译机制包装成新方向。

最新材料见[完整证据](cycle29/experimental_results.md)、[研究判断](cycle29/final_research_direction.md)与[C30负面transport诊断](cycle30/final_research_direction.md)。以下历史保留，仅指当时轮次。
<!-- cycle29-current-end -->

# 当前研究状态：通用时间序列世界模型

研究目标已按用户澄清限定为可复用的状态学习和条件演化框架。WorldTS作为问题层次的参照；数据集只是证据载体。预训练、Loop、多模态和动作输入均不是强制前提。

最新完整材料在cycle28；cycle27保留原生受控EM正控制；cycle26保留输入离散接口反证；cycle25保留原生高维belief审计；cycle24保留scalar belief审计；cycle23保存生成器契约审计及发布模型底座，cycle22保存真实任务比较，cycle20/21保存受控机制pilot及独立cohort。此前结果保留。

## 最新证据与判断（cycle28）

前轮cycle27=PROGRESS。本轮回到近期真正发布的数值world-model框架，避免继续微调已解决toy。三个独立文献视角按2026-04-16..10-16窗口核查现代框架、失败评测和相邻有限预算方法；检索日期10月16日，实验在北京时间10月17日完成。仍允许有限数据/计算优势构成创新，不要求经典方法在无限计算下永远无解；但本轮最终接受方向仍0。

新底座PCFM为52,852,000参数的any-variate图/未来条件模型，官方Git226ddc37786ab0060ab1dbe7b0b0ebcb58b24630；HFce49c06f05efeb50bd307940392af0d8c89a3796。211,418,848-byte safetensors直接在服务器经HF镜像正常TLS下载，14.219秒、完整SHA746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755；6源码Gitblob/SHA核查。没有重下已有ICL权重或改旧环境。

全部5官方example.main示例按原seed7/常数/顺序执行，15APIcalls；另12个预列计划/glue条件与1个masked-target占位检查，共28推断。ON/OFF/PULSE32原生valve MAE0.356464/0.231219/0.661514°C，ON-OFF效应MAE0.262196°C。独立seeded数据/物理递推/打印分数复核通过；28全调用同源新进程数值重放max6.104e-5，满足冻结float32容差。不是完整原论文表格复现或独立实测因果证据。

单作者window的严格延迟前仍有小提前响应，ON-OFF前48步RMS约.270°C(glue开)/.220°C(关)，其他计划最大提前差1.392°C。masked未来target换123456后输出差0。这仅是小幅合成契约诊断，未排除普通因果mask/估计修复，不能升格新框架；9边际分位数也未被误当joint sampler。

新增真实Coupled Electric Drives官方数据：两条各500点记录(同一装置低/高输入幅)，各前400校准+10测试初始化+90完整free forecast，无后续真实输出回灌。PCFM、已缓存原生概率ICL、训练内选参ARX获得相同410历史u/y与90未来u；不生成不存在的真实反事实。预列PCFM CD/known-direction/CP各glue开关，ICL native-padded30/direct10，ARX与两简单对照共22行。ICL公开约定的20个前置零是标准化占位，不增加观测。

真实RMSE(低/高幅)：ICL native0.067266/0.076028；PCFM默认CD+glue0.336438/0.434848；同权重公开glue=False变为0.188532/0.154891；ARX0.379986/0.430485。完整图/初始化模式均保留，不以测试选择最好。较专门的5.55M概率ICL是已有强模型；不能据PCFM默认低分宣称新通用框架必要，也不能将参数量差归因成架构或离线compute公平优势。

CED独立复核22预测hash、48训练内ARX验证refits、2000模型input值、142分数通过；增广LS/状态协方差递推与producer差<9e-15，12PCFM+4ICL神经重放差0，分数float32差max3.974e-8。两条短记录不是独立系统总体；没有把90时间点当IID显著性样本。只报边际覆盖与factual质量，不当joint动力学/干预真值。native/真实主运行分别8.047/7.859秒，峰值GPU约496/254MiB；无新神经训练。两结果SHA分别724119ab9d7794df8cb81c7523dd2d83fcbf11a6289cf46008a70e952f6aae95与1869e6ff8d46ae963168d6125fb66d3b6b3d2ffcdfa7ad98c387771d00c2d30c。

2610.08118已直接研究现代TSFM输入衰减、合成forced-system适配及保留代价；2610.13411是cycle16已读/下载的老边界，本轮只完成首次本项目真实质量比较；VSF明确trajectory参数随机而非process noise且目前无可执行作者发布。AURA/KAFFEE也已有可更新latent/滤波目标。三视角均未保留新lead，旧primitive存在不自动封锁组合，但当前没有有限预算下存活的独立新残差。

本轮7份中文报告、28原生调用、22实测预测、所有计划/数据/源hash/日志/PNG-SVG-PDF保留双端紧凑镜像，权重与raw NPZ留服务器。没有待等待训练/外部阻塞。停止调这两条CED及已修复toy，不把glueFalse、padding或causalmask重命名创新。下一项方法训练前仍须定义一个在本轮强ICL/PCFM及普通修复下失败的系统级能力并用独立机制证实。

cycle28=PROGRESS；总体目标ACTIVE、UNACHIEVED；最终接受方向0。以下是历史记录，不能用其中旧“当前优先方向”覆盖此节。

## 前轮证据与判断（cycle27）

通用、可复用的时间序列world model是研究目标。当前无已接受最终方向；不把某个数据集提分、源码修复或已知滤波方法包装为新框架。

已完成原生受控部分观测动力学的成熟方法正控制：官方KoopmanGeneratorEM源码固定ef830a6d4a6cff699492d54230fc2ad1729cab20，原生slow-manifold机制、标量带噪输出、5D含常数lift。只用50轨迹前250观测训练，三个初始化共享同一训练集；独立dwell50/5/100各64新轨迹，前缀25/100/250、未来251步。固定物理参数，不是64种不同系统。未来控制计划可用，未来观测/真实隐状态不提供给学习方法。

精确joint-Gaussian检验证明官方generator M-step的lag covariance方向错误：smoother给Cov(z_k,z_k+1)，回归却需要转置。修正只涉及这一充分统计量，不动process-noise公式；不是新方法。source-equivalent批处理先对齐未修改原生100步（最大参数差4.56e-11、LL差2.76e-9），再保留原日志/权重并记录受控停止，仅替换自有慢worker。三seed×source/corrected两模式共六拟合；全部拟合完成后统一评分。

独立同分布前缀250、filtered预测的干净MSE跨初始化均值：source2.007956→corrected0.008857，下降99.5589%；快速切换0.044309→0.0004196，慢切换8.883271→0.015383。标准修复后的成熟模型已经胜任此例，拒绝以原生失败论证新架构必要性。其95%边际覆盖率分别97.535%/96.039%/97.101%；仅边际校准，不是joint path证明。ARX、持久性及特权physics诊断全部保留。

预算100/训练最佳final、time0-smoothed与last-prefix-filtered两个推断约定全部报告，不选最好seed。corrected三拟合都在1000更新上限仍改善（每步LL增益.130–.203），明确标记未收敛；不得从剩余误差宣称模型类缺陷。source first seed恢复270后额外做10步到280是停止边界记账问题，280不改善训练最佳260；实际记录保留，不改科学结果。

270分数行、110初始化聚合、110配对path-bootstrap contrasts保留；先对模型seed损失求均值，再按轨迹bootstrap，不把seed当独立物理系统。独立复核4数据、387权重、270预测、120posterior histories、242000physical values、1830分数全部通过。forecast mean/variance最大差2.56e-11/4.11e-9，分数2.84e-14。主结果SHA256=ba8f88504cdb819988f131e783d776d57449b25812a1c4870e98cb7c9b242deb。修订主运行261.828秒（单CPU thread），不含之前native慢worker/JIT成本。

三份独立文献视角重新核查CoDA、Meta-iLaD、OSSM、SimDist、BRICKS-WM、continual compositional world models及MHWA。准确区分其假设和发布状态；没有任何单篇被宣称解决全部受控隐状态迁移问题。但宽泛“预训练状态+小context/input适配器”提案缺乏独特机制和存活残差，撤回为新方法候选；当前不强推替代方向。缺代码也不是Research Gap。

cycle27七份报告、全条件表、bootstrap和两组PNG/SVG/PDF图位于cycle27。全部原始数组/权重留服务器，紧凑产物双端保存。没有新训练等待、没有外部连接阻塞；没有公开发布或删除旧成果。下一次训练前必须从真正可运行的近期可复用框架中冻结一个已有方法修复后仍存在的能力失败，并用独立机制验证；不继续微调这个已由标准方法解决的toy。

cycle27=PROGRESS；总体目标ACTIVE、UNACHIEVED；最终接受方向0。以下各前轮记录是历史证据，旧“当前优先线索”已由此节更新。

## 前轮证据与判断（cycle26）

三条独立文献检索视角核查了现代框架、最强既有方法与相邻机制。Falasca2025及Falasca–Zanna2602.13847v5已覆盖宽泛的响应验证、隐变量记忆和结构化模型加MSM闭包；不能省略新版Sec5.4再宣称创新。36个官方源码文件固定398082d6dcee2e9d42832dfcd39febfa48c383dd，Git blob/SHA256均核查；源码审计不等于原论文复现。

新增全状态oracle反证实验：采用Falasca2025慢快triad的原生离散EM过程，I=1与显式修改的I=0线性控制，3个全新MC cohort各8192条独立初态/噪声实现、4个观察间隔、4个有符号外力幅度。所有方法无外力轨迹完全相同；比较持续fine外力、coarse endpoint、对称插入、普通4子周期及线性解析离散输入算子。不是8192种不同物理系统，也不是学习模型之间的比较。

h=.64时，x1响应路径归一化RMSE线性endpoint76.4589%、对称12.4296%、split4为15.8646%、解析算子约4.02e-15；非线性分别75.7138%、12.7942%、15.7219%。该失败在完美自主动力学下仍可产生，因此不能归因于学错latent或作为新架构必要性的证据。原作者h=.01的一步EM输入定义正确；本轮是刻意coarsen的扩展。非线性普通修正并未完全消除误差；本轮也不证明部分观测学习问题已解决。

864行原始分数、288组3MC聚合全部保留。28原始文件哈希、2700条选定路径独立NumPy重放、81,100,800个线性解析响应值及7776分数复算通过；最大差8.88e-16/1.93e-15/2.25e-13。主结果SHA256=bfa53eccf690cc51fcce9dead49e39f33f9dad65c1a25439ed0394f03fed3d0e。主运行67.204秒，CUDA peak allocated206.94MiB；1,366,286,032bytes原始数组/表留服务器，紧凑产物双端保存。七报告及PNG/SVG/PDF已生成，图例问题只修呈现不改数据。离散参考实验未声称连续SDE步长收敛。

另存KoopmanGeneratorEM官方5源码文件（1,658,154bytes），固定ef830a6d4a6cff699492d54230fc2ad1729cab20。论文例子为slow manifold/Duffing/pinball，公开tree只有前两类notebook；DC motor只是潜在跨基准迁移，不能称其原生实验。主环境缺NumBa已通过新task-local vendor依赖解决，官方PyPI wheel SHA256核查；不改既有环境。原生module及源码导入核查通过，尚未拟合或宣称原论文复现。SFTP MAC/TLS/初始launcher问题的恢复保留EXECUTION_NOTES，无科学重启。

目前唯一较优先的待检验线索是：预训练时间序列状态如何在少量校准下适应会改变隐状态可观测性的输入机制。已有bilinear-generator EM、GRU/NARX、受控SSM、CoDA等直接竞争者；需先执行原生正控制再冻结公平迁移实验。该线索未获得新实验支持，不能称已确定Idea。cycle26=PROGRESS；最终接受方向仍0；总体目标ACTIVE、UNACHIEVED。

## 前轮证据与判断（cycle25）

本轮将原作者非线性pendulum、四个独立quadrant action gains、16D系统latent作为新机制底座。六个原生validation-best final-budget检查点、64新系统、完整历史前缀0/8/32、两种probe、三次评估MC；无新训练。固定运行no-update、native、full-covariance SLR和1024/4096 defensive IS的粒子/diag/full Gaussian，共918行预测、306行MC聚合。所有query共用同一系统latent。主比较328.28秒、峰值GPU已分配475.58MiB。

源接口证明零控制对所有gain无信息，四query分别只依赖各自gain；没有noise、clipping或跨quadrant混入，reset保留系统identity。原native prior sampler因同seed RNG实际只产生abs gain约1.25–2，4096样本低半区率0；保留原采样行为，未擅自改分布。该协议问题不是新Research Gap。

**不能将cycle24推广为普遍质量退化或新架构必要性。** 冷启动无信息probe后，三个unconstrained checkpoint的CRPS点估计均改善（两个bootstrap区间全小于0）；positive三个点估计退化，但仅seed1区间全大于0。positive与unconstrained还同时改变stddev rescaling，不能做单因素归因。敏感probe后所有六个模型在被探测机制上明显改善；仅positive seed3的未探测机制CRPS退化区间全大于0。物理上不该改变的预测律仍有变化，说明忠实性与平均分数不能混同，但这尚不足以证明新方法必要性。

SLR和defensive IS可避免大量无信息更新变化。IS4096冷启动null的最小ESS超过2500，信息probe中位数456–1463；少数长历史case有低ESS（最低1.59），不能夸大为全部高维推断失效。全部预算和方法都保留，没有选最好结果。

独立物理核4480比较max1.78e-15；918行分数15606复算max7.60e-14；54encoder语义重放、144native预测重放、72proposal/importance核查、36独立NumPy SLR求解通过。同batch重要性权重max6.11e-16、ESS2.19e-11、矩2.36e-14、SLR3.56e-15；不同batch/编码路径的float32网络差异不超过1.79e-6。初版复核8192-vs16384 batch误差已保留，修复仅对齐代数重放batch，未动模型/数据/结果。

raw results SHA256=4bd5c4743e6bdb1670b09046b231dc3a5686439e395c72d253e80ce8013c1d1f。七份报告、306行聚合CSV、配对差异/ESS和PNG/SVG/PDF图在cycle25。51个FlowNP官方源码/配置/文档已固定f3403b6479dad33a7b93ba6edcf93c53d40f8e65并核查Git blob/SHA256；未训练或宣称复现。新读OSSM2609.33566（9月27日）明确已有传播/观测校正分离，不能泛创新。远端公开网络下载两次connection-refused已由本地取回同公开源码后hash同步解决，无外部阻塞。

本轮对一般有害更新及新架构必要性提供实质反证。更新一致性工作线索降优先级；当前最终接受方向仍0，总体目标ACTIVE、UNACHIEVED。不得靠重复相同checkpoint调参或扩表维持原路线。

## 前轮证据与判断（cycle24）

六个原生Explore-the-Context发布checkpoint（同一plant的两种actuation变体、各三个原训练seed），128个新生成系统实现；连续历史前缀0/2/5/9，48固定条件，无新训练。冷启动无信息probe后，真实query joint NLL六个checkpoint全部恶化0.559–2.437nats，各64context bootstrap区间都高于0。边际95%覆盖率从不更新的94.01%/89.84%降到77.34%/77.86%。这不是仅latent坐标变化，也不是cycle23归一化问题。

旧的defensive importance校正+Gaussian矩投影已经大幅修复：256粒子将平均损害从1.302418降到0.031634nats，约减轻97.57%；informative probe时仍能利用信息。它是事后同cohort的已知方法对照，不能重命名为新架构；prior整体校准与decoder/likelihood误差仍存在，完整Bayes也不在所有真实条件上优于native。

原生物理moment更新、真实joint mixture scores、EKF、MAP-GGN及256/1024/4096粒子各3MC实现都已执行。原GH积分、EKF尾部及58行全量尾界风险已在固定模型/数据/方法下数值修复，旧文件保留。主复核120数组/2880聚合、普通对照240数组/4224聚合、58patch逐项复算通过；24+24原生/Jacobian重放通过。统一verified_results/results.json SHA256=9fba39ecc18b50e9c1b90a78852e825b8863b94c37cfa8b53e7e96858a94034e。

当前优先工作问题是“时间序列world model的证据更新可靠性”，但只有旧模型toy证据，尚无跨复杂系统/现代模型的剩余缺口，更没有新机制必要性证明。最终接受方向仍0；总体目标ACTIVE、UNACHIEVED。

## 前序直接证据

- cycle20完成72个世界模型；紧凑正逆预测可恢复动作而丢失当前已观测、改变下一步物理状态的随机驱动。加大表示维度修复，故不是普遍不可建模性。
- cycle21完成120个学生模型和12个teacher，四类系统、三个初始化；全部模型/探针选择先于统一测试。新数据/旋转/teacher权重未复用cycle20。
- 主候选projected absolute future-bank未稳定解决问题。预列的response-only竞争目标在前三类系统的9个初始化均保留gain，显著改善物理响应；队列族中普通观测预测更强。
- 成熟敏感性降维使用动作输出目标，在独立cohort前三族也成功，且响应误差更小。新神经压缩器的必要性没有成立。
- 主实验12数据、120权重/预测、12teacher哈希检查；960个指标独立复算最大差1.49e-8；完整batch权重/响应重放最大差5.97e-7。24个成熟对照另经独立monomial重放。

## 研究判断

cycle22新增真实风洞验证：26记录/251016行，18次神经训练，原生WorldTS数值核心、同架构端到端控制、条件GRU、MLP、ridge与持久性。15584个窗口均从原CSV独立精确重建；未来只提供预排定控制计划。两份采集配置变化记录单列。

随机游走测试MSE：WorldTS32=0.244653，ridge=0.244622，GRU=0.233761，持久性=0.446406。相同配置波形：0.113769/0.105645/0.098367/0.193110。现有角色感知模型已提供有效解决方案，没有显示新架构的必要性。总体误差不能代表每个变量：波形RPM中普通GRU/ARX更好，但已有方法能够修复，不能直接认定新Research Gap。

对12个固定原生架构检查点的后验路径控制表明，替换未来latent会明显损害预测：WorldTS32波形MSE0.113769→0.195000。拒绝“模型完全只依赖RevIN均值、预测状态无用”的解释。该控制不是独立确认、MI证明或真实世界因果干预。

主结果60预测文件/27阶段权重通过3416项聚合与检查点复核。原始float32通道聚合有最多9.64e-5差异，已保留原文件并生成float64派生表；原dtype重放误差0。路径控制111文件另经独立训练均值和固定读出重放。

零条已接受最终新颖方向。此前自动目标/角色发现降为低优先级窄线索；不能把未知角色toy直接推广到角色明确的现实任务，也不能把控制物理输出改善等同于完整观测预测或通用模拟能力。

WorldTS源码已确认分开编码命名目标序列与条件；Pythia已有共享预测表示/独立读出及数值参考引导；Phys-JEPA已有物理/残差状态分离。CDL/Denoised MDPs、主动/导数子空间和非线性平衡降阶构成直接边界。一般角色、目标、阶段分离不再作为创新主张。

## 底座与下一项决定性工作

cycle23新增128个全新WH系统配对审计：只替换未来输入，原生内部归一化在128/128系统改变过去输出；cut400的历史输出RMS变化/原std中位数9.8838%，完整native13.1380%；cut744为5.2221%/6.7865%。固定参考和原始因果管线严格为0。16次原生数组重放误差0；128文件、6192指标重算、512严格前缀检查、独立NumPy递推全部通过。此结果限定生成器不适合直接验证固定物理系统的在线/分支能力；不等同于新架构缺口或推翻原观察预测指标。

进一步核查MPNP、2026条件一致性、ICML2026 Drift-to-Coherence及AISTATS2021 Explore-the-Context。后者已有跨分支共享系统latent和主动校准，不能制造弱baseline缺陷冒充创新。修正分支源码34文件固定/校验；作者修正版2.21GB发布archive已直接在服务器取得，发布SHA1一致，24个final-budget validation-best权重共57.84MB已提取。24个模型受限加载、ragged/tensor聚合一致性、因果前缀检查全部通过；其中6个低维模型已完成cycle24机制与实际质量审计。2026-07的variational manifold SysID已有摊销初始化+MAP/Laplace，普通推断桥接不是创新。

停止在旧测试集继续loss调参。上述16维latent非线性pendulum的源接口、新cohort和完整协方差/粒子对照已在cycle25完成，结果见最新证据，不重复启动。不能把所有任务的u=0都当no-information，未知terrain的mountaincar不满足这种假设。若旧方法充分修复，停止新架构论点；不要为了累积失败而扩规模。当前没有等待用户输入的外部阻塞。

cycle24前轮完整协议、文献边界、144行全方法/条件指标、负面证据及PNG/SVG/PDF图仍保留；最新cycle25见上方。原始大数组/权重位于服务器；本地是紧凑结果镜像。总体目标仍ACTIVE、UNACHIEVED，未公开发布。



