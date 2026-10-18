"""Publish a qualified general dynamics substrate and ordinary-control evidence, without claiming novelty."""
import sys,pathlib,csv,shutil,hashlib,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common80_v1 import *
n=read(OUT/'NATIVE_RESULTS80_V2.json');c=read(OUT/'ORDINARY_RESULTS80_V2.json');v=read(OUT/'VERIFICATION80_V1.json');replay=read(OUT/'CONTROL_REPLAY_VERIFICATION80_V2.json');opening=read(OUT/'OPENING80_V2.json');live=read(OUT/'LIVE_C71_SNAPSHOT80_V1.json')
assert v['native_result_sha256']==sha(OUT/'NATIVE_RESULTS80_V2.json') and replay['logged_replay_result_sha256']==sha(OUT/'ORDINARY_RESULTS80_V2.json') and replay['original_independent_verification_sha256']==sha(OUT/'VERIFICATION80_V1.json')
assert read(OUT/'run80_native_v2_exit.json')['exit_code']==read(OUT/'ordinary80_v2_exit.json')['exit_code']==0 and len(n['rows'])==6 and len(c['rows'])==12
def write(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text)
def find(rows,sid,k,method=None):return next(r for r in rows if r['system']==sid and r['initial_condition_index']==k and (method is None or r['method']==method))
table='| System | Initial condition role | FIM-ODE | Context-only weak integral | Privileged true equations |\n|---|---|---:|---:|---:|\n'
for sid in [26,42,28]:
 for k in [0,1]:table+=f"| {sid} | {'Context reconstruction' if k==0 else 'New IC, same system'} | {find(n['rows'],sid,k)['nrmse']:.9f} | {find(c['rows'],sid,k,'ordinary_weak_integral')['nrmse']:.9f} | {find(c['rows'],sid,k,'privileged_true_equations')['nrmse']:.9f} |\n"
report='''# C80：公开 FIM-ODE 的原生动力学推断与普通辨识控制

完成实际可查询动力学模型的原生执行，而非再将预测器分数解释成物理状态。70个上游源文件与固定 commit/Git blob/SHA256 一致，基础权重51,907,384字节与HF LFS哈希一致，12,968,068参数及所有加载张量逐值核验。完整包正常导入，没有跳过注册、改写 forward、替换模型结构或改动权重。三个预定教程系统、每个两个初值的全部六次原生积分均成功；这不是全文或全 ODEBench 复现。

## 信息与执行合同

上游 OpenFIM commit `cee2bb53be1e8e92b73f98dfdbe3605efc9119cc`；HF模型revision `00875c253d025dd33fd70ae751089b1be3a7c174`；ODEBench revision `91909865498f5d4bd69ea575527fee50ad329ef4`。只下载base配置、safetensors和3.7MB方程/轨迹JSON，全部直接到服务器G；未下载所有适配权重、优化器或预训练数据。公共源代码仅经内存传输并逐文件校验，没有本地结果、数据或权重文件同步。

原生教程固定系统26（LV competition）、42（CDIMA）、28（pendulum），每个使用seed0、第一条轨迹、乘性高斯噪声SD0.03、保留一半采样点。实际 helper 使用 linspace 规则保留256/512点，不应写成随机缺失或伯努利 mask。全部2D状态可见，未来无行动；模型内部按原方法补到3D。每个系统仅第一条含噪路径作为上下文，第二条路径只提供干净初值和评估时间；不提供其未来输出、真向量场或真系数给模型。

IC0的整条路径参与上下文，是重建，不是因果时间预测或独立测试；IC1是同一方程和系数下的新初值。两者分开报告，不混合成泛化指标。三个固定公开演示、一个噪声seed不足以推断整个领域；结果揭示的现象尚未经过独立系统确认。

CPU float32、两线程、仅当前进程efficient-core affinity，Python3.10.20、Torch2.8.0+cu128、NumPy1.26.4、SciPy1.15.3、Transformers4.56.2。完整隔离依赖、版本与导入记录保留。原生 `model_forward` 与教程缓存编码/解码接口的18点查询完全一致、重复查询一致；独立脚本用缓存向量场重新积分全部六条路径，预测逐值差0。这里的CPU执行没有复用历史GPU预测，故不放宽任何旧容差。

## 全部结果

NRMSE = RMSE / std(完整参考轨迹的全部坐标值)，保持原来定义，不是逐坐标标准化误差。该std仅用于评估，未进入模型或控制的选择。各方都获得同一个干净初值与时间网格。

'''+table+'''
![Native trajectories](figures/native_phase_portraits80_v2.png)

图已在内存中读取服务器原始PNG并做视觉检查，没有保存到本地。它显示摆系统的单条外轨道重建良好，而内侧新轨道错误；不能仅凭轨道图归因为某种独有的潜在机制。

## 普通控制与选择边界

六个原生分数揭示后，单独冻结 `ORDINARY_CONTROL80_V1.md`。这是事后反证，不是独立确认。仅用相同256个含噪上下文点，做标准弱积分向量场回归：三次多项式，或多项式加两坐标的sin/cos；8/16/32间隔梯形积分，ridge三候选、稀疏阈值两候选。每系统12候选在前192点拟合、最后65个可见上下文点验证，选择后用全部256点重拟合。三系统共36候选和3次全上下文拟合，没有神经训练。三者都由上下文选择了trig库。

trig库在阅读公开方程后确定，属于source-informed普通控制，不声称完全无先验或与基础模型预训练成本相同。没有把真实系数、IC1未来轨迹或干净IC0未来标签用于拟合/选择；未继续按外部结果加rational项、调ridge或加候选。所有候选与失败状态保留。

CDIMA新初值被普通回归大幅修复，摆的误差部分下降但仍高，LV新初值反而变差。不能报告普通方法普遍更好，也不能把剩余摆误差直接认定为新机制。候选库失配、单路径信息与归纳先验仍是竞争解释；GP向量场后验、MOD式保持上下文拟合的不同动力学等更强控制本轮没有执行。

真实方程/常数另作特权数值诊断；它们的最大NRMSE约0.000241，支持排除主要的参考身份和积分定义错误。它不是学习基线，也不能证明含噪单路径已足够识别全局动力学。全部18条最终轨迹完整有限，所有数值积分成功。未触发20,000次rhs或绝对状态1e6的预置保护；保护不等于稳定性证明。

## 独立验证与原始产物

重新从公开JSON和seed重建上下文、逐一核对干净参考身份，全部一致。独立计算18条结果36,864个预测/参考标量及36个上下文选择分数，最大差0；三个最终线性模型用不同的augmented least squares重构，最大系数差7.18e-10；六条原生轨迹缓存/直接积分重放差0。源和权重结束时再次核验。

初次普通控制保留了全部结构化候选、系数与预测，但stdout仅留在工具记录。为补齐服务器终端日志，进行了完全相同的单独V2回放，未改协议或选择；全部数组、模型系数和指标逐值相同。它不是额外独立实验或稳定性seed。原始V1产物保留，独立验证和精确回放链均有哈希。

新数据、源、依赖前缀、神经权重和完整数组保留G；GitHub上传作者代码、协议、完整紧凑结果、候选、系数、版本、日志、图和哈希。上传内容不包含大规模信号或神经权重。

## 研究判断

FIM-ODE的原生可查询动力学能力已经实际建立；摆新初值现象也被实际复现，但这没有证明一个未解决、独有且可投稿的新机制问题。C80的独立文献补查找到MOD、GP-ODE、LUNO、SANODEP、Active-SINDy和上下文验证等直接前例，详见 literature_review.md。只重新命名“轨迹外有不确定性”“需要收集信息”“用上下文选适配”不具备所需创新性。

保留这个已资格化底座供后续通用问题使用，停止把当前单个摆的剩余误差作为新架构动机。没有证明所有更高维/部分观测/机制迁移问题已解决，也没有进行未知传感器或行动干预实验。下一项决定性工作必须在事先固定的独立系统和预算下区分信息不足、先验和可修复学习残差，且与强普通方法同信息比较，不能再调这六条演示。C71原预算三种子任务继续运行；当前实际进程确认第一组结束、第二组在训，保留评估未打开。总目标ACTIVE/UNACHIEVED，最终接受研究方向0。
'''
write('NATIVE_DYNAMICS_FINDING80_V1.md',report)
literature='''# C80 文献补查：上下文支持、动力学后验与新增信息

本补查延续C79的三个研究问题，主流、批评、相邻辨识三路独立检索，再核验关键原文和公开实现。重点2026-04-18至10-18，必要时回溯。目的不是把每篇限制都变成Idea，而是判断C80实测新初值现象是否独有。以下均为作者结果/范围，除FIM外没有在本轮执行这些方法。

## 1. 保持上下文拟合、构造不同的动力学

[Giampiccolo、Iacca、Marchetti，2026-05-27，MOD](https://pmc.ncbi.nlm.nih.gov/articles/PMC13494032/) 已直接研究普通NODE/UDE种子ensemble在轨迹外过度自信。其方法在保持训练轨迹损失的条件下，增大选定区域的向量场分歧，并检查新初值模拟；[Julia代码](https://github.com/cosbi-research/UQNU) 已发布。区域、阈值和区间近似会影响结果，不能称为任意OOD的校准后验。它直接阻止把“多个同样拟合的场在别处不同”作为新发现。

相比这个约束分歧构造，[Hegde等，UAI2022，GP-ODE](https://proceedings.mlr.press/v180/hegde22a.html) 从含噪轨迹学习场后验，并以连贯函数样本积分；[实现](https://github.com/hegdepashupati/gaussian-process-odes) 是更直接的普通贝叶斯基线。[Magnani等，ICML2025，LUNO](https://proceedings.mlr.press/v267/magnani25a.html) 则在线性化的已训神经算子上构造函数值GP，实验中需要250个有标签验证对作校准，不能当作零标签控制。这三者不同于每步独立重抽点态方差，后者会改变固定未知场的生成假设。

## 2. 上下文系统分布与主动获取信息

[Qing等，SANODEP](https://arxiv.org/html/2406.02352v2) 使用多条轨迹推断系统潜在分布，并以任务目标选择初值/测量计划；[JAX实现](https://github.com/TsingQAQ/SANODEP) 及外部权重链接已找到，权重本身未核验。它解决的是任务BO，不能升级为所有新初值的可信度证书。其更早的 [Neural ODE Processes](https://arxiv.org/html/2103.12413) 前身也已有上下文随机动力学和观测选择，不能只换成foundation一词重提。

[Larrañaga、Fasel、Brunton，2026-06-10，Active-SINDy](https://arxiv.org/html/2606.12182v1) 通过稀疏模型ensemble的导数分歧选择ODE初值，获得新轨迹后更新；[真实Python实现](https://github.com/alarran/active-sindy) 的停止规则基于稀疏模式和系数稳定性，而非未来评估系数。它依赖状态可见、可重置初值和合适函数库。PDE部分使用候选的观测导数信息，不能一并称为无候选标签的获取。我们执行的弱积分回归不是该论文算法复现。

若初值不能自由重置，[OPAX，NeurIPS2023](https://arxiv.org/html/2306.12371v2) 已沿可达行动策略规划信息，不能把只在任意点可查询的实验与它公平并列。其不确定性校准和函数类假设需保留，[JAX实现](https://github.com/lasgroup/opax) 在5090上的兼容性未核验。本轮未执行主动数据获取。

## 3. 不用未来标签选择适配

[Huang、Michailidis，2026-09-21，PBML-LTI](https://arxiv.org/html/2609.24117v1) 已用可用上下文内的验证后缀选择适配prefix，再重拟合；[实现](https://github.com/chenfeng-huang/PBML-LTI-TMLR-2026) 可查。其证书依赖query/support Gram关系，不是任意查询的support-only保证；某个正则界仍有未知真动力学项。它证明普通上下文验证已有明确方法，却没有证明任意非线性陌生世界可被这样认证。

FIM-ODE原文已承认路径外欠定，也已披露训练损失选择和测试窗口选择的差别。辅助不确定性头主要作向量场损失加权；本轮没有测它的校准或场后验，不能写成已经发现它“没有不确定性”。只测试与接口契约相符的能力，才可能分辨实际残差。

## 4. 对研究问题的判断

RQ1：公开底座既有点估计场，也有概率场、连贯采样和主动试验，不是统一的“只会续写”。RQ2：C80支持数值执行真实、部分新初值有困难；普通回归呈异质结果，成熟的支持/先验/后验解释仍未排除。RQ3：当前没有一个达到接受标准的新机制方向。已有论文的假设边界不等于其外部问题已解决；同样，单个已知摆失败也不足以宣称一般性研究缺口。

接下来的证据若仍研究支持，应在固定数据预算下区分重复采样和几何/激励多样性，排除先验家族、积分、噪声和上下文验证因素，并对比GP-ODE、MOD/ensemble、普通辨识或主动采集。只有独立系统残差仍存在且最小方法优于匹配强基线，才接纳方向。不能因为现有方法有条件，就假定我们的版本更有创新。
'''
write('literature_review.md',literature)
entries={
 'baseline_registry.md':'FIM-ODE原生base已实际加载并执行：固定commit70源文件、13M参数、完整state精确校验、三教程系统/两个初值。弱积分回归36候选+3全上下文拟合；trig库source-informed。真方程仅特权数值参考。GP-ODE/MOD/SANODEP/LUNO/Active-SINDy/PBML-LTI为已核验先例，本轮未执行，不列作完成复现。C71第一组已结束、第二组原预算在训，保留评估未打开。',
 'experimental_results.md':'完整六条native和十二条控制结果见NATIVE_DYNAMICS_FINDING80_V1.md。新初值FIM/普通/特权NRMSE：LV .128249/.300991/.00004454；CDIMA .158431/.029383/.00021228；pendulum1.621214/.908761/.00024098。全部积分成功。36,864预测/参考标量及36选择分数独立重建；六条native缓存积分重放差0。只有3个系统1噪声seed，非泛化定论。',
 'negative_evidence.md':'单路径重建好不保证新初值模拟，但此现象及轨迹外场歧义已有FIM原文、MOD和GP-ODE等直接研究。普通回归修复CDIMA、部分修复pendulum却损害LV，不能统一宣称某模型优劣。没有测试不确定性头、主动采集或跨传感器机制。导入依赖错误和一次SSH传输MAC错误是工程错误，源/资产最终逐哈希通过，不计模型失败。',
 'hypothesis_space.md':'H1原生实现或权重装载错误：70源/全state与两种接口/独立积分相同，当前不支持。H2参考身份/积分错误：真方程低误差，主要解释不支持。H3普通拟合足够：在CDIMA支持，其余异质，强GP/后验控制未执行。H4新机制必需：不支持，信息覆盖、先验、函数库和已知不确定性方法仍是强解释。不能用单轨道反例接受方向。',
 'research_ideas.md':'最终推荐数0。当前实测建立了通用动力学推断底座，未建立新的机制。停止把三条教程或摆的剩余误差继续调成一个方法。若进一步研究上下文支持，先冻结独立系统、数据/适配预算和几何信息对照，纳入连贯场后验及强辨识/主动方法；不能仅以FIM特定头或paper未做某实验声称创新。C71原预算评估继续。',
 'final_research_direction.md':'Research Question：通用世界模型的新初值残差是否需要新机制？Evidence：原生FIM六案例，普通/真方程十二控制；数值路径已校验，单个摆外推困难，但无独立系统结论。Existing Methods：GP场后验、MOD上下文兼容分歧、SANODEP、LUNO、主动SINDy和上下文验证已有。Proposed Idea：无已验证新机制。Falsification：信息覆盖、匹配先验、普通辨识/后验或上下文选择解释残差则放弃其机制主张。Feasibility：已资格化13M模型/源码/数据，单5090或CPU可运行。Next Experiment：在独立系统与固定预算下区分覆盖与学习残差，或依据C71完成结果切换问题；不继续调教程。Overall ACTIVE/UNACHIEVED，accepted0。'}
for name,body in entries.items():write(name,'# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n\nSee NATIVE_DYNAMICS_FINDING80_V1.md and literature_review.md for complete scope.\n')
registrations=[
 dict(key='MOD',authors='Giampiccolo, Iacca, Marchetti',title='A novel approach to quantify out-of-distribution uncertainty in Neural and Universal Differential Equations',date='2026-05-27',primary='https://pmc.ncbi.nlm.nih.gov/articles/PMC13494032/',code='https://github.com/cosbi-research/UQNU',verified='Full primary paper and actual public Julia source, not executed',scope='Context-compatible off-support disagreement; chosen region/thresholds, no universal calibrated posterior'),
 dict(key='GP-ODE',authors='Hegde, Yıldız, Lähdesmäki, Kaski, Heinonen',title='Variational multiple shooting for Bayesian ODEs with Gaussian processes',date='UAI2022',primary='https://proceedings.mlr.press/v180/hegde22a.html',code='https://github.com/hegdepashupati/gaussian-process-odes',verified='Primary proceedings/full PDF and implementation, not executed',scope='Stationary GP posterior, sparse VI and coherent function samples'),
 dict(key='LUNO',authors='Magnani, Pförtner, Weber, Hennig',title='Linearization Turns Neural Operators into Function-Valued Gaussian Processes',date='ICML2025',primary='https://proceedings.mlr.press/v267/magnani25a.html',code='https://github.com/MethodsOfMachineLearning/luno',verified='Primary proceedings/full text and package, not executed',scope='Posthoc operator GP;250 labeled calibration pairs in experiments'),
 dict(key='SANODEP',authors='Qing, Langdon, Lee, Shafei, van der Wilk, Tsay, Misener',title='System-Aware Neural ODE Processes for Few-Shot Bayesian Optimization',date='arXiv2406.02352v2,2024-11-02; TMLR2025',primary='https://arxiv.org/html/2406.02352v2',code='https://github.com/TsingQAQ/SANODEP',verified='Full primary text and JAX repo; external checkpoint links not downloaded',scope='System distributions from trajectories; task BO acquisition, no generic field certificate'),
 dict(key='Active-SINDy',authors='Larrañaga, Fasel, Brunton',title='How Low Can You Go? Active Learning for Sparse Model Discovery in the Ultra-Low-Data Limit',date='arXiv2606.12182v1,2026-06-10',primary='https://arxiv.org/html/2606.12182v1',code='https://github.com/alarran/active-sindy',verified='Full paper and actual Python acquisition/stopping code, not executed',scope='Resettable observed ODE states/library; PDE candidate residual uses derivative labels'),
 dict(key='PBML-LTI',authors='Huang, Michailidis',title='PAC-Bayesian Meta-Learning for Few-Shot Identification of Linear Dynamical Systems',date='arXiv2609.24117v1,2026-09-21',primary='https://arxiv.org/html/2609.24117v1',code='https://github.com/chenfeng-huang/PBML-LTI-TMLR-2026',verified='Full paper, context-selection protocol and implementation, not executed',scope='Observed autonomous LTI; inner validation prefix selection; conditional query/support certificate'),
 dict(key='OPAX',authors='Sukhija et al.',title='Optimistic Active Exploration of Dynamical Systems',date='NeurIPS2023; arXiv2306.12371v2,2023-10-30',primary='https://arxiv.org/html/2306.12371v2',code='https://github.com/lasgroup/opax',verified='Full primary text and JAX source by adjacent scout;5090 compatibility not verified',scope='Reachable action-constrained information exploration; calibration and regularity assumptions')]
dump(OUT/'CITATION_AND_SCOPE80_V1.json',dict(utc=utc(),entries=registrations,independent_perspectives=3,literature_claims_are_not_native_replications=True,final_new_mechanism_recommendations=0))
allrows=[]
for row in n['rows']:allrows.append(dict(system=row['system'],initial_condition_index=row['initial_condition_index'],role=row['role'],method='native_FIM_ODE',privileged=False,mse=row['mse'],rmse=row['rmse'],mae=row['mae'],nrmse=row['nrmse'],finite_complete=row['finite_complete'],solver_success=all(z['success'] for z in row['solver_calls'])))
for row in c['rows']:allrows.append({k:row[k] for k in ['system','initial_condition_index','role','method','privileged','mse','rmse','mae','nrmse','finite_complete']}|dict(solver_success=row['solver']['success']))
with (OUT/'all_metrics80_v1.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
dump(OUT/'CLOSURE80_V1.json',dict(utc=utc(),status='NATIVE_FIM_DYNAMICS_QUALIFIED_MIXED_ORDINARY_CONTROLS_NO_NEW_GAP_ACCEPTED',parameters=12968068,source_files=70,native_rows=6,ordinary_rows=6,privileged_reference_rows=6,inner_candidates=36,context_refits=3,exact_logged_replay_is_not_new_independent_experiment=True,new_neural_fits=0,native_result_sha256=sha(OUT/'NATIVE_RESULTS80_V2.json'),ordinary_result_sha256=sha(OUT/'ORDINARY_RESULTS80_V2.json'),verification_sha256=sha(OUT/'VERIFICATION80_V1.json'),logged_replay_verification_sha256=sha(OUT/'CONTROL_REPLAY_VERIFICATION80_V2.json'),prediction_target_values_checked=36864,metric_error=0.,native_cached_integration_replay_error=0.,all_final_predictions_complete_finite=True,all_solvers_successful=True,independent_general_mechanism_confirmation=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_weight_files_written=0))
status='''# C80 当前状态：原生可查询动力学模型已经实际资格化

OpenFIM固定commit70源文件，HF固定revision13M参数和所有state张量逐值核对。原生教程3系统×2初值全部积分成功；独立直接/缓存查询和6积分回放差0。上下文IC0是重建，IC1是同系数的新初值。36上下文-only弱积分候选+3重拟合，另有6特权真方程数值参考。新初值FIM/普通NRMSE：LV.128249/.300991；CDIMA.158431/.029383；pendulum1.621214/.908761。普通方法不是统一胜者，特权参考不是学习基线。全部18条结果和39个上下文拟合独立复核，日志回放不算新seed。

本轮没有接纳新机制。MOD/GP-ODE/SANODEP/LUNO/Active-SINDy/PBML-LTI等先例限制了泛泛的场不确定性/主动上下文/适配选择说法，不能把单条摆轨道剩余误差当新研究问题。后续证据须来自独立系统、固定预算、匹配强辨识/概率控制，不继续调教程。

C71已按当前进程句柄核验仍在运行，第一种子结束、第二种子继续原预算；保留评估未提前开启。通用时间序列世界模型目标ACTIVE/UNACHIEVED，最终接受方向0。代码、报告、紧凑结果与图上传Gifoe/wm4health，数据/源/依赖/神经权重/完整数组保留服务器G，未同步本地载荷。
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V80.md').open('x',encoding='utf-8') as f:f.write(status)
files=['PROTOCOL80_V1.md','ORDINARY_CONTROL80_V1.md','NATIVE_DYNAMICS_FINDING80_V1.md','literature_review.md',*entries]
files+=['scripts/'+p.name for p in sorted((ROOT/'scripts').glob('*.py'))]
files+=['outputs/'+name for name in ['SOURCE_MANIFEST80_V1.json','ASSET_MANIFEST80_V1.json','DEPENDENCY_INSTALL80_V1.json','DEPENDENCY_INSTALL80_V2.json','DEPENDENCY_SUPPLEMENT80_V1.json','dependency_install80_v1.log','dependency_install80_v2.log','dependency_supplement80_v1.log','IMPORT_AUDIT80_V1.json','IMPORT_AUDIT80_V2.json','IMPORT_AUDIT80_V3.json','OPENING80_V2.json','CONTROL_OPENING80_V1.json','CONTROL_OPENING80_V2.json','NATIVE_RESULTS80_V2.json','ORDINARY_RESULTS80_V1.json','ORDINARY_RESULTS80_V2.json','VERIFICATION80_V1.json','CONTROL_REPLAY_VERIFICATION80_V2.json','CITATION_AND_SCOPE80_V1.json','LIVE_C71_SNAPSHOT80_V1.json','CLOSURE80_V1.json','all_metrics80_v1.csv','run80_native_v2_launch.json','run80_native_v2_exit.json','run80_native_v2_stdout.log','ordinary80_v2_launch.json','ordinary80_v2_exit.json','ordinary80_v2_stdout.log']]
patterns=json.loads(os.environ['C80_PRIVATE_EXPORT_PATTERNS_JSON']);dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);pub=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 b=raw.decode('utf-8-sig').encode();p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='utf-8'))
figure=pathlib.Path(n['figure_path']);assert sha(figure)==n['figure_sha256'];b=figure.read_bytes();rel='figures/native_phase_portraits80_v2.png';p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='binary',visually_checked_in_ram=True))
dump(dest/'manifest.json',dict(utc=utc(),files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),large_neural_weights_raw_benchmark_arrays_included=False,compact_classical_coefficients_and_all_metrics_published=True,binary_figure_count=1,source_commit='cee2bb53be1e8e92b73f98dfdbe3605efc9119cc',native_result_sha256=sha(OUT/'NATIVE_RESULTS80_V2.json'),ordinary_result_sha256=sha(OUT/'ORDINARY_RESULTS80_V2.json'),accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_weight_files_written=0))
print(json.dumps(dict(status='C80_SNAPSHOT_READY',files=len(pub),bytes=sum(z['bytes'] for z in pub),manifest_sha256=sha(dest/'manifest.json'))),flush=True)
