"""Cycle-local reports and exportable figures from verified completed pilot outputs."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
analysis=read('outputs/analysis/analysis.json');result=read('outputs/innovation_pilot/results.json');verify=read('outputs/innovation_pilot/independent_verification.json')
active=read('outputs/active_subspace/results.json');active_verify=read('outputs/active_subspace/independent_verification.json');assert active_verify['pass']
assert verify['pass'];groups={(g['family'],g['method']):g['metrics'] for g in analysis['groups']}
families=['coupled_linear','rotating','saturating'];methods=['joint','joint_idm','ema_idm','gaussian_idm','mixture_idm','obs_idm','fullobs','joint_idm_full']
def write(name,body):(ROOT/name).write_text(body.strip()+'\n',encoding='utf-8')
def value(f,m,k):return groups[(f,m)][k]['mean']
table=['| Family |4D joint+inverse: gain accuracy | Action accuracy | Physical prediction MSE |4D gain-modulation NRMSE |12D gain accuracy |12D modulation NRMSE |',
       '|---|---:|---:|---:|---:|---:|---:|']
for f in families:
    table.append(f"|{f}|{value(f,'joint_idm','gain_accuracy'):.4f}|{value(f,'joint_idm','inverse_joint_action_accuracy'):.5f}|{value(f,'joint_idm','native_physical_MSE'):.5f}|{value(f,'joint_idm','native_modulation_relative_RMSE'):.4f}|{value(f,'joint_idm_full','gain_accuracy'):.4f}|{value(f,'joint_idm_full','native_modulation_relative_RMSE'):.4f}|")
table='\n'.join(table)
full=['| Family | Method | Gain accuracy | Native physical MSE | Frozen-encoder successor probe MSE | Native modulation NRMSE |', '|---|---|---:|---:|---:|---:|']
for f in families:
    for m in methods:full.append(f"|{f}|{m}|{value(f,m,'gain_accuracy'):.5f}|{value(f,m,'native_physical_MSE'):.6f}|{value(f,m,'frozen_encoder_successor_probe_MSE'):.6f}|{value(f,m,'native_modulation_relative_RMSE'):.5f}|")
full='\n'.join(full)
active_table=['| Family | Output target | Gain accuracy | Physical MSE | Modulation NRMSE |','|---|---|---:|---:|---:|']
for f in families:
    for choice in ['all_outputs','action_outputs']:
        rows=[r for r in active['rows'] if r['family']==f and r['output_choice']==choice]
        active_table.append(f"|{f}|{choice}|{np.mean([r['gain_accuracy'] for r in rows]):.5f}|{np.mean([r['native_physical_MSE'] for r in rows]):.6f}|{np.mean([r['modulation_relative_RMSE'] for r in rows]):.6f}|")
active_table='\n'.join(active_table)
write('literature_review.md','''# 文献与创新边界

本轮从“不可控但与奖励相关的特征可能被丢弃”缩小到：**当前已观测的随机驱动会改变下一步物理状态，紧凑的自预测表示是否仍会丢掉它？** 本轮已运行受控实验；新颖性尚未成立。

[2606.30068v1](https://arxiv.org/html/2606.30068v1) 的特征决定奖励，提供了直接的近邻问题。把 uncontrollable-but-relevant 重新命名不构成创新。[CDL,ICML2022](https://proceedings.mlr.press/v162/wang22ae.html) 更早已区分可控变量、不可控但影响动作结果的变量和无关变量，因此“保留不可控的因果父变量”也不是新概念。

[Sensorimotor World Models](https://arxiv.org/html/2606.20104v1) 的联合正逆预测、[SLIM](https://arxiv.org/html/2610.03137v1) 的逆动作辅助、[Hamiltonian JEPA](https://arxiv.org/html/2609.33497v1) 的双空间/结构化输入端口均是必要边界。[MoP-JEPA](https://arxiv.org/html/2607.05238v1) 采用EMA、单位球、路由和硬分配，不能把本轮简化的 joint-target two-head arm 当作其原生实现。[Var-JEPA](https://arxiv.org/html/2603.20111v1) 已给出变分生成视角；本轮对角Gaussian控制不是该论文复现。[CausalDreamer](https://arxiv.org/html/2610.12016v1) 已做奖励/可控性分组，不能再把一般因子化包装为创新。

更精确的待核查方向：在非因子化观测中，从动作响应学得输出目标，再用它要求输入状态保留必要因果信息。动作响应输出空间与完整输入状态不必相同。[主动子空间](https://arxiv.org/abs/1304.2070) 和 [导数辅助输入/输出降维](https://www.jmlr.org/beta/papers/v27/25-1861.html) 已是成熟机制。现已实际运行敏感性降维：全部输出目标失败，而同一压缩器使用动作响应输出目标成功。动作输出选择是两个成功方法共享的待核查成分；不能把旧压缩器改名为创新，也不能据此声称旧论文已经提供了当前自动目标选择。

完整事实卡和范围区分见 LITERATURE_BOUNDARY.md。八份固定版本原文/PDF、提取文本和SHA256登记在 outputs/literature。原文阅读与源码执行分开：cycle20 是数学目标的受控任务迁移，没有复现上述论文的原生视觉排行榜。
''')
write('baseline_registry.md','''# Baseline registry: completed72-fit pilot

Three Markov physics families, each under an invertible12D mixed observation with nine AR nuisance factors and one current visible fresh gain. Each method is independently trained per family,3initializations. This is neither one cross-system pretrained model nor a real-data benchmark.

| Arm | Interface/objective | Scope |
|---|---|---|
| joint |4D shared encoder, joint future-target MSE, variance/covariance penalty | Mathematical JEPA-style control |
| joint_idm | Same+action inverse MSE | Same-protocol inverse control; adds anti-collapse beyond native SMWM |
| ema_idm | EMA detached target+.99 update+inverse | Existing asymmetry control |
| gaussian_idm | Diagonal latent Gaussian NLL+inverse | Simplified probability control, not native Var-JEPA |
| mixture_idm |2head joint-target hard-min MSE+inverse | Omits source MoP's normalization/EMA/router; not native MoP |
| obs_idm |4D input bottleneck predicts full next observation+inverse | Ordinary observation-prediction control |
| fullobs | Raw12D observation MLP, no learned bottleneck | Larger state access, ordinary positive control |
| joint_idm_full |12D learned joint+inverse | Capacity control |

Each world fit uses1500 Adam updates,batch256,lr1e-3,64-wide MLPs. Physical labels are used only after encoder freezing for diagnostic current/successor probes; they never update world encoders. Probe candidates: linear ridge and64-wide MLP,500 updates, validation-selected. ExtraTrees controls are post-pilot exploratory checks on27 frozen encoders, not new independent confirmations.

Post-pilot classical control:18 fits use the same9 full-observation teachers. Top4 input gradient-covariance directions predict either all12 teacher outputs or the learned2D action-response output span; train-only whitening and polynomial ridge of degree1/2/3 select by validation latent error. Physical labels are confined to a final linear diagnostic readout. This is an established compressor with two output-target choices, not a new active-subspace algorithm. Models and raw predictions are in outputs/active_subspace; independent monomial replay and arithmetic passed.

All checkpoints,9split files,72raw prediction files and job stdout/stderr/source snapshots are on the server under outputs/innovation_pilot. Tree checkpoints and predictions are under outputs/strong_probes. Compact outputs and reports are mirrored locally. Do not claim all raw arrays/weights were downloaded locally.

Runtime: existing project Python3.10.20,torch2.8.0+cu128,numpy1.26.4,RTX5090. No environment installation or previous-cycle modification. Exact data seeds,train-only normalization,model seeds and hashes are in the frozen plan/config/registry. Current cycle21 prototype has a new independent cohort and separate protocol; no result is yet claimed here.
''')
write('experimental_results.md',f'''# 已完成的紧凑世界状态机制实验

在三个可重复的合成动力系统中，4D正逆预测表示可以几乎完美恢复动作，却没有保留当前可见、决定物理动作幅度的随机增益。增大表示维度基本修复，因此当前证据指向容量约束下的状态选择偏差，不能表述为所有世界模型均失败。

## 数据和控制

gain g_t在0.4/1.6之间独立采样，当前g随观测提供，下一步g未知。两维随机二值动作覆盖全部四种组合；训练/验证/测试gain和动作支持相同。q,p的更新由当前g调制。九个AR(.98)干扰因子与动作独立。生成因子的训练集标准化后作固定正交混合，模型不能读取物理变量身份。物理标签只用于评估和冻结表示探针。

每族192/64/128独立轨迹，各64个记录步，另64步burn-in；数据随机流与模型初始化分开。全部72个世界模型和探针选择完成后才计算测试指标。三seed结果是初始化变异，不能当作三个真实系统重复实验。

## 主结果（3个初始化的均值）

{table}

Modulation NRMSE比较同一(q,p,nuisance,action)下当前gain取两种支持内值时的下一状态差。1约等于完全忽略gain造成的差，0为精确响应。它有已知生成真值，不是从观测相关性推断出来的因果标签。

4D joint+inverse对当前物理(q,p)的解码MSE均值分别为{value('coupled_linear','joint_idm','current_physical_decoding_MSE'):.6f}/{value('rotating','joint_idm','current_physical_decoding_MSE'):.6f}/{value('saturating','joint_idm','current_physical_decoding_MSE'):.6f}，表示有效秩约4，排除了整个表示完全坍塌这一简单解释。冻结表示再拟合更强successor probe仍接近gain-blind误差水平，问题不只在原先的转移头。

## 强探针与修复证据

27个冻结编码器的ExtraTrees分类/回归检查没有恢复4D joint+inverse的gain：九次分类准确率范围49.19%–51.16%。树探针的successor误差也没有通过验证集选择替代原探针。该结论仅适用于实际测过的读出族，不是严格零互信息证明。

12D joint+inverse恢复gain接近100%，原始观测MLP也达到100%。原始观测模型三个系统物理MSE均值0.00135/0.00112/0.00250，均通过预设验证readiness。双头控制在coupled_linear的一个seed恢复gain，另外两个seed失败；不隐藏这个反例，也不声称所有随机模型均失败。源码兼容的MoP/EMA/路由控制尚待cycle21独立测试。

## 完整目标对照

{full}

## 成熟敏感性降维对照（探索性，同一pilot数据）

{active_table}

18次旧方法控制只改变输出目标选择，压缩器均为输入梯度协方差的前4方向，再用多项式ridge拟合转移。全部输出目标的gain约50%、响应NRMSE约1；动作响应目标九次gain均100%，物理MSE0.00035–0.00174、响应NRMSE0.00140–0.01429。没有用物理变量标签选择输出、输入投影或转移超参数；物理标签只在冻结后拟合诊断读出。

这证明无需新神经压缩器便能修复当前toy。成功控制本身共享“从teacher动作响应选择输出”成分，所以它尚不能否定目标选择问题，也不能证明该成分新颖。18个模型/teacher哈希、72个指标、验证选择和手写monomial完整重放通过；这是同cohort的探索性反证，不是新独立数据确认。

## 可复核性

主训练和探针共797.43s；72个权重、72个预测、9个数据文件哈希验证。504个测试指标独立重算最大差1.84e-7，8个完整batch checkpoint重放误差0，物理更新独立复算最大差2.35e-7。所有训练权重/探针选择文件先于测试预测生成。更强探针27个权重/预测另作哈希检查、54项算术复算和3456样本的逐树独立遍历。

初次preflight在写receipt时因outputs目录不存在退出1；创建目录后完整重跑，8种模型的同seed三步更新重放均为0。失败说明保留，主科学job一次完成exit0。源码中的梯度tensor转float日志警告没有改变训练步骤；没有中途修改科学脚本或依据测试调参。

![机制结果](outputs/analysis/causal_driver_pilot.png)

这些是合成、完整观测、单步、任务迁移证据；尚未证明真实任务频率、多步概率模拟、跨系统零样本能力或新方法优势。
''')
write('negative_evidence.md','''# 负面证据与限制

- 12D表示和原始观测预测已经修复主要失败。当前不能声称任务不可辨识、所有世界模型不足或必须增加新架构。
- 4D pure joint丢掉的不只有gain，故不能用它单独归因。主要机制证据来自接近完美逆动作恢复、良好q,p解码但gain/响应丢失的joint_idm。
- 普通EMA和对角概率头没有稳定恢复gain，但它们不是所有随机状态模型的代表。本轮两头控制也不是原生MoP-JEPA；其一个seed成功必须保留。
- 树探针是看到初步结果后提出的诊断加强，不能包装为预注册独立确认。多个实际读出失败仍不等价于严格证明表示中没有任何gain信息。
- 保留不可控但影响动作结果的变量，是CDL等已有研究明确提出的原则。一般因果充分状态、逆动力学、随机latent、双空间、主动子空间均不能重新命名为创新。
- 生成因子先标准化再线性混合是受控观测构造；它不是未经审计即可迁移到真实传感器的预处理。没有真实数据、非线性观测几何或长期模拟证据。
- 新teacher目标原型消耗额外训练和查询；必须与相同预算、相同teacher、相同辅助头参数量以及成熟敏感性降维比较，不能只比1500-step弱基线。
- 成熟active-subspace压缩器配合动作响应输出目标，在九个pilot组合全部恢复gain并显著降低响应误差；新神经压缩架构的必要性没有证据。该控制共享自动输出选择成分，因此其结果应把候选缩小到目标选择及其适用范围，不能误写成旧文献已完整解决或新方法已成立。
- cycle21正在独立cohort上执行，当前不引用其选择性验证结果，也不宣称已经有方法收益。队列族同时有已知action-equivalence限制，不是纯incoming-driver机制隔离。
''')
write('hypothesis_space.md','''# 竞争解释和反证

| Explanation | Evidence / falsifier | Current status |
|---|---|---|
| Driver absent/unidentifiable | Invertible observation construction,randomized support,fullobs positive controls | Rejected for these tasks |
| Entire representation collapsed | Effective rank,current q,p decoding,inverse accuracy | Rejected for joint_idm |
| Weak linear probe | Nonlinear MLP and post-pilot ExtraTrees; frozen successor probes | Strongly weakened within tested readout families |
| Only a bad transition head | Refit successor predictor on frozen encoding | Not sufficient to explain primary joint_idm result |
| Intrinsic4D information bottleneck | True causal state has three factors; a suitable4D map exists | Dimension alone is not an information-theoretic impossibility; learned selection still matters |
| More capacity solves it |12D control and full raw observation | Supported; prevents a universal new-architecture claim |
| Any stochastic predictor fixes it | SimplifiedGaussian/mix versus proper normalized EMA/router/SSM | Simplified controls insufficient; source-informed control pending |
| Grounded target selection is the missing ingredient | Same teacher:full bank,contrast bank,projected bank,random subspace,extra-budget reconstruction | cycle21 fixed independent test underway |
| A new neural compressor is necessary | Classical active-subspace+action-output target succeeds in9/9 cases | Rejected on these toys; target-selection novelty remains unresolved |

Future source-native/realistic checks must distinguish a new usable state-selection mechanism from an old sensitivity method, extra teacher capacity, auxiliary-loss weighting or an artificially favorable observation map. Do not choose a new setting merely to hide an already-successful simple baseline.
''')
write('research_ideas.md','''# 一条待发展的候选；最终批准方向仍为零

Parent: reward-only exogenous-feature failure + existing causal state abstraction. Operation: refine to physically coupled incoming innovations, and instrument input information separately from action recovery and target predictability.

**候选问题：** 能否在无需奖励、无需手工指定物理状态通道的条件下，学得紧凑且对动作条件演化充分的世界状态，同时保留不可控但影响结果的当前驱动？

**本轮支持：** 三类系统中接近完美的逆动作预测仍伴随随机gain丢失和响应幅度错误；强探针未恢复。更大状态修复，说明有明确可解决性，也给方法提出强容量/预算对照。

**拟检验机制：** 在完整观测teacher中用支持内动作查询估计响应输出空间；把该空间作为预测目标，而不是把“输入状态必须记住什么”和“未来目标必须可预测什么”强行设成同一对象。已有因果抽象、控制子空间、导数降维可能实现同样原则；当前不把这段思路称为已证实创新。

**具体救援与反证：** cycle21同时比较完整future-bank蒸馏、单纯action-contrast、投影future-bank、同rank随机投影、额外预算重建/预测、更大状态和规范化EMA路由混合。成熟active-subspace已在同pilot上配合动作响应目标成功，故当前不提出新的神经压缩器。它与候选共享目标选择成分，后续应检验目标选择是否确有独立价值、是否超出已知因果降维，以及跨系统和非线性观测能否成立。若只在线性观测toy成立，不能提交一般world-model框架结论。

它是当前最值得继续验证的候选，不是最终已推荐选题。真实/native任务、长期闭环模拟与精确新颖性仍缺失。
''')
write('final_research_direction.md',f'''# 当前候选：输入状态的因果充分性与预测目标的分离

**Research Question：** 通用时间序列世界模型在学习紧凑状态时，如何保留当前已观测、但未来难以预测且会影响动作结果的变量？问题属于世界状态学习，不是某个数据集的预测分数优化。

**Evidence：** 已在服务器完成72次训练。三个系统的4D joint+inverse动作恢复均接近100%，gain识别仍约50%，响应差NRMSE约1；q,p可读且表示未整体坍塌。非线性和树探针没有恢复。12D和原始观测对照基本修复，因此当前证据支持容量约束下的状态选择问题，不能宣称所有模型不足。

**Existing Methods：** CDL早已明确不可控但影响动作结果的变量应保留；多步逆模型、双空间、随机JEPA、主动子空间和导数降维均是现有方法。需要证明候选的具体机制和使用条件超出它们，而非只换名称。

**Proposed Idea：** 待验证的架构无关原则：用动作响应识别有依据的输出目标，再要求内部状态保留预测这些输出所需的输入信息。成熟敏感性压缩器配合该目标选择已在同pilot的九个组合成功；这排除了新神经压缩器的必要性，但其共享目标选择成分尚需文献和独立数据核查。cycle21的teacher只用普通离线观测/动作训练，不读取物理标签或奖励；通过支持内动作查询学习投影。尚无可宣称的新方法收益。

**Falsification：** 新神经压缩器主张已被成熟方法修复证据否定。若同容量/预算重建、future-bank蒸馏、随机投影或既有因果抽象无需待检验的目标选择也同样有效，则目标选择的必要性也减弱；只有线性观测小系统有效，不足以支持一般框架。不能靠调测试集权重取得正面结果。

**Feasibility：** 单张RTX5090已完成主pilot约13.3分钟；独立算术、权重、数据划分和重放检查通过。cycle21使用新轨迹和固定预算，增加队列压力测试、teacher/随机投影/预算/容量/随机世界模型对照。所有source/checkpoint/data/logs保存在独立服务器目录。

**Next Experiment：** cycle21独立cohort原型正在执行；在其全部模型固定前不检查测试效果。成熟敏感性降维已完成同pilot对照，还需独立cohort复核、精确closest-work对照和一个合法真实或source-native时间序列任务。真实数据有效性和长期模拟能力尚未建立。

**当前判断：** 只有一条待发展的候选，零条已通过最终新颖性/证据标准的研究方向。总体目标仍ACTIVE、UNACHIEVED；没有公开发布。
''')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(13.5,7),layout='constrained');x=np.arange(8)
labels=['Joint','+inverse','EMA','Diag-G','2 heads','Obs pred.','Raw','12D inv.']
for j,f in enumerate(families):
    ax=axes[0,j]
    for i,m in enumerate(methods):
        stat=groups[(f,m)]['gain_accuracy'];mean=stat['mean'];ax.errorbar(i,mean,yerr=[[max(0,mean-stat['min'])],[max(0,stat['max']-mean)]],fmt='o',color='#3375a2',capsize=3,label='Linear/MLP probe' if i==0 else None)
    for k,m in [(1,'joint_idm'),(6,'fullobs'),(7,'joint_idm_full')]:
        s=next(r for r in analysis['stronger_probe_groups'] if r['family']==f and r['method']==m)
        ax.scatter(k,s['gain_accuracy_mean'],marker='x',color='#b76426',s=50,label='Tree probe' if k==1 else None)
    ax.axhline(.5,c='#888888',ls='--',lw=1);ax.set_ylim(.44,1.04);ax.set_xticks(x,labels,rotation=45,ha='right',fontsize=8);ax.set_title(f.replace('_',' '));ax.set_ylabel('Observed gain recovery accuracy')
    if j==0:ax.legend(fontsize=8,loc='center right')
    ax=axes[1,j]
    for key,color,label,offset in [('native_modulation_relative_RMSE','#3375a2','World transition',-.06),('frozen_encoder_modulation_relative_RMSE','#b76426','Refit frozen-state probe',.06)]:
        values=[groups[(f,m)][key]['mean'] for m in methods]
        lower=[max(0,groups[(f,m)][key]['mean']-groups[(f,m)][key]['min']) for m in methods]
        upper=[max(0,groups[(f,m)][key]['max']-groups[(f,m)][key]['mean']) for m in methods]
        ax.errorbar(x+offset,values,yerr=[lower,upper],fmt='o-',c=color,lw=1,capsize=2,label=label)
    ax.axhline(1,c='#888888',ls='--',lw=1);ax.set_ylim(-.03,1.12);ax.set_xticks(x,labels,rotation=45,ha='right',fontsize=8);ax.set_ylabel('Gain-modulation relative RMSE')
    if j==0:ax.legend(fontsize=8,loc='lower left')
fig.suptitle('Controlled pilot: action recovery does not ensure causal-driver retention',fontsize=14)
fig.supxlabel('Three synthetic families;3 initializations per arm. Bars show seed ranges. Current gain is observed.\nMethods are mathematical task controls; no native-paper leaderboard claim. More capacity repairs the failure.',fontsize=10)
out=ROOT/'outputs/analysis'
for ext in ['png','svg','pdf']:fig.savefig(out/('causal_driver_pilot.'+ext),dpi=170)
plt.close(fig)
(out/'result_tables.md').write_text(full+'\n',encoding='utf-8');print(json.dumps(dict(reports=7,figure=str(out/'causal_driver_pilot.png'),accepted_novel_directions=0)))
