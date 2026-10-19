"""Seven reports from completed independently checked results; no model selection."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/response_target_prototype'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    verification=read(BASE/'independent_verification.json');assert verification['pass']
    result=read(BASE/'results.json');fits=read(BASE/'fit_records.json');teachers=read(BASE/'teacher_records.json')
    classical=read(ROOT/'outputs/active_subspace_v1/results.json');assert read(ROOT/'outputs/active_subspace_v1/independent_verification.json')['pass']
    families=result['config']['families'];methods=result['config']['methods'];groups=[]
    keys=['gain_accuracy','inverse_joint_action_accuracy','native_physical_MSE','native_modulation_relative_RMSE','frozen_encoder_successor_probe_MSE','current_physical_decoding_MSE']
    for family in families:
        for method in methods:
            rows=[v for v in result['rows'] if v['family']==family and v['method']==method];assert len(rows)==3
            g=dict(family=family,method=method,n_initializations=3,metrics={})
            for key in keys:
                values=[v[key] for v in rows];g['metrics'][key]=dict(mean=float(np.mean(values)),min=float(min(values)),max=float(max(values)),sd=float(np.std(values,ddof=1)))
            groups.append(g)
    by={(v['family'],v['method']):v['metrics'] for v in groups}
    def val(f,m,k):return by[f,m][k]['mean']
    table=['| Family | Method | Gain acc. | Inverse acc. | Physical MSE | Modulation NRMSE | Frozen-state next MSE |','|---|---|---:|---:|---:|---:|---:|']
    for f in families:
        for m in methods:table.append('|'+f+'|'+m+'|'+'|'.join(f'{val(f,m,k):.6f}' for k in ['gain_accuracy','inverse_joint_action_accuracy','native_physical_MSE','native_modulation_relative_RMSE','frozen_encoder_successor_probe_MSE'])+'|')
    table='\n'.join(table)
    active_table=['| Family | Output choice | Gain acc. | Physical MSE | Modulation NRMSE |','|---|---|---:|---:|---:|']
    for f in families:
        for choice in ['all_outputs','action_outputs']:
            rows=[v for v in classical['rows'] if v['family']==f and v['output_choice']==choice]
            active_table.append('|'+f+'|'+choice+'|'+'|'.join(f'{np.mean([v[k] for v in rows]):.6f}' for k in ['gain_accuracy','native_physical_MSE','modulation_relative_RMSE'])+'|')
    active_table='\n'.join(active_table)
    readiness='\n'.join(f"- {f}: teacher readiness={result['teacher_validation_readiness'][f]}, validation MSE={[round(v['validation_physical_MSE'],6) for v in teachers if v['family']==f]}" for f in families)
    comparisons=[]
    for f in families:
        for candidate in ['projected_bank','projected_mop','response_only']:
            for baseline in ['joint_budget','reconstruction','forecast_bank','mop_budget']:
                pairs=[]
                for seed in [2021,2022,2023]:
                    a=next(v for v in result['rows'] if v['family']==f and v['method']==candidate and v['seed']==seed)
                    b=next(v for v in result['rows'] if v['family']==f and v['method']==baseline and v['seed']==seed)
                    pairs.append(a['native_physical_MSE']-b['native_physical_MSE'])
                comparisons.append(dict(family=f,candidate=candidate,baseline=baseline,paired_physical_MSE_differences=pairs,mean_difference=float(np.mean(pairs))))
    out=ROOT/'outputs/analysis';out.mkdir(exist_ok=True)
    summary=dict(groups=groups,paired_descriptive_comparisons=comparisons,teacher_readiness=result['teacher_validation_readiness'],accepted_novel_directions=0,objective_achieved=False)
    (out/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    def write(name,text):(ROOT/(name+'.md')).write_text(text.strip()+'\n',encoding='utf-8')
    write('literature_review','''# World-model scope and novelty boundary

目标是通用时间序列世界状态与条件演化框架，数据集是证据载体。最新范围与具体源码依据见 RESEARCH_SCOPE_UPDATE.md 和 LITERATURE_INTERFACE_BOUNDARY.md。

WorldTS已分别编码目标序列与条件，不能把“分开条件和预测目标”本身当成创新。CDL与Denoised MDPs明确保留影响动作结果的不可控因子；主动子空间、导数辅助输入/输出降维、非线性平衡降阶是成熟机制。H-JEPA的双空间、MoP-JEPA的EMA归一化/路由与Var-JEPA随机建模也已存在。本轮不宣称这些原生模型被复现或被推翻。

候选剩余问题仅为：在没有手工目标/协变量角色划分的观测中，能否从条件响应学到合适的预测目标，并形成紧凑且充分的输入状态？当前尚无已核实的新颖性结论。新文献原文、固定版本、代码字节和哈希见 outputs/literature。JMLR导数降维及CoBRAS/平衡降阶的筛查不等于完整复现。
''')
    write('baseline_registry',f'''# Completed independent-cohort development

四个合成系统，3个初始化，10个学生配置，共120次学生训练和12个完整观测teacher。全部模型/探针选择固定后统一评估测试集。新轨迹、旋转和teacher权重均未复用cycle20；前三族机制相同，queueing是新的简单裁剪机制，仍不是真实数据。

所有学生64-wideMLP，4D状态；wide_budget为12D容量控制。joint/obs/reconstruction/wide/MoP为3000更新；四种teacher-bank与projected_mop为1500学生+1500teacher更新。辅助头48输出、weight1，BANK类有相同结构；teacher、辅助头参数和时间单独记录。这个更新预算对照不等于精确FLOP/墙钟匹配。原型MoP保留归一化、EMA.99和context-only路由，额外加入共同inverse/variance约束，是任务迁移而非原生OGBench复现。

旧方法控制24次：同一批12个teacher，全部输出或动作响应输出目标，前4个输入梯度协方差方向、train-only whitening、多项式ridge，由验证latent误差选阶数/正则。物理标签仅用于冻结后的诊断解码。既有压缩器与待核查目标选择必须分开归因。

{readiness}

数据、120权重/原始预测、12teacher、24旧方法系数/预测和job日志保存在服务器cycle21/outputs。紧凑报告和结果镜像在本地；不声称所有大数组已下载。没有修改此前环境或公开发布。
''')
    write('experimental_results',f'''# Independent-cohort results

主候选在执行前固定为projected absolute future-bank；response_only是预先列出的竞争机制。即使后者表现更好，也只能形成下一轮机制假设，不能把主候选改名后称为事先确认。所有配置均完整报告。

**实质判断：** 主候选未取得稳定的预期结果。在前三类独立cohort中，projected_bank的gain准确率均值分别为49.95%/50.12%/67.88%，前两族响应NRMSE接近1。response_only三个系统的9个初始化全部gain准确率100%，物理MSE均值0.00232/0.00122/0.00219，响应NRMSE0.06594/0.08962/0.06567；这是差分目标有价值的开发证据。队列族却是普通obs_budget更好：物理MSE0.00514、响应0.07618，而response_only分别0.03420/0.20134。因此不能把差分目标宣称为统一获胜方法。更强随机控制有部分收益，不能用旧的简化双头失败来概括所有随机方法。

{readiness}

{table}

三个seed是相同数据cohort上的初始化变异，不是三个独立真实系统。gain为当前已提供、未来未知的二值随机驱动。Modulation NRMSE比较相同其余条件下两种支持内gain的物理后果；0为精确，1约为忽略差异。真值来自生成方程，不代表观测数据因果识别。queueing还有动作等价/裁剪，不能将全部误差解释为gain丢失。

## Mature reduction control

{active_table}

动作输出版本共享候选目标选择成分；与全部输出版本的差异用于定位目标的作用。成功只证明旧压缩器足够，不证明目标选择新颖。保留失败、种子变异及不满足teacher readiness的系统，禁止测试后调rank、loss权重或decoder。

新cohort上，动作目标+成熟压缩器在前三族全部gain准确率100%，响应NRMSE均值0.00180/0.00356/0.00423；queueing物理MSE0.03870，仍不及普通观测预测0.00514。绝对投影future中的信息并非原则上不足：同一投影用于成熟敏感性压缩器可成功，而joint neural辅助目标失败，说明不能把现象单纯归因为目标没有信息。优化、目标权重与压缩器归纳偏置仍是竞争解释。

补充旧方法第一次启动在拟合前因模块别名被局部变量遮蔽而退出1。原始源码快照与失败日志保留；只修正命名，另起attempt1输出目录完成24次拟合和独立monomial重放。见CLASSICAL_CONTROL_REPAIR.md。主120-fit job和验证均一次exit0。

主job总耗时{result['elapsed_seconds']:.2f}s，GPU峰值分配{result['environment']['maximum_allocated_bytes']/2**20:.1f}MiB。数据哈希{verification['dataset_hashes']}、模型/预测各120、teacher12经复核；{verification['metrics_recomputed']}个指标独立重算最大差{verification['metric_max_error']:.3g}；完整batch权重重放最大差{verification['checkpoint_replay_max_error']:.3g}。原始方程、观测逆变换、validation选择和测试时序另有检查。receipt范围是可重复性，不是新颖性证明。

![Independent cohort](outputs/analysis/response_target_comparison.png)

完整每seed数值、参数/时间、验证选择与逐项比较见原始results/fit_records和outputs/analysis/analysis.json。这里尚无真实数据、非线性观测、跨系统共享参数或闭环多步概率模拟结果。
''')
    write('negative_evidence','''# Negative evidence and unresolved alternatives

- 更大状态或普通完整观测模型可修复cycle20现象；没有普遍架构不可能性。
- 成熟敏感性压缩器配合动作输出目标已成功，新的神经压缩器不是必要结论。
- 同一绝对投影目标在敏感性压缩中可保留gain，在神经辅助目标中未稳定保留；不能宣称它在信息论上丢掉了因果驱动。尚未排除目标权重/优化与结构偏置解释。
- 这里把动作无关AR坐标视为相对于控制物理结果的干扰，并非声称通用预测应忽略它们。若任务要求预测完整观测，它们就是合法预测信号；只报告特权物理读出不足以证明通用world-model能力更强。
- WorldTS的命名条件接口已区分目标/条件；当前混合输入任务不能直接作为它失败的证据。
- 主候选projected_bank与response_only不同。不能用后者的正面测试结果替代前者的失败，也不能把普通重建、随机或规范MoP的成功删掉。
- projected_bank在coupled_linear/rotating未保留gain，在saturating有seed变异；原始“绝对未来投影比差分更好”的候选机制未获支持。response_only在前三族成功，但queueing由普通obs_budget明显超过；无跨机制统一胜者。
- 物理标签仅用于诊断，但诊断decoder/探针依然是特权评估，不是无需标签的完整可部署读出系统。
- 线性正交混合、有限二值动作、已知当前gain构成受控问题，可能偏向global输出span。实际模态、非线性传感几何和未知未来条件尚缺证据。
- 队列机制有已知action-equivalence限制，不能归因到一个新机制。teacher若未通过预设readiness，则该族不能支撑方法收益。
- 3000更新与teacher+student各1500只控制更新数量，teacher训练/查询与辅助头都有成本；参数和耗时必须一起解读。
- 最终新颖方向仍为0；不把一次合成prototype当成通用时间序列世界模型论文。
''')
    write('hypothesis_space','''# Competing hypotheses

| Hypothesis | Discriminating control | Interpretation rule |
|---|---|---|
| More updates suffice | joint_budget versus earlier1500-fit | Ordinary optimization alternative |
| Any current information grounding suffices | reconstruction | Successful reconstruction weakens selector necessity |
| Teacher or auxiliary capacity alone suffices | forecast_bank,random_projection | Compare same teacher/head and actual budget |
| Full projected future is sufficient | projected_bank | Primary frozen mechanism; do not substitute contrast if it fails |
| Only action contrast isolates the important driver | response_only | Predeclared competitor; positive evidence is mechanism development |
| Stochastic target treatment suffices | mop_budget,projected_mop | Task-adapted source-informed control, not all stochastic models |
| A new neural compressor is needed | Established active-subspace arms | Success falsifies this on the tested systems |
| Named covariate/target roles solve the realistic problem | Native conditional-state /WorldTS interface | Not yet executed; essential before general framework claim |

Next decisive evidence should reduce uncertainty about unknown roles, realistic support and multi-step state evolution. Repeatedly tuning the inspected synthetic task will not establish a new general problem.
''')
    write('research_ideas','''# Candidate development after a fixed experiment

Question: how can a time-series world model retain observed transition-relevant inputs when their future values are intrinsically unpredictable, without forcing every observation into a self-predicted target?

The generic premise is old: CDL,Denoised MDPs,predictive-state theory and conditional state-space models cover causal relevance and separate conditioning. Any possible contribution must isolate automatic target/role discovery under non-factorized observations and show a capability which those methods do not already supply under matching assumptions.

The fixed primary prototype uses projected future banks; action-contrast targets and established reduction are competing mechanisms. Their individual results are reported rather than folded into a post-test selected 'new architecture'. A useful rescue may be a simpler classical compressor plus a precisely justified target-selection rule. It remains only a development route until realistic/nonlinear/multi-step evidence and closest-work review pass.

Scope boundary: action-independent AR coordinates are nuisance only relative to the chosen controlled-physical-output evaluation. They remain legitimate signals for full-observation forecasting. This experiment cannot elevate action relevance to the sole definition of a general time-series world state. The final framework's target capability must be explicit.

结果后的细化：projected-bank原假设未稳定成立；response-only在前三种增益调制系统的9个初始化均保留gain，在队列中却败给普通观测预测。下一轮若继续，问题应改为“何种响应目标保留何种动力学信息、何时应该退回普通条件预测”，而不是把response-only直接推广成通用框架。新假设需要新的独立任务，当前测试集不再调参。

One development candidate,zero final accepted directions. No additional dataset-specific loss/weight search is authorized by this interpretation of the evidence.
''')
    write('final_research_direction','''# Current research judgment

**Research Question:** can compact time-series world-state learning discover which observed information must condition evolution, separately from which outputs should be predicted, when signal roles are not supplied?

**Evidence:** completed72-fit pilot and120-fit fresh-cohort development, plus strong frozen probes and established sensitivity-reduction controls. All configurations are reported and independently checked. The phenomenon concerns state selection with a finite bottleneck; larger states repair it.

The fixed projected-future candidate did not consistently preserve the driver. The predeclared response-only competitor recovered it in9/9 original-family fits with substantially lower physical/response errors. An ordinary observation predictor outperformed it on the queueing stress test. There is no universal winning method, and no post-test replacement of the primary hypothesis.

**Existing Methods:** WorldTS already separates named target/covariates. CDL andDenoised MDPs already preserve uncontrollable causes of controlled outcomes. Active/derivative subspaces and balanced reduction already supply mature compression machinery. A general role-splitting architecture is not a novel contribution by itself.

**Proposed Idea:** only a provisional automatic output-target/role-discovery mechanism. The original projected-future candidate and action-contrast competitor must keep separate identities. Keep the simplest successful established compressor; no new-neural-architecture claim is supported solely by this prototype.

**Falsification:** ordinary role-aware conditional models solving the realistic setting, equivalent improvements from matched reconstruction/random/full-bank objectives, or closest work covering the same automatic selector eliminate the claimed gap. Synthetic-only gains do not suffice.

**Feasibility:** singleRTX5090,existingPython3.10/torch2.8 environment; all raw artifacts retained on server. There are no source-nativereal-data or multi-step probability results for this candidate yet.

**Next Experiment:** first distinguish the named-covariate interface from automatic role discovery on an independently audited real/source-native time-series task; enforce future-information availability and supported conditions. Then test multi-step evolution with stochastic drivers rather than oracle future inputs. No further test-set tuning on the current toys.

**Decision:** one provisional route,zero accepted final novel directions. Overall objectiveACTIVE,UNACHIEVED. Current findings refine the world-model research problem and do not justify declaring a complete novel framework.
''')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    labels=['Joint','Obs','Recon','Full bank','Contrast','Proj.bank','Random','12D','MoP','Proj.MoP']
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained');x=np.arange(10)
    for j,f in enumerate(families):
        for axis,key,caption in [(axes[0,j],'gain_accuracy','Current gain recovery'),(axes[1,j],'native_modulation_relative_RMSE','Physical response NRMSE')]:
            values=[by[f,m][key]['mean'] for m in methods];lower=[v-by[f,m][key]['min'] for v,m in zip(values,methods)];upper=[by[f,m][key]['max']-v for v,m in zip(values,methods)]
            axis.errorbar(x,values,yerr=[lower,upper],fmt='o',capsize=2,color='#3375a2');axis.set_xticks(x,labels,rotation=60,ha='right',fontsize=7)
            axis.set_ylabel(caption);axis.axhline(.5 if key=='gain_accuracy' else 1,c='#888888',ls='--',lw=1)
            if key=='gain_accuracy':axis.set_ylim(.42,1.06);axis.set_title(f.replace('_',' '))
    fig.suptitle('Independent-cohort development: all fixed objectives,3 initializations',fontsize=14)
    fig.supxlabel('Current driver observed; future driver unknown. Bars show initialization ranges.\nSynthetic task adaptations; no source-native leaderboard or novelty claim.',fontsize=10)
    for ext in ['png','svg','pdf']:fig.savefig(out/('response_target_comparison.'+ext),dpi=160)
    plt.close(fig);(out/'result_tables.md').write_text(table+'\n\n'+active_table+'\n',encoding='utf-8')
    print(json.dumps(dict(reports=7,groups=len(groups),active_models=len(classical['rows']),accepted_novel_directions=0)),flush=True)
if __name__=='__main__':main()
