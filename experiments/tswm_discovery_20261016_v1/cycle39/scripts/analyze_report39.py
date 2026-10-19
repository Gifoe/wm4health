"""Describe all verified outcomes and close evidence, without accepting an idea."""
import pathlib,sys,json,csv,datetime,hashlib
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent;sys.path.insert(0,str(B/'cycle29/scripts'))
import runtime_profile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
Q=R/'outputs/native_queries_v2';K=R/'outputs/kernel_controls_v1';D=read(Q/'result.json');C=read(K/'result.json');V=read(Q/'verification.json');W=read(K/'verification.json')
assert V['passed'] and W['passed'] and V['results_sha256']==sha(Q/'result.json') and W['results_sha256']==sha(K/'result.json')
A=R/'outputs/analysis_v1';assert not A.exists();A.mkdir();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name,rows in [('all_query_moments.csv',D['rows']),('all_kernel_scores.csv',C['rows'])]:
    with (A/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in rows))));writer.writeheader();writer.writerows(rows)
group=[]
for task in ['solar','jena']:
    for arm in ['native_joint','canonical96','canonical8','ar8']:
        rows=[x for x in C['rows'] if x.get('dataset')==task and x.get('arm')==arm]
        group.append({'dataset':task,'arm':arm,'contexts':len({x['end'] for x in rows}),'mc_reps':2,
                      **{key:float(np.mean([x[key] for x in rows])) for key in ['hourly_crps','mean_crps','terminal_crps','ramp_crps','prediction_seconds']}})
summary=[]
for task in ['solar','jena']:
    for mode in ['native','fixed_reference']:
        for split in [96,352,360]:
            rows=[x for x in D['rows'] if x['dataset']==task and x['mode']==mode and x['split']==split]
            summary.append({'dataset':task,'mode':mode,'split':split,'moment_rows':len(rows),
                            'median_abs_mc_z':float(np.median([abs(x['mc_z']) for x in rows])),
                            'max_abs_mc_z':max(abs(x['mc_z']) for x in rows)})
analysis={'utc':now,'status':'completed','query_summary':summary,'kernel_descriptive_means':group,
          'query_result_sha256':sha(Q/'result.json'),'query_verification_sha256':sha(Q/'verification.json'),
          'kernel_result_sha256':sha(K/'result.json'),'kernel_verification_sha256':sha(K/'verification.json'),
          'model_state_unchanged':D['model_state_before']==C['model_state_after'],'test_scored':False,'accepted_directions':0,
          'interpretation':'Self-consistency residual and ordinary-control tradeoff on exploratory contexts; no verified need for new mechanism'}
(A/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':170})
fig,axes=plt.subplots(2,2,figsize=(11.4,7.2),constrained_layout=True)
colors={'native':'#bd763c','fixed_reference':'#367a93'}
for row,task in enumerate(['solar','jena']):
    for col,probe in enumerate([0,7]):
        ax=axes[row,col];contexts=sorted({x['end_index_exclusive'] for x in D['rows'] if x['dataset']==task})
        for mode in colors:
            for rep,offset,marker in [(2026391,-.12,'o'),(2026392,.12,'s')]:
                recs=[next(x for x in D['rows'] if x['dataset']==task and x['end_index_exclusive']==end and x['split']==96 and x['probe']==probe and x['mode']==mode and x['rep']==rep) for end in contexts]
                x=np.arange(4)+offset+(-.035 if mode=='native' else .035)
                ax.errorbar(x,[v['difference'] for v in recs],yerr=[1.96*v['mc_se_outer'] for v in recs],fmt=marker,
                            color=colors[mode],capsize=2,ms=4,label=mode+f', MC rep {rep%10}')
        ax.axhline(0,color='#777777',lw=1);ax.set_xticks(np.arange(4),[str(x) for x in contexts]);ax.set_xlabel('History end index')
        ax.set_ylabel('Direct − re-query moment');ax.set_title(task+(' — suffix mean' if probe==0 else ' — prefix-weighted terminal'))
        ax.grid(axis='y',alpha=.15)
axes[0,0].legend(fontsize=7);fig.suptitle('Fixed primary 96+96 query; intervals show outer-group MC error only',fontsize=11)
for ext in ['png','pdf']:fig.savefig(A/f'query_moments.{ext}')
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True)
arms=['native_joint','canonical96','canonical8','ar8'];labels=['Native joint','Canonical 96','Canonical 8','AR8'];palette=['#507b9c','#549284','#b98144','#8f8f8f']
for ax,metric,title in zip(axes,['hourly_crps','mean_crps','prediction_seconds'],['Marginal fair CRPS','192-point mean fair CRPS','End-to-end sampling seconds']):
    for i,(arm,label,color) in enumerate(zip(arms,labels,palette)):
        vals=[next(x[metric] for x in group if x['dataset']==task and x['arm']==arm) for task in ['solar','jena']]
        ax.bar(np.arange(2)+(i-1.5)*.19,vals,width=.18,color=color,label=label)
    ax.set_xticks([0,1],['Solar','Jena']);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    if metric=='prediction_seconds':ax.set_yscale('log')
axes[0].legend(fontsize=7);fig.suptitle('Exploratory: four early contexts × two MC reps per task; no neural training / no TEST',fontsize=10.5)
for ext in ['png','pdf']:fig.savefig(A/f'ordinary_kernel_controls.{ext}')
plt.close(fig)
base=f'UTC {now}。原生query诊断与普通kernel控制均完成且独立复核；最终接受方向0，整体ACTIVE_UNACHIEVED。新neural updates0；只有8个history-only AR8解析fits。'
scope='目标是可复用时间序列world-model框架；不把这些FEV片段的最优分数当idea，不强制action/causal/zero-shot、多模态或全新算子。'
table='| task | arm | marginal CRPS | whole-192 mean CRPS | terminal CRPS | max-ramp CRPS | sample seconds |\n|---|---|---:|---:|---:|---:|---:|\n'
for x in group:table+='| '+' | '.join([x['dataset'],x['arm']]+[f"{x[k]:.6f}" for k in ['hourly_crps','mean_crps','terminal_crps','ramp_crps','prediction_seconds']])+' |\n'
qt='| task | coordinates | split | median abs MC z | max abs MC z |\n|---|---|---:|---:|---:|\n'
for x in summary:qt+='| '+f"{x['dataset']} | {x['mode']} | {x['split']} | {x['median_abs_mc_z']:.3f} | {x['max_abs_mc_z']:.3f} |\n"
verification=f'''第一项独立检查8raw histories、48NPZ SHA、864moment rows，max probe差{V['max_probe_abs']:.12g}、summary差{V['max_summary_abs']:.12g}。第二项独立pairwise CRPS替代生产排序公式，16NPZ/256score cells、8AR拟合及递推，max score差{W['max_score_abs']:.12g}，AR replay差{W['max_ar_replay_abs']:.12g}，native及canonical第一prefix重放差{W['max_native_first_prefix_abs']:.12g}。两项model state同一SHA。6个same-noise API资格（长度672/768/1032 × native/reference）max relative差0；此API资格由worker执行，独立verifier没有再跑GPU API，不能夸为双实现模型全重放。'''
limits='只有每task4个早期history、2个MC repetitions；rep不是训练seed，重叠origin不是独立样本。MC CI只描述同history抽样误差，不是现实泛化区间；864个probes未校正多重比较。质量分数使用真实未来192点，但这些是先前探索过的早期载体、非独立TEST，预训练覆盖未知。未测试新增真实prefix后的固定tail条件质量，未执行FlowNP/SFM/ForcingDAS的完整训练；coherence≠accuracy。'
finding='原720head的短/长请求prefix本来一致；本轮测的是observed-history重编码后的law。固定原history坐标后，部分96+96 context仍有超MC误差的moment差，不能全归因RevIN或360点patch相位。具体例：solar end1536无权suffix mean两rep fixed-reference差约−.212/−.223（MC z约−12.67/−11.82）；Jena end1344 fixed-reference prefix-weighted terminal差约−.087/−.081（MC z约−3.92/−3.69）。这是必要moment的不兼容，不是已证明真实质量损害，更没有决定哪个law应当是teacher。'
decision='普通canonical kernels无需训练就定义一致的自身block law，但改变原joint law。canonical96约2.6倍原single-head采样耗时；本小样本上solar mean CRPS变差、terminal却改善，Jena主分数变化小；canonical8更慢且主分数更差，AR8便宜但质量混合。不能只挑一个metric断言新框架必要；也不能称普通一致性修复没有实际代价。当前保留的是固定future-tail真实reveal下的quality/cost问题，尚非被证明未解决的机制。'
literature='''## 摘要与问题

本轮由mainstream、critic、constructive三个独立检索视角延续各自上下文，不读当前siblings；root合并前另核关键原文。RQ1区分同joint条件律、latent CK、history重编码和真实forecast版本；RQ2检查强普通方案后的有限质量/成本残差；RQ3寻找最多一个精确新交互。既有C23–25 system-posterior、C34 first-hit MLE、C37 marginal-rank问题不复活。

## 按概率对象分类的交叉综合

| 对象 | 最近完整方法及具体边界 | 本轮能推出什么 |
|---|---|---|
| fixed finite joint的条件化 | NDP通过joint定义conditioning；换target∪context universe须另核边缘一致性 | 不能将不同query的密度比自动当固定joint归一条件 |
| amortized arbitrary conditional | FlowNP、MissPath-FM有真正可执行接口；FlowNP §3.3不保证marginal/conditional consistency | 无理论保证不等于质量失败，必须实际强对照 |
| latent多步composition | MDN-VJEPA的trajectory mode+sample CK，SC的resampling一致性训练 | 已覆盖宽泛正则；CK下降不等于真实accuracy改善 |
| 新观测后的分布transport | SFM旧预测source+hybrid训练；ForcingDAS按schedule filtering/smoothing | 暖启动和统一conditioning均非新机制；有限成本质量仍须同合同验证 |
| 同批paths上的函数查询 | PathFM cache/PDD/CRPS | 多functional不是新观测条件化；不能用旧sampler慢概括现代生成模型 |

[FlowNP §3.3](https://arxiv.org/html/2512.23853v1)仅架构保证exchangeability，joint与任意conditional一起训练，是最近完整对照。[SFM §3](https://arxiv.org/html/2602.05319v1)实际的hybrid source/target pairs不自动成为某冻结joint的精确Bayes条件；这限定理论外推，不能据此宣布它失败。[ForcingDAS](https://arxiv.org/html/2605.14285v1)已通过noise schedule统一过滤/平滑，校准是作者承认的未系统评估项。

[MDN-VJEPA](https://arxiv.org/html/2610.13956v1)已明说一致性与质量不同，某些CK指标改善伴随probability/calibration tradeoff；其保证不是观测history重编码。From Drift to Coherence是重复同QA目标的SC微调，不能把同样正则迁来称新机制。[Forecast sequence理论](https://arxiv.org/html/2606.31621v1)还需要终端outcome校准，内部martingale本身不足。

[MissPath-FM](https://arxiv.org/html/2610.17715v1) observed-value clamping保证返回值相等，不保证各mask来自同一未知部分条件law。[PathFM](https://arxiv.org/html/2610.22358v1)已组合联合path、cache、1–2step蒸馏与proper-score校准；code/weights仍承诺将发布，未称本项目运行其权重。组件已有不排除新交互，但不能把这一整套已存在功能重新命名。

## 测量反证与实测联系

仅suffix边际矩相等仍可丢失全部prefix相关。本轮Gaussian AR正/独立重抽负控证实测量灵敏度，因此采用prefix-weighted probes和outer-group MC。反过来，旧joint错、新requery对时也会检出不兼容，故非零不能决定应蒸馏哪个teacher。完整实测见experimental_results.md。

## 对三个问题的回答

RQ1各方法约束不同概率对象，不能用“Bayes/CK/consistency”混称。RQ2当前获得了native自查询残差和普通kernel质量/成本代价，但尚未证明强完整方法不足。RQ3没有精确新机制候选；下一项高信息试验是固定同tail、真实prefix揭示的FlowNP/SFM/AR强普通screen，而非直接开发新coherence loss。

## 来源与限制

7个主文HTML和2个固定作者接口文件已正常TLS归档，SOURCE_RECEIPTS给9个SHA；旧MDN/NDP/MPNP原文及精确章节见三个独立notes和旧轮archive。所有作者reported scores仅用于原文边界，非本项目复现。未可访问来源不填造数字；MDN metadata日期不直观，不凭搜索卡片断言首发日。
'''
reports={}
reports['literature_review.md']='# C39 跨查询概率律与状态更新的文献综合\n\n'+base+'\n\n'+scope+'\n\n'+literature
reports['experimental_results.md']=f'''# C39 完整实测

{base}

{scope}

冻结协议：[V2 native计划](NATIVE_QUERY_PLAN_V2.md)、[追加普通控制计划](ORDINARY_KERNEL_CONTROL_PLAN.md)。追加方案来自独立mainstream反证、在root读任何effect rows前写下。V1因solar前864点缺失导致有效context仅1而按原guard退出，未加载模型；原源码/日志保留。V2仅改为C34既有first-finite anchor+固定offsets，未按误差选窗口。

source-native Sundial revision3212e42564493f520593e5414af4367fc4b49226，128329680参数，50flow steps；原数据/weights沿用C34 hash，没有新下载signal/weight。8histories、2MC reps、splits96/352/360、Nouter256、Ninner8，full-growing history、native/fixed-reference两坐标。权重全程不变，9moment probes有Gaussian解析正/负控；inner先取非线性g再平均、fresh noise独立于prefix。

{verification}

## 全query分组描述

{qt}

{finding}

全部864行（包括不显著/反号）在[CSV](outputs/analysis_v1/all_query_moments.csv)。median/max z仅描述MC信噪，不作多重比较后的普遍failure结论。

## 普通kernel与真实未来质量控制

所有192点预测先用该history尺度归一。head_calls为native1、canonical962、canonical824；native一次也实际生成720维flow。AR8是history-only解析fit，跨未来不重新拟合。下表为4contexts×2MC reps均值，不是8独立随机训练。

{table}

{decision}

原native诊断{D['seconds']:.2f}秒，kernel控制{C['seconds']:.2f}秒；峰值显存{D['peak_allocated_bytes']/2**20:.1f}MiB。端到端采样列含每arm必要history编码，模型加载/warmup单列未计该列。8个AR拟合时间另在原JSON。

{limits}

![Query moments](outputs/analysis_v1/query_moments.png)

![Ordinary controls](outputs/analysis_v1/ordinary_kernel_controls.png)
'''
reports['baseline_registry.md']=f'''# C39 执行登记

{base}

| 对象 | 本轮执行 | 范围 |
|---|---|---|
| 固定Sundial128M | 原生成head及新增history重编码；6same-noise资格 | 组件诊断，不是原论文score复现 |
| fixed-reference | 同weights和inner noise，原坐标控制 | 普通RevIN解释对照 |
| canonical96/8 | 同weights，分别2/24完整head调用 | 已知wrapper，自身block law一致；不是原joint条件sampler |
| AR8 | 8个history-only ridge解析fits、16次MC | 固定系数普通law；非新neural训练 |
| FlowNP/SFM/ForcingDAS/MissPath-FM | 原文及部分固定代码接口审计 | 未实训，不能称已被本实验击败 |
| PathFM | 近期完整原文 | code/weights待发布，未执行 |

{verification}

{limits}
'''
reports['negative_evidence.md']=f'''# C39 反证与限制

{base}

“随机JEPA/噪声分离”宽泛动机已因历史C11/14/20/21及完整近邻关闭；未重新训练。自条件化也有NP、NDP、SC/CK、SFM等完整先例，不能只加一致性loss当创新。

{finding}

{decision}

{limits}

V1数据reader选择从0开始是可避免预检错误，原失败保留；V2依据事先存在的first-finite元数据修正，不含性能选择。未把失败算作模型缺陷。C23–25已关闭配置和C34first-hit MLE不复活。
'''
reports['hypothesis_space.md']=f'''# C39 竞争解释

{base}

| 假设 | 已执行/下一项区分 | 当前判断 |
|---|---|---|
| 只是MC | outer分组、2独立MC reps、正/负控 | 部分固定context残差超该MC误差；非泛化证明 |
| 只是RevIN/patch phase/遗忘 | fixed-reference、96/352aligned vs360、full-growing history | 部分96残差仍在，不能全归因这些项 |
| 只是缺少一个wrapper | canonical96/8同weights真实质量与耗时 | 能定义一致law但有本小样本代价；未排除更强ordinary方法 |
| 原joint错而requery更好 | 需真实prefix固定tail proper score | 目前未执行；不能选原joint作真teacher |
| 需要新的state/transition机制 | FlowNP/SFM/AR强普通及SC/CK修复 | 尚无证据，尚无新机制 |

{limits}
'''
reports['research_ideas.md']=f'''# C39 当前工作问题

{base}

{scope}

保留一个具体问题：**固定同一未来tail，当真实观测分批揭示时，可复用world-model状态如何在较低成本下更新条件分布，同时保留有用预测质量？** native重编码与原joint不兼容是自生成证据；ordinary kernels的代价只是有限screen，未证明完整近邻不足。

没有接受方法。FlowNP arbitrary-query、SFM old-belief source、ForcingDAS统一schedule、PathFM cache/distill/calibrate都已覆盖可直接写出的宽泛组合。不能把teacher conditional amortization或SC loss换名。root优先后续同合同普通方法screen，暂不开发新架构；若强对照足够就关闭此问题的具体动机。
'''
reports['final_research_direction.md']=f'''# C39 研究方向尚未最终成立

{base}

**Research Question：** {scope} 当前优先检验固定future-tail真实reveal下的state/conditional-law质量—成本关系。

**Evidence：** source-native fixed128M模型8contexts、864moments、6API资格，以及4arm普通控制64score rows/8AR解析fits均独立核查。{finding}

**Existing Methods：** canonical AR已能定义coherent law；FlowNP、SFM、ForcingDAS、NDP及SC/CK训练尚未在同合同被排除。PathFM进一步覆盖快速joint paths，不能用旧成本作普遍新颖性。

**Proposed Idea：** 尚无通过的新机制。当前获得的是研究问题的具体证据和可失败下一步，不冒充完整创新方向。

**Falsification：** 若ordinary conditional/AR/SFM/SC可同时达到实用质量与成本，或requery虽不兼容却更准，或现象只限当前8contexts，就不能据此立新框架。

**Feasibility：** 单5090已完成本轮约{D['seconds']+C['seconds']:.1f}秒两项主计算，无neural训练。{limits}

**Next Experiment：** 冻结独立时间split与固定tail的真实prefix reveal协议，先执行同信息AR-spline、FlowNP-style、SFM-style普通screen；必要再加已知SC/CK修复，明确模型/训练/采样/cache真实成本。该建议尚未执行或变成新方法，不能算完成目标。最终推荐方向0，整体ACTIVE_UNACHIEVED。
'''
reports['CURRENT_STATE.md']=f'''# C39 两项诊断完成，目标未完成

{base}

{verification}

{decision}

原V1失败、V2成功、kernel控制各独立目录保留。所有raw samples/weights/source在服务器；紧凑JSON/报告/plots按SHA镜像。{limits}
'''
for name,content in reports.items():
    p=R/name;assert not p.exists();p.write_text(content,encoding='utf-8')
manifest={'utc':now,'reports':{name:sha(R/name) for name in reports},'analysis_sha256':sha(A/'analysis.json'),
          'query_verification_sha256':sha(Q/'verification.json'),'kernel_verification_sha256':sha(K/'verification.json'),
          'accepted_directions':0,'goal':'ACTIVE_UNACHIEVED','script_sha256':sha(pathlib.Path(__file__))}
(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'reports':len(reports),'figures':4,'moment_rows':len(D['rows']),'kernel_rows':len(C['rows']),'goal':'ACTIVE_UNACHIEVED'}))
