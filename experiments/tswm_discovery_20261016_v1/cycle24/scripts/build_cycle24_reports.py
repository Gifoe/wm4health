"""Reports/figures derived from verified, numerically corrected frozen outputs."""
import csv,hashlib,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
FILE=ROOT/'outputs/verified_results/results.json';B=json.loads(FILE.read_text());M=B['main'];P=B['particle'];O=B['ordinary']
assert all(v['status']=='pass' for v in B['verification'].values())
def group(kind,n,info):
    cc=[c for c in M['cells'] if (c['kind'],c['context_size'],c['physically_informative'])==(kind,n,info)]
    oo=[c for c in O['cells'] if (c['kind'],c['n'],c['informative'])==(kind,n,info)]
    pp=[c for c in P['cells'] if (c['kind'],c['n'],c['informative'])==(kind,n,info)]
    result={}
    for arm in ['no_update','native','reweighted','physical_oracle']:
        result[arm]={k:float(np.mean([c['actual_query'][arm][k] for c in cc])) for k in ['nll','rmse','marginal_coverage95','predictive_variance']}
    for arm in ['ekf','map_ggn_laplace']:
        result[arm]={k:float(np.mean([c['actual_query'][arm][k] for c in oo])) for k in ['nll','rmse','marginal_coverage95','predictive_variance']}
    for count in ['256','1024','4096']:
        result['particle'+count]={k:float(np.mean([c['actual_query'][count][k] for c in pp])) for k in ['nll','rmse','marginal_coverage95','predictive_variance']}
    return result
groups={(k,n,i):group(k,n,i) for k in [1,2] for n in [0,2,5,9] for i in [False,True]}
OUT=ROOT/'outputs/analysis';OUT.mkdir(exist_ok=True)
rows=[]
for (kind,n,info),g in groups.items():
    for arm,metrics in g.items():rows.append(dict(variant=kind,context_size=n,informative=info,method=arm,**metrics))
with (OUT/'full_results.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cold=[]
for c in [c for c in M['cells'] if c['context_size']==0 and not c['physically_informative']]:
    native_delta=c['actual_query']['native']['nll']-c['actual_query']['no_update']['nll']
    pc=[x for x in P['cells'] if (x['kind'],x['seed'],x['n'],x['informative'])==(c['kind'],c['checkpoint_seed'],0,False)]
    ci=next(x for x in B['verification']['main']['paired_context_nll_differences'] if (x['kind'],x['seed'],x['n'],x['informative'])==(c['kind'],c['checkpoint_seed'],0,False))
    pd=float(np.mean([x['actual_query']['256']['nll'] for x in pc]))-c['actual_query']['no_update']['nll']
    cold.append(dict(kind=c['kind'],seed=c['checkpoint_seed'],native_delta=native_delta,ci=ci['context_bootstrap95'],particle256_delta=pd))
cold.sort(key=lambda x:(x['kind'],x['seed']))
average_native=float(np.mean([r['native_delta'] for r in cold]));average_particle=float(np.mean([r['particle256_delta'] for r in cold]));fraction=1-average_particle/average_native
ct='| Variant / checkpoint seed | Native Δjoint NLL | 95% context bootstrap | Known particle256 ΔNLL |\n|---|---:|---|---:|\n'
for row in cold:ct+=f"| {row['kind']} / {row['seed']} | {row['native_delta']:.4f} | [{row['ci'][0]:.4f}, {row['ci'][1]:.4f}] | {row['particle256_delta']:.4f} |\n"
table='| Variant | History n | Probe informative | No update | Native | Dense Bayes | EKF | MAP-GGN | Particle256 | Physical oracle |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n'
for (kind,n,info),g in groups.items():
    table+=f"| {kind} | {n} | {'yes' if info else 'no'} | "+' | '.join(f"{g[arm]['nll']:.4f}" for arm in ['no_update','native','reweighted','ekf','map_ggn_laplace','particle256','physical_oracle'])+' |\n'
mv=B['verification']['main'];ov=B['verification']['ordinary'];tv=B['verification']['tail_patches']
timing=float(np.median([c['max4096_update_seconds'] for c in P['cells']]))
reports={
'literature_review.md':'''# 文献与创新边界

本轮新增阅读[Variational meta-learning inference for low dimensional neural system identification](https://arxiv.org/abs/2607.18965)，2026-07-21。其方法已有潜在系统参数流形、摊销初始化、MAP与Laplace近似。本文只核查相关方法边界，未复现该JAX实现，也未独立认可其全部数学/校准主张。原文已保存outputs/literature/，SHA256可查。

旧边界继续成立：[Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html)已有上下文条件动力学与主动校准；[MPNP](https://arxiv.org/abs/2304.09431)、[From Drift to Coherence](https://proceedings.mlr.press/v306/kim26ah.html)、[CNP conditioning gap](https://arxiv.org/abs/2604.19312)覆盖概率一致性与近似推断的重要部分。CNP factorized-joint条件化与本轮共享latent联合律的tower检验不是同一量，不能混为一谈；但泛化的“更新一致性”并非无人研究。

EKF、MAP/Laplace、importance sampling、defensive mixture proposal、Gaussian moment projection/assumed-density filtering均为已知方法。本轮把它们用于原生检查点的失败归因，不是提出新算法。不能把“编码器提议+似然校正”改名为新的世界模型框架。

当前有实测的可靠性问题，但没有足以证明现有方法仍不足的证据。更广的高维、非线性、部分可观测场景仍需检验；也不能仅通过让已有方法进入不合理的预算或支持范围制造失败。
''',
'baseline_registry.md':'''# 模型、数据与运行身份

- 原生模型：EmbodiedVision/explorethecontext，修正模型选择分支f0baf6262523d9c5abaf8c0381b4db97818f7c34。六个作者validation-best检查点；action_squash_1/2各3训练seed，原始50k训练预算；无新训练。三种seed不是三份独立测试数据。
- 两变体共享一个两状态plant，只改变actuation map。每变体64个新alpha/初态/控制/噪声实现，共128；源模型训练数据本轮未下载，未进行原训练参数ID逐一重叠核查，不声称严格原参数集合去重。新生成流及新轨迹与现有评测分离，不通过新指标选择checkpoint。
- 原物理核、参数范围、噪声、控制支持不变。连续历史前缀n=0/2/5/9，避免随机漏掉的历史让“精确后验”定义错误。query从独立固定状态0开始，probe也从0开始；区分系统信念与当前状态变化。测量不是临床或真实设备数据。
- 对照：不更新；原生重新编码；同decoder完整一维Bayes积分；单步EKF；MAP+GGN局部Gaussian；defensive proposal importance校正+Gaussian矩投影（256/1024/4096粒子，每条件3MC seed）；已知物理alpha后验。最后一项是privileged reference，不能冒充通用学习模型。
- 主48条件在看结果前固定；普通滤波/粒子对照为事后诊断，计划先于各自执行，但不算独立确认。所有方法接收相同真实probe/query，未使用query结果选择粒子数。

运行环境：服务器RTX5090、PyTorch2.8+cu128、Python3.10、原生float32数值网络，概率积分/聚合float64。source和weight不变，旧环境未修改；复用cycle23隔离的legacy依赖与已验证max适配。详细配置/sha、原始日志、所有失败/修复尝试和检查点来源均保留。大数组在服务器，本机紧凑镜像。
''',
'experimental_results.md':f'''# 实验结果：无信息证据与物理预测

主问题是同一未知系统的分布能否在未来分支/新证据间保持一致；这不是数据集SOTA优化。原生物理核中探测响应为A*x+[alpha*g(u),0]+noise。probe初态0，g(u)=0时观测与alpha独立。query另从0使用informative action。原生latent是隐变量beta，不要求它等于或可辨识为alpha；检验的是物理预测。

**冷启动、无信息probe：** 六个原生检查点均出现真实query联合NLL恶化。每seed的64个物理context bootstrap区间都高于0。单位是两输出联合NLL的nats，不是每时间点轨迹覆盖或每变量NLL。

{ct}

六checkpoint平均native损害{average_native:.6f}nats，普通256粒子矩投影后残余{average_particle:.6f}；相对降幅{100*fraction:.2f}%。这是同一已检查cohort上的事后已知方法对照，不能据此宣称新方法或独立确认。两个变体的95%逐变量边际覆盖率：native {100*groups[1,0,False]['native']['marginal_coverage95']:.2f}%/{100*groups[2,0,False]['native']['marginal_coverage95']:.2f}%，不更新{100*groups[1,0,False]['no_update']['marginal_coverage95']:.2f}%/{100*groups[2,0,False]['no_update']['marginal_coverage95']:.2f}%，particle256 {100*groups[1,0,False]['particle256']['marginal_coverage95']:.2f}%/{100*groups[2,0,False]['particle256']['marginal_coverage95']:.2f}%。旧prior也不完全达到95%，修复更新不等于已解决整体校准。

**不能靠忽略一切证据通过测试：** n=0且probe informative时，particle256 query RMSE为{groups[1,0,True]['particle256']['rmse']:.6f}/{groups[2,0,True]['particle256']['rmse']:.6f}，不更新为{groups[1,0,True]['no_update']['rmse']:.6f}/{groups[2,0,True]['no_update']['rmse']:.6f}。部分已有历史条件下，原生更新即使probe不含alpha信息，也会改善其原本不完美的分布；因此不能把低tower误差与真实质量直接画等号。

**全部条件joint NLL（越低越好）：** 每行先在固定系统上聚合，再平均原训练seed；particle另平均3个MC实现。不是将重复系统当作新的独立样本。完整RMSE/coverage/variance及所有粒子预算见outputs/analysis/full_results.csv。

{table}

**数值与独立复核：** 原128-node GH的部分误差超预设门槛，保留原文件后改513-point Normal grid，1025/8193复核最大误差0.0001267 prior std。EKF原7sigma密度尾部不足，固定后验参数后扩大积分范围并验证。更强全64context尾界不能认证的58行做稀疏数值修复，不删除困难样本、不改模型/后验参数/科学比较。主复核120数组、2880聚合指标差{mv['aggregate_max_error']:.3g}，原生重放24case差{mv['native_replay_max_error']:.3g}；普通对照240数组、4224聚合项差{ov['aggregate_max_error']:.3g}，importance代数差{ov['importance_algebra_max_error']:.3g}；58patch全项复算差{tv['score_max_error']:.3g}，最大refinement误差{tv['max_refinement_nll_error']:.3g}。完整边界/覆盖范围见三个receipt。

**成本与限制：** 主运行{M['elapsed_seconds']:.2f}s，初始积分修复{M['integration_repair_elapsed_seconds']:.2f}s，粒子对照{P['elapsed_seconds']:.2f}s。4096粒子、batch64的更新中位耗时{1000*timing:.2f}ms，含提议编码器/采样/probe likelihood，排除共同prior encoder和离线query积分；不推断256粒子的单样本延迟或大型基础模型速度。MAP-GGN最差距已知grid mode约0.335objective、归一化梯度约1.11，不能将其局部优化失败当成所有MAP方法不行。当前仅老公开模型的toy机制实验；没有真实设备或现代大模型的同类缺口证明。
''',
'negative_evidence.md':'''# 负面证据和被否定的扩大解释

1. “只是未来归一化/坐标错误”：本轮使用独立原生因果核，连续前缀、固定query初值与全范围来源校验；新failure仍出现。与cycle23那份WH生成器问题分开。
2. “只是模型latent坐标变化”：真实物理query联合NLL和覆盖率确实恶化；不是只看latent entropy或latent mean。
3. “已证实整个时间序列world-model领域都不能更新证据”：没有。只测一个公开方法、两个actuation变体、三个checkpoint seed，不能外推全领域。
4. “通过不更新就解决了问题”：不更新对informative probe明显差，需要同时检查信息利用和无信息保护。
5. “一致性更高就必然更好”：普通完整Bayes并不在所有真实条件上优于原生更新；decoder/likelihood有模型误差，摊销更新可能补偿它。一致性与truthfulness需分开验证。
6. “必须另造新架构”：普通defensive importance校正+Gaussian矩投影已经修复大部分冷启动损害。generic Bayesian correction不是创新；本轮没有证明必要的新结构。
7. “EKF/MAP的不足说明老推断都无效”：不成立。EKF是局部线性近似；MAP-GGN还存在局部优化/单模态限制。更完整粒子对照已给出相反证据。
8. “坏NLL就是科学失败”：原GH积分、EKF尾部及少数全量尾界问题均先排查并保留原尝试，修复数值后才生成表。独立verifier的float32 log(scale)导致ESS差异也只修复verifier dtype，没有放宽阈值或改producer。

未证明的部分：高维非线性系统、其他模型、长时联合模拟、部分可观测条件下同一失败是否重要，及已有高质量推断是否仍不足。不得用扩大模型/数据规模维持一个已被普通方法解决的论点。
''',
'hypothesis_space.md':'''# 竞争解释与实证状态

| 解释 | 区分证据 | 状态 |
|---|---|---|
| 未来数据/状态初值/随机漏历史污染 | 原因果源、完整前缀、独立query/probe reset | 排除本轮已知混淆 |
| 数值积分伪影 | uniform grid/refinement、全量尾界、SciPy/native重放 | 主结果通过，原失败尝试保留 |
| q encoder更新不等于其decoder联合律的条件化 | model-generated probe tower +同decoder精确重加权 | 支持，且有真实冷启动后果 |
| 普通likelihood/decoder也有误差 | 完整Bayes未全条件优于native；physical oracle更好 | 支持，不能只优化coherence |
| 只有复杂新模型才能修复 | defensive importance +Gaussian矩投影 | 本toy反证：旧方法已修复大部分 |
| 缺陷普遍且新框架必要 | 新现代模型/高维系统/复杂信息结构比较 | 尚未执行、不能宣称 |

原生先验的不足、后验近似误差和数值误差是不同解释。不能将后两者排除后就默认产生新的研究方向；还要证明问题未被已有方法充分解决，并明确实际用途。
''',
'research_ideas.md':'''# 当前候选与创新门槛

优先工作问题：时间序列世界模型加入新观测时，能否同时保持物理预测分布的一致性、避免无信息证据造成过度自信，并利用有信息证据？

现在已从文字假设推进到一个可重现的失败和已知修复：六个公开checkpoint的无信息冷启动更新使概率预测恶化；普通粒子校正能大幅修复。问题的存在、后果与可修复性在这个窄scope成立；“尚未被现有方法解决”及创新边界尚未成立。

不能将importance weighting、ADF、MAP/Laplace或一个consistency loss改名为新架构。也不能从旧toy推进到更多toy，仅为了累积失败数量。下一步应先检查另一个实质不同的模型/信息结构，使用强且合理预算的已知推断，再决定这是一项有价值的评测问题还是需要新机制。

一个可判别的后续是原生16维系统latent的非线性pendulum，比较发布的variance-constrained与unconstrained检查点，并明确只在源码证明对未知参数确实无信息的控制条件上测试；不能把所有环境的u=0都当作no-information（例如terrain未知的mountaincar不满足）。需核查观测噪声、action repeat、reset和初态可用性，安排新cohort。该后续尚未执行。

如果已有full-covariance/particle方法仍足够，停止“新的更新架构”论点。评测研究是否有发表价值仍取决于多模型/现实机制/新指标边界证据。当前最终接受方向0；保留这条可靠性候选作为待反证的问题，不包装成已确定投稿idea。
''',
'final_research_direction.md':'''# 当前决定：有实测问题，尚无最终创新方向

**Research Question：** 通用时间序列世界模型如何在新证据下更新同一未知系统的物理预测分布，同时避免无信息证据造成过度自信并保持有效的信息利用？

**Evidence：** 六个原生发布checkpoint、128个新物理系统实现、48个固定条件；冷启动no-information更新的真实joint NLL全部恶化，context bootstrap区间均高于0，边际覆盖率下降。模型生成证据下的物理moment偏差及完整Bayes对照提供机制归因。数值与独立源码重放通过。

**Existing Methods：** 当前不能证明它们不足。普通defensive importance校正+Gaussian矩投影已修复大部分冷启动损害；完整Bayes、EKF、MAP-GGN也被明确比较。通用NP/martingale、主动辨识和MAP/Laplace均有直接先例。

**Proposed Idea：** 尚不提出或接受新架构。当前候选首先是一项“证据更新可靠性”的问题/评测方向；已知推断作为competent controls，不能重命名为创新。需要进一步确定其高维/复杂机制下是否有现有方法无法兼顾的质量与成本缺口。

**Falsification：** 如果原生其他模型已满足要求，或合理预算的旧推断足够，或失败只来自单一旧toy/近似优化/数值/分布支持，则放弃新的框架主张。若改进不保留informative evidence的价值，同样不成立。

**Feasibility：** 单RTX5090已完成本轮，六模型冻结无重训；原始数据、脚本、配置、checkpoint/source SHA、日志与三类独立复核在服务器，当前表和图在本机镜像。下一步可以复用已取得的16维latent发布权重，无需盲目启动大型预训练。

**Next Experiment：** 在新的非线性高维系统及独立cohort中，先验证“no-information”的物理定义，再比较native constrained/unconstrained、full-covariance/defensive-particle对照及真实预测后果。若旧方法已充分修复，结束架构路线；不通过调旧测试集制造新idea。

最终正式推荐方向：0。当前优先工作问题为上面的证据更新可靠性，创新性与顶会潜力未获充分验证。总体目标ACTIVE、UNACHIEVED；本轮科学结论不是任务完成声明。
'''}
for name,text in reports.items():(ROOT/name).write_text(text,encoding='utf-8')
# Two panels: per-checkpoint actual failure, then probabilistic quality after
# known corrections; no invented sampling independence or confidence intervals.
fig,(a,b)=plt.subplots(1,2,figsize=(11.4,4.8),layout='constrained')
yy=np.arange(6);val=np.array([x['native_delta'] for x in cold]);lo=np.array([x['ci'][0] for x in cold]);hi=np.array([x['ci'][1] for x in cold])
a.errorbar(val,yy,xerr=np.vstack((val-lo,hi-val)),fmt='o',color='#be4b39',capsize=3,label='Native update (context bootstrap95%)')
a.scatter([x['particle256_delta'] for x in cold],yy,marker='s',s=35,color='#3477a5',label='Known particle256 (MC mean)')
a.axvline(0,color='#555',linewidth=.8);a.set_yticks(yy,[f"Variant{x['kind']} / seed{x['seed']}" for x in cold]);a.invert_yaxis();a.set_xlabel('Joint NLL change after an uninformative probe\n(positive is worse)');a.set_title('A  Cold start: six frozen checkpoints',loc='left',fontsize=12);a.legend(fontsize=8,frameon=False,loc='upper right')
methods=['no_update','native','reweighted','particle256','physical_oracle'];labels=['No update','Native','Dense Bayes','Particle256','Physical oracle']
for j,kind in enumerate([1,2]):
    cover=[100*groups[kind,0,False][m]['marginal_coverage95'] for m in methods]
    b.plot(np.arange(5),cover,marker='o',label=f'Variant{kind}',color=['#3265a8','#d48a34'][j])
b.axhline(95,color='#666',linestyle='--',linewidth=.8,label='Nominal95%');b.set_xticks(np.arange(5),labels,rotation=22,ha='right');b.set_ylim(70,100);b.set_ylabel('Marginal coverage across two state outputs (%)');b.set_title('B  Known inference restores prior coverage',loc='left',fontsize=12);b.legend(frameon=False,fontsize=9,loc='lower left')
for ax in [a,b]:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax==a else 'y',alpha=.16);ax.set_axisbelow(True)
fig.text(.01,-.065,'One toy plant, two actuation maps. Fixed published models; particle controls are post-hoc known methods. No new-architecture claim.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'belief_update_audit.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(dict(verified_bundle_sha256=hashlib.sha256(FILE.read_bytes()).hexdigest(),cold_rows=cold,mean_native_damage=average_native,mean_particle256_residual=average_particle,fraction_damage_removed=fraction,median_4096_batch64_update_seconds=timing,accepted_novel_directions=0),indent=2),encoding='utf-8')
print(json.dumps(dict(reports=7,plots=3,full_result_rows=len(rows),mean_native_damage=average_native,mean_particle256_residual=average_particle,removed=fraction,accepted_directions=0)))
