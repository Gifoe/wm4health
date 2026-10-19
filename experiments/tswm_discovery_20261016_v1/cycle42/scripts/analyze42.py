"""Render all effects and preserve ordinary-model/novelty distinction."""
import sys,pathlib,json,hashlib,datetime,csv,collections
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=R/'outputs';F=O/'fits_v1';A=O/'analysis_v1';assert not A.exists();A.mkdir()
load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=load(O/'verification_v1.json');result=load(F/'results.json');assert v['status']==result['status']=='PASS' and v['results_sha256']==sha(F/'results.json')
rows=result['rows'];groups=collections.defaultdict(list)
for r in rows:groups[(r['n_per_seen_cell'],r['group'],r['obs_mod'],r['dyn_mod'])].append(r)
agg=[]
for key,values in sorted(groups.items()):
 row=dict(zip(['n_per_seen_cell','group','obs_mod','dyn_mod'],key));row['fits']=len(values)
 for metric in ['future_nll','excess_nll','next_symbol_oracle_kl']:
  vals=[x[metric] for x in values];row[metric]=float(np.mean(vals));row[metric+'_range']=[min(vals),max(vals)]
 agg.append(row)
interactions=[]
for w in range(2):
 for n in [16,128]:
  for seed in [2026421,2026422]:
   for group in ['seen','heldout']:
    rr={(r['obs_mod'],r['dyn_mod']):r['future_nll'] for r in rows if r['world']==w and r['n_per_seen_cell']==n and r['init_seed']==seed and r['group']==group}
    interactions.append({'world':w,'n':n,'init_seed':seed,'group':group,'I':rr[1,0]+rr[0,1]-rr[1,1]-rr[0,0]})
fitrows=[r for r in rows if r['group']=='heldout'];selected_cap=sum(r['selected_step']==1500 for r in fitrows);last_improved=0
for folder in F.glob('w*_n*_i*_o*d*'):
 curve=load(folder/'curves.json');last_improved+=curve[-1]['seen_val_future_nll']<curve[-2]['seen_val_future_nll']
out={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_fits':32,'formal_updates':48000,'new_method_fits':0,'ordinary_structured_HMM_is_candidate_11':True,'aggregate':agg,'interactions':interactions,'selected_cap':selected_cap,'last_interval_improved':last_improved,'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED','synthetic_evaluation_scored':True,'old_real_test_scored':False,'verification_sha256':sha(O/'verification_v1.json')}
(A/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
with (A/'all_results.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (A/'interactions.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=list(interactions[0]));writer.writeheader();writer.writerows(interactions)

fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
combos=[(0,0),(1,0),(0,1),(1,1)];colors=['#777777','#0072B2','#D55E00','#009E73']
for i,n in enumerate([16,128]):
 for j,group in enumerate(['seen','heldout']):
  ax=axes[i,j]
  for k,(om,dm) in enumerate(combos):
   rr=groups[n,group,om,dm];ys=[r['excess_nll'] for r in rr]
   ax.bar(k,np.mean(ys),width=.6,color=colors[k],alpha=.6)
   ax.scatter(k+np.linspace(-.14,.14,4),ys,color=colors[k],edgecolor='black',s=24,zorder=3)
  ax.axhline(0,color='black',lw=.8);ax.set_title(f'{group}: {n} train sequences / seen cell');ax.set_ylabel('Future joint NLL / symbol minus oracle');ax.set_xticks(range(4),['00','10: view','01: dynamics','11: both']);ax.grid(axis='y',alpha=.25)
fig.suptitle('Ordinary learned HMMs: 2 worlds × 2 initializations; synthetic screening only',fontsize=12)
fig.tight_layout();fig.savefig(A/'factorization.png',dpi=160);fig.savefig(A/'factorization.pdf');plt.close(fig)

table=['|N / seen cell|evaluation group|model|future NLL/symbol|excess vs oracle|next-symbol oracle KL|','|---:|---|---|---:|---:|---:|']
for row in agg:table.append(f"|{row['n_per_seen_cell']}|{row['group']}|{row['obs_mod']}{row['dyn_mod']}|{row['future_nll']:.6f}|{row['excess_nll']:.6f}|{row['next_symbol_oracle_kl']:.6f}|")
itable=['|N|group|mean I|range over 2 worlds × 2 initializations|','|---:|---|---:|---:|']
for n in [16,128]:
 for group in ['seen','heldout']:
  values=[r['I'] for r in interactions if r['n']==n and r['group']==group];itable.append(f'|{n}|{group}|{np.mean(values):.6f}|[{min(values):.6f}, {max(values):.6f}]|')
main='\n'.join(table);ints='\n'.join(itable)
summary=f'''服务器完成32个未知参数ordinary conditional HMM fits /48000正式updates；独立NumPy从selected checkpoint重建全部模型，核对{v['raw_sequence_scores']}条未来序列score和{v['next_distribution_cells']}个next-probability cells，max NLL差{v['max_sequence_nll_abs']:.3g}。此前真实TEST未评分；本轮人工held-out evaluation按冻结协议评分。最终接受方向0，整体ACTIVE_UNACHIEVED。'''
reports={
'literature_review.md':f'''# C42 文献与创新边界

{summary}

本轮不是大范围新文献survey，沿用[C41综合](../cycle41/literature_review.md)。NCDSSM的观测证据/动态分离、modular NRI的动态复用、LIFTS的联合生成与推断均已存在。新查[Compositional Interfaces for Compositional Generalization, CoLLAs2025](https://proceedings.mlr.press/v274/luketina25a.html)官方abstract也已包含模块化observation接口、共享controller、未见组合和new-modality适应。检查层级为官方abstract；没有全文/代码复现。详见[边界](PRIOR_ART_BOUNDARY.md)。

这些近邻不自动否定一个实证充分的新交互，但足以否定“把观测和动态拆开就是新框架”。本轮11的概率模型就是ordinary structured HMM。没有新的预测状态理论、额外loss或新算子。
''',
'baseline_registry.md':f'''# C42 Baseline registry

{summary}

四模型00/10/01/11均为本项目自主设计的ordinary neural-conditioned four-state HMM，不称原论文复现。00两conditioners均依赖view+system；10将emission的system输入置0；01将transition的view输入置0；11同时置0。每组名义1632parameters，11有效1376，单侧1504；零梯度输入列差异明确披露，不能宣称完整capacity match。exact forward filtering和joint maximum likelihood相同。

普通11也是候选初始离散形式，无独立new-method fit。Oracle只作privileged known-parameter reference，不是未知参数的公平竞争者。NCDSSM/modular NRI/LIFTS/FlowNP/GRU完整跨框架对比尚未在本轮执行。

两independent world、N=16/128、两个initializations，1500updates/fit，batch32，lr.01→.001。每250步以seen VAL选择，未用held-out combination选模。选中cap {selected_cap}/32，最后interval VAL仍改善{last_improved}/32；不声称收敛。
''',
'experimental_results.md':f'''# C42 Actual experimental results

{summary}

24step sequences，prefix8/future16；8seen temporal cells/8held-out cells；每个world另32组四view同步无label静态anchors，system0下给所有模型相同信息。测试目标是有anchor的未见**动态组合**；不能声称完全没见该view–system的任何观测。原理、split和信息合同见[PROTOCOL](PROTOCOL.md)。

{main}

每行均为4fits均值，不是四个独立数据集。完整raw rows见outputs/analysis_v1/all_results.csv。

交互 I=R10+R01−R11−R00，正值表示分离观测时动态factorization收益更大：

{ints}

只报告descriptive range，不以两个world做可靠总体显著性判断。11最好、正交互或oracle excess下降均不足证明新方法或现实research gap。

![all initializations](outputs/analysis_v1/factorization.png)

精确likelihood枚举max8.88e-16、anchor枚举0、梯度finite-diff约1.41e-11；CPU8.0ms/update快于GPU17.9ms，资格权重丢弃。正式用CPU，其他GPU任务不动。独立验证结果见outputs/verification_v1.json，最终验证覆盖全checkpoint/score而不是复述训练日志。
''',
'negative_evidence.md':f'''# C42 Negative evidence and rejected claims

{summary}

1. **拒绝初始架构新颖性主张**：11等于ordinary structured HMM；不能把同一模型另命名后报告超越ordinary baseline。
2. 00/10/01若有失败，只能归因到这些具体conditional parameterizations与训练预算。其输入有同样信息，未见cell由shared MLP生成而非未训练lookup，但这不代表所有单体world model。
3. 人工worlds按factorization生成，本轮只能筛选可学习性及known-component inductive bias，不能说明真实系统普遍存在同样可组合机制。
4. 只2world、2initializations；名义参数一致，有效参数有差；{selected_cap}/32选cap，{last_improved}/32末段仍改善。未完成计算/容量/优化充分性及跨现实域强基线闭环。
5. 强known-parameter oracle与learned方法信息不同，只作truth reference；不以其差距直接命名representation failure。

没有通过改变noise、训练cell或seed挽救正面故事。原C41的ordinary正确dynamics/充分anchor控制结论继续有效。
''',
'hypothesis_space.md':f'''# C42 Competing hypotheses

{summary}

- H1：观测/动态分离只是ordinary结构先验。预测11可学习，甚至最好，但没有独立新机制；本轮11概率模型与候选相同，结构层面已成立。
- H2：两侧共享有额外有限数据交互。以预注册I及其全部world/init范围检查；正均值不足总体证明，见实际表。
- H3：差异由optimization/参数量解释。需要强ordinary joint SSM足够优化和active-capacity/compute controls，不能从本轮完全排除；保留cap/curve计数。
- H4：任务信息不足。C41给出了对称性反例；本轮用asymmetric transitions及同信息anchors，并用known-law oracle和可学习ordinary模型区分。未知sensor任意泛化不作为目标。

下一步不得以latent坐标对齐替代observable-law指标，或者将弱conditioner差距说成整个领域失败。
''',
'research_ideas.md':f'''# C42 Candidate status

{summary}

候选A的最初离散形式已经明确是ordinary因子化模型。该组合复用**研究目标**可以保留，但尚缺少超出已有结构建模的具体必要交互；不批准为最终新idea。若后续仅扩宽MLP/增加latent loss/换数据集，不足产生新颖性。

候选B（共享coarse law×MLMC）保持备选，无本轮效果；mr-Diff/WaveletDiff/generative MLMC近邻已存在，不能直接声称新。最多保留这两个调查方向，当前没有被实验证明的最终推荐。

WorldTS是贡献范围参考：同一框架可复用的预测状态/动态能力与多任务证据。单个HMM人工集合的优化不作为成果终点。
''',
'final_research_direction.md':f'''# C42 Research decision — framework objective retained, no accepted new method

**Research Question**：观测接口与动态机制分别学习和复用，能否改善未见组合的有限数据推演；进一步是否存在超出普通结构化SSM的必要新机制？

**Evidence**：{summary} 真实数值与2×2交互完整见[results](experimental_results.md)。它们只回答受控人工环境的ordinary可学习性，不证明领域缺口。

**Existing Methods**：普通jointly fitted factorized HMM就是11；NCDSSM、modular NRI、LIFTS与CoLLAs compositional interfaces覆盖多项现有组件。目前不能断言它们共同不足。

**Proposed Idea**：当前初始参数化属于已知结构模型，本轮没有合格的新方法。允许在研究目标不变下继续发现更具体的必要交互；不以新增模块掩盖等价性。

**Falsification**：最强ordinary模型吸收全部收益，或者交互只在人工先验、特定seed和不充分优化下出现，即否决对应创新/普适性主张。新框架必须给出区别于现有模型的可证伪机制与实际用途。

**Feasibility**：本轮小型exact模型服务器CPU更快，32fits已完成，RTX5090保留给需要的神经/跨域验证。已有native LIFTS/FlowNP资格和原有数据证据可复用，无需重新下载大库。

**Next Experiment**：先用完整最近方法与代码厘清尚未覆盖的框架能力，提出区别于ordinary11的单一机制假设，再冻结一个同信息/同预算的决定性test；若无此假设，停止扩展当前合成2×2。不以继续加数据集替代新机制发现。

最终方向未达到接受标准；整体ACTIVE_UNACHIEVED。
''',
'CURRENT_STATE.md':f'''# C42 — bounded screen completed, overall goal active

{summary}

未知参数观测/动态2×2 ordinary HMM已真实学习；选模只用seen VAL，人工heldout评分后未调参。所有raw scores/selected checkpoints独立重建。11本身就是ordinary structured HMM，当前不能把它当新框架贡献。结果、交互、优化限制见[experimental_results](experimental_results.md)。当前接受最终方向0；后续须发现区别于普通模型的具体机制，或换到更有证据的研究问题。
'''}
initial=R/'CURRENT_STATE.md'
if initial.exists():(R/'CURRENT_STATE_RUNNING.md').write_bytes(initial.read_bytes())
for name,body in reports.items():(R/name).write_text(body,encoding='utf-8')
manifest={'reports':{name:sha(R/name) for name in reports},'analysis_sha256':sha(A/'analysis.json'),'verification_sha256':sha(O/'verification_v1.json'),'results_sha256':sha(F/'results.json'),'protocol_sha256':sha(R/'PROTOCOL.md')}
(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({k:out[k] for k in ['status','formal_fits','formal_updates','selected_cap','last_interval_improved','accepted_final_directions','goal']}),flush=True)
