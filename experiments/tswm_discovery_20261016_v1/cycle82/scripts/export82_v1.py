import sys,pathlib,hashlib,csv,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common82_v1 import *
import numpy as np
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def write(name,value):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(value)
def main():
 r=read(OUT/'RESULTS82_V1.json');v=read(OUT/'VERIFICATION82_V1.json');assert v['status'].startswith('PASS_') and read(OUT/'run82_v1_exit.json')['exit_code']==0;kinds=['FIM_local2','FIM_full3_diagnostic','weak_polynomial3','weak_rff128'];lookup={(z['mechanism'],z['seed'],z['regime'],z['kind']):z for z in r['fields']};summaries=[];pairs=[]
 for kind in kinds:
  closed=[lookup[m,s,'closed',kind]['nrmse'] for m in range(4) for s in [2026821,2026822]];opened=[lookup[m,s,'open',kind]['nrmse'] for m in range(4) for s in [2026821,2026822]];ratio=[b/a for a,b in zip(closed,opened)];sim=[z['nrmse'] for z in r['rollouts'] if z['kind']==kind and z['finite_complete'] and z['solver']['success']];summaries.append(dict(kind=kind,closed_field_median=float(np.median(closed)),open_field_median=float(np.median(opened)),median_paired_open_closed_ratio=float(np.median(ratio)),pairs_at_least_double=int(sum(z>=2 for z in ratio)),controlled_rollout_median=float(np.median(sim)) if sim else None,successful_controlled_rollouts=len(sim),expected_controlled_rollouts=32))
 local=summaries[0];full=summaries[1];ordinary_ok=any(z['open_field_median']<=.15 and z['median_paired_open_closed_ratio']<=1.5 for z in summaries[2:]);passed=local['closed_field_median']<=.3 and local['pairs_at_least_double']>=6 and full['open_field_median']<=.3 and ordinary_ok;decision='CLOSURE_SIGNATURE_REQUIRES_MATCHED_GEOMETRY_FOLLOWUP' if passed else 'SIMPLE_CLOSURE_FAILURE_HYPOTHESIS_NOT_SUPPORTED'
 dump(OUT/'DECISION82_V1.json',dict(utc=utc(),decision=decision,frozen_gate_passed=bool(passed),summaries=summaries,full_state_has_richer_information=True,clean_polynomial_development_carrier_only=True,accepted_final_research_directions=0,overall_goal='ACTIVE_UNACHIEVED'))
 table='| Method | Closed field median | Open field median | Paired open/closed median | At least 2x worse | Controlled-rollout median | Complete rollouts |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for s in summaries:table+=f"| {s['kind']} | {s['closed_field_median']:.8f} | {s['open_field_median']:.8f} | {s['median_paired_open_closed_ratio']:.6f} | {s['pairs_at_least_double']}/8 | {s['controlled_rollout_median']:.8f} | {s['successful_controlled_rollouts']}/32 |\n"
 cells='| Mechanism | Initial-state bank | Regime | FIM local2 | FIM full3 diagnostic | Weak polynomial3 | Weak RFF128 |\n|---|---|---|---:|---:|---:|---:|\n'
 for m in range(4):
  for seed in [2026821,2026822]:
   for regime in ['closed','open']:cells+=f'| {m} | {seed} | {regime} | '+' | '.join(f"{lookup[m,seed,regime,kind]['nrmse']:.8f}" for kind in kinds)+' |\n'
 label='局部开放性在当前受控载体中产生了预定的退化信号，需要更严格的几何和先验对照。' if passed else '“局部不闭合就会使预训练场推断失效”没有得到当前实验支持。不能继续把这个解释当作既定研究缺口。'
 report=f'''# C82：联合动力学先验能否复用为局部机制？

{label} 本轮接受最终方向0。已有模型的正面能力和强普通控制均完整保留。

## 研究问题与实际合同

低维FIM-ODE以闭合自主系统为原任务。更大系统里的局部邻域可能受外部变量驱动，但目标分量仍只依赖已观测的直接父变量。我们检查这两种条件是否必须同时满足。保持同一个目标f0(x0,x1)，只将邻居x1的外部驱动系数从0改成1.2。完整生成系统有三个状态，目标一阶导数严格不依赖第三状态。

这是四个相关多项式机制×两个初态bank×两种边界条件，共16个上下文任务，不是16个独立真实数据集。两种bank在不同机制间共用初态；没有用结果筛选方程或seed。每任务8条干净轨迹、每条256点、t∈[0,8]。先冻结协议再生成/推断。原预算C71和C81保留评估均未打开。

同一机制/bank的开放与闭合案例使用完全相同的256个评分查询。查询处在两个上下文点云的凸包内，且经过合并TRAIN坐标标准化后，距两侧点云都不超过0.1。此条件减少支持范围差异，但不等于密度匹配或一般可辨识性证明。查询真导数仅供评分。全部生成器和文件哈希保留。

四组固定方法：原生FIM局部2变量、原生FIM完整3变量、目标分量弱三次多项式和目标分量弱RFF128。完整3变量模型得到更多上下文，属于诊断组，不能凭它优于局部模型就归因。普通模型只拟合目标导数，不拟合邻居的闭合动力学；它们与局部FIM获得相同两变量上下文。

多项式库包含真实目标族，明确为source-informed参数辨识控制；RFF是另一个固定灵活控制。它们都不是新方法。每族3个ridge，以前192点拟合、最后65个可见上下文点作给定x1输入的中心状态轨迹验证，再用全部256点重拟合。共96个候选、32次最终弱模型拟合；每候选检查8条验证路径，共768次内部验证积分。没有神经训练，没有把真场或独立新输入计划作为选择信号。

## 全部汇总

Field NRMSE=目标导数RMSE/同组真实导数标准差。控制轨迹NRMSE用该参考轨迹的标准差，均只用于评分。两个量不能直接混为同一指标。

{table}

## 每个场结果

{cells}

## 实际受控模拟

每个模型还在两条新的正弦/余弦输入计划下模拟中心变量，共128条结果。输入计划与初值在协议中预定；它们是do(x1(t)=u(t))条件，不是未来观测特征。完整3变量诊断在查询时固定x2=0，真实目标机制对它不敏感。RK45容差和失败保护对所有方法相同，参考用高精度DOP853。没有只报告导数拟合而省略实际模拟。

这仍是局部受控原语模拟。未运行高维自由网络、未知图发现、隐状态恢复、未知传感器转移，或对该系统做真实实验。不能把这里的2变量局部成功升级成一般高维世界模型已实现。

![Field and controlled simulation](figures/closure82_v1.png)

## 独立核验

原生2D/3D缓存与直接model_forward在17个查询完全一致；显式零补齐第三坐标与原生2D输入完全一致。32个原生场结果独立重放最大差0；全部64场记录、128控制轨迹的指标独立计算最大差{v['score_max_abs']:.3g}，共检查{v['prediction_target_scalars']}个预测/参考标量和{v['identity_arrays_checked']}个身份数组。32个弱模型共2224个系数以逐窗口积分和augmented least squares复核，最大差{max(z['coefficient_max_abs'] for z in v['independent_coefficient_checks']):.3g}。没有失败记录被丢弃。源70文件和权重结束后再次核验。

实际数值执行耗时{r['wall_seconds']:.2f}秒，CPU两线程、进程局部E-core affinity；固定OpenFIM commit `cee2bb53be1e8e92b73f98dfdbe3605efc9119cc`和HF基础权重revision `00875c253d025dd33fd70ae751089b1be3a7c174`，12,968,068参数未更新。数据和数组留在服务器15TB项目子目录，GitHub只上传代码、协议、紧凑结果、模型系数、日志、图和哈希；未同步本地载荷。

## 研究判断与下一项决定性问题

决策：`{decision}`。C82并没有因C81失败就继续改同一模型；它检验了另一种可复用动力学能力。闭合先验可否作局部机制的答案应来自这些实际结果，而不是从任务假设直接推导失败。

即使局部推断成功，它与可扩展世界模型之间仍缺少证据：拼接很多局部场后，完整系统的误差、反馈稳定性、未见初态和干预传播是否保持？更大系统中，已知父图、样本数和局部模型的先验成本必须同强普通辨识控制对齐。此前CompNO/ICON和函数编码器已经覆盖模块组合，不能把简单拼接改名当新框架。当前结果只为下一次真正的高维组合实验提供资格化依据；没有完成创新性和独立改进空间验证。
'''
 write('LOCAL_MECHANISM_AUDIT82_V1.md',report)
 for name,body in {
 'literature_review.md':'See LITERATURE_BOUNDARIES82_V1.md. The selected target differs from generic modularity, but no new mechanism is established. FIM-ODE assumes autonomous observed state; CompNO/ICON, function encoders and input-output identification are direct competitors. Counterfactual coupling and observation transfer alternatives remain explicitly qualified by prior art and missing evidence.',
 'baseline_registry.md':'Executed: fixed released13M FIM with local2/full3 contexts; ordinary target-only polynomial3 and RFF128 weak ridge. Sixteen controlled development tasks,96 candidate fits,32 full weak refits,64 field outcomes,128 actual controlled rollouts. Full3 is a richer-information diagnostic; polynomial library is source-informed. No new neural fit or original-paper table reproduction.',
 'experimental_results.md':table+'\nAll field cases, candidate coefficients and every rollout are in outputs/RESULTS82_V1.json and the complete audit. No filtered success-only comparison is used.',
 'negative_evidence.md':label+'\nGeneric closure failure is not inferred from an open neighborhood alone. Ordinary controls, actual local FIM successes and all paired results are retained. Four related polynomial mechanisms with clean observations do not settle general hidden-state or high-dimensional inference.',
 'hypothesis_space.md':f'H1 numerical/API mismatch: direct/cached/zero-padding qualification and independent field replay pass. H2 local mechanism cannot be identified: source-informed weak controls test this explicitly. H3 closed-system pretraining necessarily fails under open neighbor dynamics: frozen gate result {decision}. H4 existing component estimates can compose into a general world model: untested; requires an actual larger-system and intervention experiment.',
 'research_ideas.md':'No qualified final direction. Candidate for further development: reusing a released low-dimensional inference prior as local target mechanisms in a larger controlled system. C82 qualifies only the local primitive. Existing composition methods, source-informed weak fits and arbitrary-input identification are strong overlap/ordinary alternatives. Nonlinear unknown sensor transfer remains needs-mechanism; generic cross-world coupling is already directly studied.',
 'final_research_direction.md':f'Research Question: can low-dimensional pretrained dynamics be reused as local mechanisms and ultimately composed? Evidence: current screen16 tasks,64 field results,128 controlled rollouts; decision {decision}. Existing Methods: CompNO/ICON, function encoders, ICL-SysID and target-only weak/kernel identification cover important parts. Proposed Idea: no accepted new mechanism; broad local-composition route remains provisional. Falsification: ordinary matched local models explain the capability/cost, or composed dynamics fail without a differentiated repair. Feasibility: qualified13M checkpoint and server-only generator, single5090/CPU. Next Experiment: actual higher-dimensional composed-system free simulation and interventions, with known-parent and data-budget contracts frozen; no new local-failure claim. Accepted final directions0; overall ACTIVE/UNACHIEVED.'}.items():write(name,'# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n')
 for label,rows in [('field',r['fields']),('rollout',r['rollouts'])]:
  keys=['tag','mechanism','seed','regime','kind','finite_complete','mse','rmse','mae','nrmse']+(['plan_index','omega','initial_state'] if label=='rollout' else ['information'])
  with (OUT/f'{label}_metrics82_v1.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:z.get(k) for k in keys} for z in rows])
 fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');colors=['#0072B2','#333333','#009E73','#D55E00'];names=['FIM local2','FIM full3 (more input)','Weak polynomial','Weak RFF128']
 for j,(kind,color,name) in enumerate(zip(kinds,colors,names)):
  for regime,marker in [('closed','o'),('open','x')]:
   fieldvalues=[z['nrmse'] for z in r['fields'] if z['kind']==kind and z['regime']==regime];sim=[z['nrmse'] for z in r['rollouts'] if z['kind']==kind and z['regime']==regime and z['finite_complete'] and z['solver']['success']];offset=-.12 if regime=='closed' else .12;axes[0].scatter(np.full(len(fieldvalues),j+offset),fieldvalues,color=color,marker=marker,alpha=.65,label=regime if j==0 else None);axes[1].scatter(np.full(len(sim),j+offset),sim,color=color,marker=marker,alpha=.55)
 for ax,title in zip(axes,['Central vector field','Actual controlled trajectories']):ax.set_yscale('log');ax.set_xticks(range(4),names,rotation=18,ha='right',fontsize=8);ax.set_ylabel('NRMSE (log scale)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
 axes[0].legend(title='Boundary regime',fontsize=8);figure=storage('artifact')/'closure82_v1.png';fig.savefig(figure,dpi=170);plt.close(fig)
 dump(OUT/'CLOSURE82_V1.json',dict(utc=utc(),decision=decision,results_sha256=sha(OUT/'RESULTS82_V1.json'),verification_sha256=sha(OUT/'VERIFICATION82_V1.json'),decision_sha256=sha(OUT/'DECISION82_V1.json'),figure_path=str(figure),figure_sha256=sha(figure),field_records=64,rollout_records=128,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0))
 with (PROJECT/'CURRENT_RESEARCH_STATUS_V82.md').open('x',encoding='utf-8') as f:f.write('# C82 current research status\n\n'+label+'\n\n'+table+f'\nDecision: {decision}. Qualified local inference is not yet a high-dimensional world model. No final research direction accepted. The prior original-budget GPU campaign remains separate; no early reserved evaluation is used here. All raw payloads remain on server storage, GitHub publication only.\n')
 files=['PROTOCOL82_V1.md','RESEARCH_BRIEF82_V1.md','LITERATURE_BOUNDARIES82_V1.md','LOCAL_MECHANISM_AUDIT82_V1.md','literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md'];files+=['scripts/'+p.name for p in sorted((ROOT/'scripts').glob('*.py'))];files+=['outputs/'+n for n in ['DATA_OPENING82_V1.json','DATA_MANIFEST82_V1.json','OPENING82_V1.json','RESULTS82_V1.json','VERIFICATION82_V1.json','DECISION82_V1.json','CLOSURE82_V1.json','field_metrics82_v1.csv','rollout_metrics82_v1.csv','run82_v1_launch.json','run82_v1_exit.json','run82_v1_stdout.log']];dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);pub=[]
 for rel in sorted(files):
  b=(ROOT/rel).read_text(encoding='utf-8-sig').encode();assert all(z.encode() not in b for z in json.loads(os.environ.get('C82_PRIVATE_EXPORT_PATTERNS_JSON','[]'))),(rel,'private export pattern');p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='utf-8'))
 b=figure.read_bytes();rel='figures/closure82_v1.png';p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='binary'))
 dump(dest/'manifest.json',dict(utc=utc(),files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),large_data_weights_arrays_included=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0));print(json.dumps(dict(status='SNAPSHOT_READY',decision=decision,summaries=summaries,manifest_sha256=sha(dest/'manifest.json'),files=len(pub))),flush=True)
if __name__=='__main__':main()
