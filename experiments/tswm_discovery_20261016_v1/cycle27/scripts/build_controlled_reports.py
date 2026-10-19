"""Derive all-setting summaries and reports only from independently verified arrays."""
import runtime_profile
import collections,csv,hashlib,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/controlled_em_study';DEST=ROOT/'outputs/analysis'
METRICS=['mse_observed','mse_clean','relative_l2_clean','marginal_nll','coverage95','mean_predictive_std','finite_fraction']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def csvwrite(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 result=json.loads((OUT/'results.json').read_text());verification=json.loads((OUT/'verification.json').read_text())
 assert verification['status']=='passed' and verification['results_sha256']==sha(OUT/'results.json')
 with (OUT/'rows.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 assert len(rows)==270
 DEST.mkdir(parents=True,exist_ok=True)
 grouped=collections.defaultdict(list)
 for r in rows:grouped[(r['dataset'],int(r['prefix']),r['algorithm'],r['method'],r['checkpoint'])].append(r)
 aggregate=[];lookup={};pathloss={};datasets={}
 for item in result['data']:datasets[item['name']]=dict(np.load(ROOT/'outputs/native_em'/item['path']))
 for key,rs in sorted(grouped.items()):
  name,p,algorithm,method,checkpoint=key
  assert len(rs)==(1 if algorithm=='baseline' else 3)
  r=dict(dataset=name,prefix=p,algorithm=algorithm,method=method,checkpoint=checkpoint,n_initializations=len(rs))
  for metric in METRICS:
   values=[float(x[metric]) for x in rs if x.get(metric)]
   if values:r[metric+'_mean']=float(np.mean(values));r[metric+'_min']=float(min(values));r[metric+'_max']=float(max(values))
  aggregate.append(r);lookup[key]=r;losses=collections.defaultdict(list)
  data=datasets[name];target=data['y'][:,p:p+251,0];clean=data['x'][:,p:p+251,1]
  for x in rs:
   filename=f'prediction_{name}_p{p}_'+(f'{algorithm}_seed{x["seed"]}_{checkpoint}_{method}' if algorithm!='baseline' else method)+'.npz'
   pred=dict(np.load(OUT/filename));mu=pred['mean'];losses['mse_clean'].append(np.mean((mu-clean)**2,axis=1));losses['mse_observed'].append(np.mean((mu-target)**2,axis=1))
   if 'variance' in pred:
    var=pred['variance'];losses['marginal_nll'].append(.5*np.mean(np.log(2*np.pi*var)+(mu-target)**2/var,axis=1))
  pathloss[key]={m:np.mean(v,axis=0) for m,v in losses.items()}
 assert len(aggregate)==110
 csvwrite(DEST/'aggregate.csv',aggregate)
 # Exploratory paired path bootstrap. Average model-initialization losses first;
 # never treat them as independent physical systems or separate training cohorts.
 rng=np.random.default_rng(20262781);contrasts=[]
 specs=[('source_final_vs_corrected_final',('corrected_em','filtered_prefix','trainbest_final'),('source_mode','filtered_prefix','trainbest_final')),
  ('standard_filtering',('corrected_em','filtered_prefix','trainbest_final'),('corrected_em','native_smoothed_initial','trainbest_final')),
  ('corrected_vs_arx',('corrected_em','filtered_prefix','trainbest_final'),('baseline','arx','none')),
  ('training_budget',('corrected_em','filtered_prefix','trainbest_final'),('corrected_em','filtered_prefix','iteration100'))]
 for name,data in datasets.items():
  for p in ([250] if name=='train_continuation' else [25,100,250]):
   indices=rng.integers(len(data['y']),size=(4000,len(data['y'])))
   for label,a,b in specs:
    left=pathloss[(name,p,*a)];right=pathloss[(name,p,*b)]
    for metric in sorted(set(left)&set(right)):
     delta=left[metric]-right[metric];ci=np.quantile(np.mean(delta[indices],axis=1),[.025,.975])
     contrasts.append(dict(dataset=name,prefix=p,contrast=label,metric=metric,left_mean=float(left[metric].mean()),right_mean=float(right[metric].mean()),mean_difference=float(delta.mean()),ci95_lower=float(ci[0]),ci95_upper=float(ci[1]),n_paths=len(delta),n_bootstrap=4000))
 csvwrite(DEST/'paired_contrasts.csv',contrasts)
 def get(name,p,algorithm,method='filtered_prefix',checkpoint='trainbest_final',metric='mse_clean'):
  return lookup[(name,p,algorithm,method,checkpoint)][metric+'_mean']
 headline={}
 for name in datasets:
  p=250;source=get(name,p,'source_mode');correct=get(name,p,'corrected_em')
  headline[name]=dict(source_clean_mse=source,corrected_clean_mse=correct,relative_reduction=1-correct/source,
   arx_clean_mse=get(name,p,'baseline','arx','none'),corrected_coverage95=get(name,p,'corrected_em',metric='coverage95'),
   corrected_marginal_nll=get(name,p,'corrected_em',metric='marginal_nll'))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
 colors={'source_mode':'#B64939','corrected_em':'#256B89'}
 fig,axes=plt.subplots(1,3,figsize=(11.2,3.8),sharex=True,sharey=True)
 for ax,seed in zip(axes,[20262711,20262712,20262713]):
  for algorithm in ['source_mode','corrected_em']:
   trace=json.loads((OUT/algorithm/f'seed{seed}/trace.json').read_text())
   ax.plot([r['iteration'] for r in trace],[r['likelihood']/(50*250) for r in trace],label=algorithm.replace('_',' '),color=colors[algorithm],lw=1.7)
  ax.set_title(f'Initialization {seed}');ax.set_xlabel('EM updates');ax.grid(alpha=.18)
 axes[0].set_ylabel('Training log likelihood / observation');axes[0].legend(frameon=False,fontsize=8)
 fig.suptitle('An implementation defect is not a world-model architecture gap',fontsize=12)
 fig.text(.5,.015,'Same train prefix and initialization. Corrected runs reach the 1000-update cap while still improving.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.06,1,.94))
 for ext in ['png','svg','pdf']:fig.savefig(DEST/f'training_likelihood.{ext}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(11.2,4.2),sharey=True)
 methods=[('baseline','arx','none','ARX', '#777777'),('source_mode','filtered_prefix','trainbest_final','Source','#B64939'),('corrected_em','filtered_prefix','iteration100','Corrected 100','#BD9132'),('corrected_em','filtered_prefix','trainbest_final','Corrected final','#256B89')]
 for ax,name in zip(axes,['independent_dwell50','independent_dwell5','independent_dwell100']):
  for i,(algorithm,method,checkpoint,label,color) in enumerate(methods):
   vals=[lookup[(name,p,algorithm,method,checkpoint)] for p in [25,100,250]]
   means=np.array([v['mse_clean_mean'] for v in vals]);lo=np.array([v['mse_clean_min'] for v in vals]);hi=np.array([v['mse_clean_max'] for v in vals])
   ax.errorbar(np.arange(3)+(i-1.5)*.16,means,yerr=np.vstack((means-lo,hi-means)),fmt='o-',ms=4,capsize=2,lw=1.2,label=label,color=color)
  ax.set_title(name.replace('independent_','Control ').replace('dwell','dwell='));ax.set_xticks(range(3),['25','100','250']);ax.set_xlabel('Observed prefix length');ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Clean output MSE, 251-step forecast');axes[0].legend(frameon=False,fontsize=8)
 fig.suptitle('Standard corrected EM is competent on independent control schedules',fontsize=12)
 fig.text(.5,.015,'64 trajectories per control regime; fixed physical parameters. Bars show 3-initialization range, not confidence intervals.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.065,1,.93))
 for ext in ['png','svg','pdf']:fig.savefig(DEST/f'controlled_forecast.{ext}',dpi=180)
 plt.close(fig)
 compact=dict(status='verified',scientific_decision='REJECT_FRAMEWORK_NECESSITY_FROM_NATIVE_EM_FAILURE',accepted_final_directions=0,
  goal_status='ACTIVE_UNACHIEVED',headline=headline,n_score_rows=len(rows),n_aggregates=len(aggregate),n_paired_contrasts=len(contrasts),
  results_sha256=sha(OUT/'results.json'),verification=verification,fits=[{k:v for k,v in f.items() if k not in ['checkpoints']} for f in result['fits']],
  limitations=['One physical family with fixed parameters;three initialization seeds share one training dataset',
   'Corrected EM is censored at 1000 while improving;source fits stop on negative likelihood gain',
   'No cross-system pretrained transfer or new framework evaluated','Marginal Gaussian scores do not certify joint trajectory law'])
 write(DEST/'results.json',compact)
 table=['| Independent control dwell, prefix=250 | Source filtered MSE | Corrected filtered MSE | ARX MSE | Corrected 95% coverage |','|---|---:|---:|---:|---:|']
 for name,label in [('independent_dwell50','50 (same distribution)'),('independent_dwell5','5 (faster switching)'),('independent_dwell100','100 (slower switching)')]:
  h=headline[name];table.append(f"| {label} | {h['source_clean_mse']:.7g} | {h['corrected_clean_mse']:.7g} | {h['arx_clean_mse']:.7g} | {100*h['corrected_coverage95']:.3f}% |")
 table='\n'.join(table)
 fit_table=['| Mode | Seed | Iterations | Train-only stopping | Latest average LL gain |','|---|---:|---:|---|---:|']
 for f in result['fits']:fit_table.append(f"| {f['algorithm']} | {f['seed']} | {f['total_iterations']} | {f['stopping']} | {f['last_average_ll_gain']:.6g} |")
 fit_table='\n'.join(fit_table)
 source='[Otto, Peitz and Rowley, actuated Koopman generators](https://arxiv.org/abs/2209.09977), [pinned official source](https://github.com/samotto1/KoopmanGeneratorEM/tree/ef830a6d4a6cff699492d54230fc2ad1729cab20).'
 scope='Native released slow-manifold generator/settings plus an explicitly new independent-control-schedule evaluation. This is not a complete paper dimension sweep, a new neural model, cross-dataset transfer, or proof that all controlled partial-observation problems are solved.'
 reports={
 'experimental_results.md':f'''# Cycle27: competent controlled hidden-state baseline

General reusable time-series world modeling remains the objective. This cycle rejects one weak justification for a new framework. Final accepted directions: 0. Overall goal: ACTIVE, UNACHIEVED.

{scope}

The fixed native mechanism is x1dot=-x1+u, x2dot=5*(x1^3-x2), observed y=x2+N(0,.01). dt=.01. Fifty training trajectories; only the first250 observations/249 transitions are used. Augmented latent dimension5 includes a constant and four stochastic coordinates. Three initialization seeds share that one training dataset. Each independent regime has64 fresh initial/noise/control trajectories at fixed physical parameters, not64plants. Prefix25/100/250; horizon251. All models receive the observed prefix and scheduled future controls, never true hidden state or future observations.

{table}

These are mean clean-output MSEs across all three initializations, not best-seed scores. Complete110 aggregates preserve all270 rows, two training checkpoints, two state-initialization conventions, all prefixes and the within-training-trajectory continuation. See outputs/analysis/aggregate.csv. The known-physics reference receives privileged current hidden state and is a numerical diagnostic, not a fair learned comparator.

The displayed final-source versus final-corrected contrast changes achieved training length as well as source math:source stops at220–280 while corrected continues to1000. It is not a matched-update single-factor estimate of the transpose's causal effect. Checkpoint100 supplies a matched-update diagnostic in the complete table. The defensible conclusion is competence of corrected standard EM with the predeclared train-only continuation,not that the entire final MSE reduction is caused by one code line alone.

The source smoother returns Cov(z_k,z_k+1|Y); the generator regression requires its transpose. Exact joint-Gaussian audit agrees with the native smoother at1.249e-16 and shows a .522644 generator discrepancy. Corrected standard EM changes only that generator-regression transpose, not the process-noise formula. This is a baseline implementation repair, not innovation. Source mode faithfully preserves the defect.

{fit_table}

All corrected fits are still improving at the1000-update cap; do not claim convergence or residual model-class failure. Source stopping values are negative gains, not successful stationary EM convergence. The finite-budget corrected baseline is already competent, so extending it solely to seek failure is unnecessary. These budgets and stopping conditions were frozen before held-out scoring. Training log likelihood per observation and all seed traces remain available.

Resume bookkeeping limitation:first source seed already had negative average LL gain at native270;the resumed backend nonetheless ran one additional10-update block to280. That checkpoint did not improve the train-best selection. Actual traces and stopping counts are preserved;do not claim identical stop timing or silently rewrite them. This does not affect corrected fits or their declared1000-update budget.

State estimation also matters: native smoothed-time0 free-run and standard last-prefix filtering use the same weights and inputs but different conditioning approximations. Filtering improvements are ordinary inference repairs. Paired path bootstrap averages losses over initializations first;4000 resamples,seed20262781,95%percentile intervals. All contrasts, including negative or uncertain ones, are saved. They are exploratory, not multiplicity-corrected or independent training-seed inference. Different dwell cohorts are independent, not paired causal interventions.

Runtime of the amended study: {result['runtime_s']:.3f}s on one CPU thread; no GPU training. This excludes the preserved original slower native worker and one-time JIT/proof costs. Source-equivalent batching reproduced literal native iteration100 parameters within4.561e-11 and likelihood within2.758e-9. The source-mode first seed resumed native iteration270; other five fits used their predeclared initializations. Original source, logs and checkpoints remain unchanged.

Independent verification:4data hashes,387checkpoint hashes,270prediction hashes,242000physical state values,120posterior histories,270complete forecast arrays and1830scores. Canonical NumPy filtering/RTS, closed-form physical dynamics, independent augmented least-squares ARX and all-score recomputation passed. Maximum forecast mean difference2.561e-11,variance4.108e-9,score2.843e-14. Physical odeint/closed-form discrepancy1.596e-6 under predeclared tolerances. Result SHA256={sha(OUT/'results.json')}.

Gaussian NLL/coverage are pointwise marginal metrics. They do not establish correct joint path uncertainty, control policies, or general physical faithfulness. {source}
''',
 'negative_evidence.md':f'''# Negative evidence

The proposed broad claim that controlled partial observations require a new state-learning framework is not supported by this native baseline. Standard corrected EM with the predeclared train-only continuation reduces same-distribution independent clean-output MSE from{headline['independent_dwell50']['source_clean_mse']:.8g} to{headline['independent_dwell50']['corrected_clean_mse']:.8g} ({100*headline['independent_dwell50']['relative_reduction']:.4f}% reduction), with strong improvements in faster/slower input switching too. The two final modes have different achieved iteration counts;this is not a single-factor estimate of the transpose alone. This is one physical family, not a universal solution theorem.

1. Official-source failure can be an implementation defect. Exact Gaussian sufficient statistics exposed an orientation error; the correction is old mathematics. Do not rename it as a method.
2. A100-step run is an inadequate sole competent-baseline comparison. Both100/final budgets are retained. Corrected runs improve through1000; no residual deficiency is accepted from a censored optimizer.
3. Inferring time0 from a prefix and then freely simulating through that already observed prefix discards process-noise conditioning. Ordinary filtering is the relevant repair, not a new recurrent architecture.
4. Low-dimensional adaptation, shared latent mechanisms and observation correction already have direct precedents. Their conjunction is not automatically solved, but no residual experiment currently supports a new combination.
5. Native polynomial full lift is deliberately well specified. This experiment does not verify the broader pretrained-transfer hypothesis. It cannot justify either promoting that hypothesis or claiming it universally solved.
6. Source mode negative likelihood gains are evidence about the defective implementation, not a calibrated baseline optimization limit.

The broad pretrained-state transfer candidate is withdrawn as a proposed method. No replacement is promoted. No public release has been made. Full results and disfavored seeds remain.
''',
 'baseline_registry.md':f'''# Baseline registry

| Baseline | Executed status | Information and interpretation |
|---|---|---|
| Official controlled Koopman/EM module | Literal native first seed through270updates; all original checkpoints retained | Native dimension-sweep settings, not full dimension sweep or all paper results |
| Source-equivalent batched EM | Three seeds,220/250/280 updates | Source defect preserved; iteration100 replay proves numerical fidelity |
| Standard corrected EM | Three seeds,1000 updates each | Same dimension, data, initialization and inference access; generator covariance transpose only; capped while improving |
| Native smoothed-time0 forecast | All source/corrected weights and budgets | Prefix information only; free propagation through prefix is a modeling/inference convention |
| Standard filtered-prefix forecast | Same weights and budgets | Ordinary last-prefix posterior propagation; no future observation access |
| ARX10 | Train-prefix ridge selection, refit, evaluated | Same observed controls/output,10lags each; fixed candidate regularization set |
| Persistence | All conditions | Last observed scalar output repeated |
| Known physics / exact exponential lift | All conditions | Privileged true current state and coefficients, diagnostic only |
| WorldTS native numerical core | Previously executed cycle22 | Recent framework reference; no new run or transfer result here |
| CoDA / Meta-iLaD / OSSM / SimDist | Literature/source audit; no new fitting | Exact assumptions and public-asset availability in research notes; never claim reproduced |

{source}

Data hashes,fit-input boundary,source commit and all270prediction paths are in outputs/native_em/fit_input_receipt.json and outputs/controlled_em_study/results.json. Dependencies stay task-local under cycle26/vendor/native_control. All raw NPZ data/weights/predictions stay on the server; compact verified outputs and reports are mirrored locally. No prior artifacts or environments deleted.
''',
 'literature_review.md':f'''# Literature and innovation boundary

Three independent primary-literature perspectives are preserved verbatim in research/modern_reassessment.md,research/strong_reassessment.md,research/adjacent_reassessment.md. They explicitly distinguish verified source/checkpoint assets from promised releases and retain the2026-04-16 to2026-10-16 search window. Earlier DynaBase/sysid-transformer findings are credited as refreshes, not new discoveries.

{source} This work already learns controlled bilinear latent dynamics from partial observations using Kalman/RTS and EM. Its native example is therefore a required competence control, not a new proposal. The verified source defect does not invalidate the underlying existing method.

[CoDA](https://proceedings.mlr.press/v162/kirchmeyer22a.html) already freezes shared dynamics and adapts a low-dimensional context on fully observed systems. [Meta-iLaD author materials](https://github.com/GoForIt-YYB/Meta-iLaD) assume invertible mixing and locally constant latent environments; the inspected equations lack arbitrary future action programs, and native source is not released. [OSSM](https://arxiv.org/abs/2609.33566) explicitly separates propagation from observation correction. These are actual assumption boundaries: no one paper is being claimed to solve every possible controlled hidden-state transfer setting.

Recent [BRICKS-WM](https://arxiv.org/abs/2606.16489) already uses reusable agent/background modules and a standardized interface. [SimDist](https://github.com/CLeARoboticsLab/simdist) already reuses encoder/reward/value and adapts dynamics through identification; Go2 source/weights are public, but inspected real/sim dataset folders contain only.gitignore. [Continual compositional world models](https://arxiv.org/abs/2609.22055) distinguish reuse from forgetting and expose temporal-composition/state-routing limits, with budget and privileged-encoder caveats. The inspected project does not supply a verified working native code link. [MHWA](https://arxiv.org/abs/2610.14727) already uses slow context and fast state-conditioned routing. Generic modularity, adapter reuse, event timing or routing is not an unoccupied contribution.

[Neural Event ODEs](https://arxiv.org/abs/2011.03902) and [Neural Hybrid Automata](https://papers.neurips.cc/paper_files/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html) occupy learned event/stochastic transition primitives. Source/paper links and exact release caveats are in adjacent_reassessment.md; these facts do not prove all hybrid inference problems solved.

Synthesis: withdraw the broad pretrained-state transfer candidate as a new method. Existing components alone do not logically prohibit every novel combination, but no demonstrated residual currently justifies one. Cycle27 provides direct counterevidence to deriving architecture necessity from a broken ordinary baseline. Zero accepted final directions; the general world-model objective remains open.
''',
 'hypothesis_space.md':'''# Competing hypotheses and decisions

| Hypothesis | Discriminating evidence | Current judgment |
|---|---|---|
| Source baseline fails because hidden controlled dynamics require a new framework | Correct standard EM on identical data,dimension,initialization | Rejected as the explanation of this failure |
| Baseline weakness is a source generator M-step error | Exact Gaussian sufficient-statistic regression; source-equivalent replay | Supported; implementation defect, not contribution |
| Short training alone explains all weakness |100/final weights for both modes; full train LL traces | Training matters; corrected1000 still improving; do not separate all effects using one contrast |
| Smoothed-time0 free-run is equivalent to filtered-prefix prediction | Same weights, same observed prefix, both conventions | Generally false; standard conditioning is an ordinary repair |
| Reusable state transfer under new mechanisms has a residual beyond existing identification/adaptation | Competent matched-information cross-system experiment | Not tested here; broad candidate withdrawn, not declared solved universally |
| A general new world model has now been established | Independent multi-mechanism evidence plus prior-art distinction | No; zero accepted directions |

Do not continue adjusting this toy or add another architecture to preserve an unsupported line. A new hypothesis must specify a failure of competent recent reusable models, the missing capability and an independent falsifier before training. Missing code alone is not a Research Gap. No test metric is used for model/hyperparameter selection.
''',
 'research_ideas.md':'''# Research direction triage

Accepted directions:0. Do not recommend an unverified idea merely to fill a report.

Withdrawn as a proposed new framework: economical pretrained-state transfer phrased as shared latent dynamics plus a small context/input adapter. Direct predecessors already supply those mechanisms, and this cycle's ordinary corrected baseline is competent. This does not settle every interaction of hidden observation,control and transfer; it removes the current justification.

Also not promoted: generic continual modular reuse, state-conditioned expert routing, learned event timing, branch sharing or observation-correction consistency. The three independent perspective notes explain direct predecessors and remaining assumption boundaries. No narrow task score or code defect is offered as a substitute innovation.

The next useful selection must begin from a verified failure of an actually competent reusable world-model framework on an existing independent task family. First freeze the intended capability and access to observations/inputs; then compare an established repair with the native model under the same information. Avoid starting another custom generator merely because existing candidates have been falsified. This is a selection rule, not an accepted research direction or a training launch.
''',
 'final_research_direction.md':f'''# Current research decision: no final direction yet

**Research Question.** Can we identify a reusable time-series world-model capability that existing competent frameworks fail to learn or transfer, beyond ordinary numerical/inference/optimization repairs? This remains the overarching question, not a validated paper problem.

**Evidence.** Cycle27 completed6 declared EM fits and270forecast settings; independent verification passed. Same-distribution clean MSE{headline['independent_dwell50']['source_clean_mse']:.7g}→{headline['independent_dwell50']['corrected_clean_mse']:.7g} with standard corrected EM and predeclared train-only continuation; faster/slower switching also improve. Final iteration counts differ,so this does not isolate the code transpose's effect from additional training. This rejects the present native-failure argument for a new framework. One fixed physical family is not cross-system evidence.

**Existing Methods.** Controlled latent EM, CoDA-style context adaptation, observer correction, SimDist selective adaptation and recent modular world models are competent relevant precedents. Exact assumptions differ; their existence is not a proof that every combination is solved. Nevertheless no surviving empirical residual currently supports the broad adapter-based proposal.

**Proposed Idea.** None accepted. The pretrained-state transfer proposal is withdrawn in its current broad form. Source correction/filtering/batching are not new methods.

**Falsification.** A new candidate must survive a same-information ordinary repair,train convergence/capacity controls and evaluation on an independent mechanism. If an established repair removes its defining failure,abandon its architecture-necessity claim. Current candidate fails this evidential gate; the overarching field is not declared solved.

**Feasibility.** Server source,data,all weights and verification are preserved in D:/tswm_discovery_20261016_v1/cycle27. Existing single RTX5090 remains available; this positive control needed one CPU thread. Corrected runs hit1000updates while improving,which is a limit on convergence claims,not evidence of unsolved dynamics.

**Next Experiment.** No additional training is justified on this same native toy. Before a new launch,select and freeze a capability failure from an independently released recent reusable framework,with its strongest established repair and a second existing mechanism. The current audit does not supply that verified failure; literature-selection work remains. Do not present this pending selection as an implemented experiment or final recommendation.

General objective ACTIVE,UNACHIEVED. Final recommended directions0. No public publishing or deletion. All seven requested living reports are updated in this cycle.
'''}
 for name,body in reports.items():(ROOT/name).write_text(body,encoding='utf-8')
 write(ROOT/'CYCLE_STATUS.json',dict(cycle=27,classification='PROGRESS',status='verified',goal_status='ACTIVE_UNACHIEVED',accepted_final_directions=0,
  decision=compact['scientific_decision'],n_score_rows=270,n_aggregates=110,n_paired_contrasts=len(contrasts),main_results_sha256=sha(OUT/'results.json'),verification_sha256=sha(OUT/'verification.json'),report_script_sha256=sha(pathlib.Path(__file__))))
 print(json.dumps(dict(headline=headline,aggregates=len(aggregate),contrasts=len(contrasts),reports=len(reports))),flush=True)

if __name__=='__main__':main()
