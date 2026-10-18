from common87_v1 import *
import csv,io,shutil
import numpy as np
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def put(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
def main():
 result=read(O/'EVALUATION_RESULTS87_V3.json');verify=read(O/'VERIFICATION87_V4.json');assert verify['status']=='SOURCE_FIELDS_SELECTIONS_METRICS_AFFINE_REPLAYED_ROLLOUT_QUALIFICATION_INCOMPLETE' and verify['results_sha256']==sha(O/'EVALUATION_RESULTS87_V3.json');selection=read(O/'SELECTION87_V1.json');empirical=read(O/'EMPIRICAL_SELECTION87_V1.json');ordinary=read(O/'ORDINARY_SELECTION87_V1.json');data=read(O/'DATA_MANIFEST87_V1.json');pub=O/'publication_v1';pub.mkdir(exist_ok=False);figroot=RAW/'figures_v1';figroot.mkdir(exist_ok=False)
 rows=result['rows'];groups=['frozen','whole_filter','resampled_ic','unselected','ordinary','empirical_prior_whole_filter','empirical_prior_unselected','persistence'];labels=['Frozen native','Native filtered','Native new ICs','Native all','Weak polynomial','EB filtered prior','EB all prior','Persistence'];colors=['#666666','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#999999']
 fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
 for fi,fam in enumerate(['duffing','stuart_landau']):
  for ci,cohort in enumerate(['retained','crossed','unselected']):
   ax=axes[fi,ci]
   for j,(group,color) in enumerate(zip(groups,colors)):
    hits=[r for r in rows if r['family']==fam and r['cohort']==cohort and r['arm']==group];assert hits
    values=[r['horizons'][-1]['failure_penalized_mean_nrmse'] for r in hits];ax.scatter(np.full(len(values),j),values,color=color,s=30,alpha=.85);ax.plot([j,j],[min(values),max(values)],color=color);ax.scatter(j,np.mean(values),marker='_',s=180,color=color)
   ax.set_yscale('log');ax.set_xticks(range(len(groups)),labels,rotation=55,ha='right',fontsize=8);ax.set_ylabel('Mean NRMSE, failed path =1000');ax.set_title(f'{fam} / {cohort}, horizon10');ax.grid(axis='y',alpha=.25)
 fig.suptitle('Native nonlinear comparison: all optimizer seeds and ordinary controls\nDots are optimizer seeds; ranges are not confidence intervals',fontsize=12);figure1=figroot/'simulation87_v1.png';fig.savefig(figure1,dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for fit in selection['fits']:
  color=colors[groups.index(fit['arm'])];trace=fit['trace'];ax.plot([t['step'] for t in trace],[t['val_normalized_field_l1'] for t in trace],color=color,alpha=.65,label=fit['arm'] if fit['seed']==2026872 else None)
 for choice in empirical['selections']:ax.axhline(choice['val_native_normalized_field_l1'],linestyle='--',label='EB '+choice['arm'])
 ax.axhline(read(O/'VAL_QUALIFICATION87_V1.json')['selected']['exact_native_normalized_field_l1'],color='#CC79A7',linestyle=':',label='weak polynomial');ax.set_xlabel('Native optimizer updates');ax.set_ylabel('Common VAL native-normalized field L1');ax.set_title('Source competence and same-supervision prior control');ax.legend(fontsize=8);ax.grid(alpha=.2);figure2=figroot/'validation87_v1.png';fig.savefig(figure2,dpi=160);plt.close(fig)
 text=['# C87 native nonlinear data-selection comparison','', 'This is a controlled native-model adaptation study, not full-paper reproduction. Learned-rollout numerical qualification remains incomplete; see NUMERICAL_OUTCOME87_V1.md. Simulation comparisons below describe the declared RK4/TF32 protocol. Final accepted research directions remain0. Exact released-checkpoint training data are unavailable. Source and native-model qualification succeeded; held-out model selection remained sealed.','', '## Source competence','', '| seed | native arm | selected update | common VAL normalized field L1 |','|---|---|---:|---:|']
 for fit in selection['fits']:text.append(f"| {fit['seed']} | {fit['arm']} | {fit['selected']['step']} | {fit['selected']['val_normalized_field_l1']:.6g} |")
 text+=['','The existing empirical-prior controls use the same source32 independent field labels, without family or parameter labels. Their two-component/noise choices were selected before test opening. This is an ordinary empirical Bayes procedure, not a proposed novel method.','']
 for e in empirical['selections']:text.append(f"- {e['arm']}: VAL native-normalized L1 {e['val_native_normalized_field_l1']:.6g}; raw field MSE {e['val_raw_field_mse']:.6g}.")
 text+=['','## Source screening','']
 for d in data['files']:
  if 'accepted_count' in d:text.append(f"- {d['split']} {d['family']}: {d['accepted_count']}/{d['pool_count']} retained ({d['retained_fraction']:.3%}); all numerical paths finite.")
 text+=['','## Held-out results','', 'Each cohort has128 mechanisms per family. Retained/crossed contexts preserve the mechanism/IC-pair marginals but change their joint pairing and visited-state information. New-IC free simulation uses no future observations. Every failure remains counted. Population MSE is undefined if any path fails; finite-only values are explicitly separated in JSON/CSV.','', '| model | family | cohort | H10 failures | population MSE | population mean NRMSE | penalized mean NRMSE |','|---|---|---|---:|---:|---:|---:|']
 table=[]
 for r in rows:
  h=r['horizons'][-1];fmt=lambda x:'undefined' if x is None else f'{x:.6g}';text.append(f"| {r['model']} | {r['family']} | {r['cohort']} | {h['failed_count']} | {fmt(h['population_mse'])} | {fmt(h['population_mean_nrmse'])} | {h['failure_penalized_mean_nrmse']:.6g} |")
  for hh in r['horizons']:table.append({**{k:r[k] for k in ['model','seed','arm','family','cohort','context_field_mse','future_field_mse']},**hh})
 text+=['','## Evidence limits and verification','', f"Replayed all{verify['closed_source_checkpoints']} native source-VAL checkpoints and all reported metric rows. Native field arrays were independently queried through direct model_forward ({verify['direct_native_field_scalars']} scalars); high-accuracy physical truth and empirical Gaussian conditioning were checked. Full-population nonlinear learned rollout integration was not independently replayed. Supplementary learned-rollout checks failed the declared1% criterion on the frozen model and did not reach the adapted-model checks. See NUMERICAL_QUALIFICATION87_V1.json and NUMERICAL_OUTCOME87_V1.md. Metrics are fixed-protocol numerical observations, not certified continuous-time errors. See VERIFICATION87_V4.json for exact scope.",'', 'Native source losses and ordinary controls must be considered before assigning a sampler-specific failure. Resampling changes both native observation contexts and its appended supervised field-query locations. A fixed-supervision control is required for shortcut attribution. C87 does not establish novelty, released-checkpoint causality, or a need for a new architecture.']
 put(pub/'experimental_results.md',('\n'.join(text)+'\n').encode());put(pub/'negative_evidence.md',b'# Negative evidence and unproved claims\n\nThe prospective nonlinear populations have high acceptance; no threshold was changed to force bias. The native learner is compared with a same-supervision empirical-prior control. All source VAL choices are retained. The weak per-task fitter alone does not account for the mechanism prior available through synthetic supervision. Held-out errors, failures and metric tradeoffs are all retained in experimental_results.md. C87 alone cannot isolate a context shortcut from changed supervised state coverage, explain a released checkpoint causally, or establish an original method. Final accepted directions remain0.\n')
 output=io.StringIO();writer=csv.DictWriter(output,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table);put(pub/'metrics87_v1.csv',output.getvalue().encode())
 for path in C.glob('*.md'):
  if path.name!='final_research_direction.md':put(pub/path.name,path.read_bytes())
 for path in (C/'scripts').glob('*.py'):put(pub/'scripts'/path.name,path.read_bytes())
 # Derive conservative final study judgment from the actual closed evidence.
 wins=0;total=0
 for r in rows:
  if r['arm'] not in ['whole_filter','resampled_ic','unselected']:continue
  baseline='empirical_prior_unselected' if r['arm']=='unselected' else 'empirical_prior_whole_filter'
  control=next(q for q in rows if q['model']==baseline and q['family']==r['family'] and q['cohort']==r['cohort'])
  total+=1;wins+=control['horizons'][-1]['failure_penalized_mean_nrmse']<r['horizons'][-1]['failure_penalized_mean_nrmse']
 decision='The matched-source ordinary empirical-prior control beats every reported native adaptation comparison under the declared numeric protocol. Together with source-VAL evidence and close prior art, this does not justify a new resampling method. Full continuous-time numerical qualification is incomplete.' if wins==total else 'Ordinary-control outcomes are mixed; do not infer universal robustness or method necessity.'
 direction=f"# C87 closed study decision\n\nResearch Question: Does trajectory-dependent source selection impair reusable dynamics inference and new-initial-state simulation?\n\nEvidence:9 unchanged native12.968M-parameter adaptations,3 optimizer seeds,2 bounded nonlinear families,3 held-out cohorts and14 models. The matched-source empirical-prior baseline wins {wins}/{total} native comparisons in H10 failure-penalized mean NRMSE. All MSE/NRMSE/failure/field results are retained separately.\n\nExisting Methods: ordinary empirical Bayes with weak-form identification is a working same-supervision comparator. Selection bias, censor-aware inference and IC augmentation are established.\n\nProposed Idea: no original method accepted. {decision}\n\nFalsification: source losses, ordinary sufficiency, metric tradeoffs, and changed native observation-derived supervision prevent a shortcut attribution from this study alone. No released-checkpoint causal defect is established.\n\nFeasibility: the declared native runs and evaluation are closed; raw data/weights/predictions remain on the remote15TB disk. Independent verification scope is explicit in VERIFICATION87_V4.json.\n\nNext Experiment: review the paired effect and ordinary controls before committing further compute. A new direction requires a failure left unresolved by these controls, with independent novelty grounding; expanding this sampler claim without such a failure is not justified.\n\nFinal accepted directions0. Overall goal ACTIVE/UNACHIEVED.\n"
 put(pub/'final_research_direction.md',direction.encode())
 write(O/'STUDY_DECISION87_V1.json',dict(utc=utc(),matched_empirical_prior_wins=wins,total_native_comparisons=total,decision=decision,metric='H10 failure_penalized_mean_nrmse',accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED'))
 receipts=['PAIRED_EFFECT_ANALYSIS87_V1.json','PRECISION_DIAGNOSTIC_STOP87_V1.json','NUMERICAL_QUALIFICATION87_V1.json','BATCH_PRECISION87_V2.json','AFFINE_DIAGNOSIS87_V1.json','AFFINE_CORRECTION87_V1.json','EVALUATION_RESULTS87_V2.json','STUDY_DECISION87_V1.json','DATA_MANIFEST87_V1.json' ,'DATA_VERIFICATION87_V4.json','NATIVE_SMOKE87_V1.json','TRAIN_OPENING87_V1.json','SELECTION87_V1.json','ORDINARY_SELECTION87_V1.json','VAL_QUALIFICATION87_V1.json','EMPIRICAL_SELECTION87_V1.json','EVAL_OPENING87_V1.json','EVAL_MANIFEST87_V2.json','EVALUATION_RESULTS87_V3.json','VERIFICATION87_V4.json','GENERATION_PROVENANCE87_V1.json','QUEUE_HOLD87_V1.json']
 for name in receipts:put(pub/'evidence'/name,(O/name).read_bytes())
 receipt=read(O/'SOURCE_PROVENANCE87_V1.json');receipt.pop('paper_filtering_excerpt_audit_only',None);receipt['original_server_receipt_sha256']=sha(O/'SOURCE_PROVENANCE87_V1.json');put(pub/'evidence/SOURCE_PROVENANCE87_V1.json',json.dumps(receipt,indent=2).encode())
 for stem in ['check_rollout87_v1','check_rollout87_v2','check_rollout87_v3']:
  put(pub/'evidence'/stem/'worker.log',(O/stem/'worker.log').read_bytes());put(pub/'evidence'/stem/'EXIT.json',(O/stem/'EXIT.json').read_bytes())
 for path in [figure1,figure2]:put(pub/'figures'/path.name,path.read_bytes())
 put(pub/'README.md',b'# Native nonlinear sampler study\n\nRead experimental_results.md, negative_evidence.md, STRONG_PRIOR_CONTROL87_V1.md and SUPERVISION_BOUNDARY87_V1.md. Native9-fit adaptation and14-model held-out comparison; raw arrays/checkpoints/logs remain server-only. No final research direction accepted. Learned-rollout numerical qualification is incomplete; failed checks are retained.\n')
 files=[]
 for p in sorted(pub.rglob('*')):
  if not p.is_file():continue
  raw=p.read_bytes();files.append(dict(path=p.relative_to(pub).as_posix(),server_snapshot_file=str(p),bytes=len(raw),sha256=sha(p),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),encoding='binary' if p.suffix=='.png' else 'utf-8'))
 write(pub/'manifest.json',dict(utc=utc(),stage='C87',status='NATIVE_NONLINEAR_STUDY_WITH_INCOMPLETE_ROLLOUT_QUALIFICATION',files=files,goal='ACTIVE_UNACHIEVED',accepted_directions=0))
 write(O/'CLOSURE87_V1.json',dict(utc=utc(),status='READY_FOR_FIGURE_QA_AND_PUBLICATION',publication_root=str(pub),manifest_sha256=sha(pub/'manifest.json'),figures=[dict(path=str(p),sha256=sha(p)) for p in [figure1,figure2]],overall_goal='ACTIVE_UNACHIEVED',accepted_final_directions=0))
 log(dict(stage='snapshot_finalized_pending_figure_qa',files=len(files),manifest_sha256=sha(pub/'manifest.json')))
if __name__=='__main__':main()
