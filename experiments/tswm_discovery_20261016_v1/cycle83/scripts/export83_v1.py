import sys,pathlib,hashlib,csv,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import numpy as np
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def write(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text)
def main():
 r=read(OUT/'RESULTS83_V1.json');v=read(OUT/'VERIFICATION83_V1.json');assert v['status'].startswith('PASS_') and read(OUT/'run83_v1_exit.json')['exit_code']==0;kinds=['FIM_local_composed','weak_polynomial3','weak_rff128'];groups=[]
 for family in ['fhn','hill']:
  for noise in [0.,.03]:
   for kind in kinds:
    rr=[z for z in r['rollouts'] if z['family']==family and z['noise_sigma']==noise and z['kind']==kind];ff=[z for z in r['fields'] if z['family']==family and z['noise_sigma']==noise and z['kind']==kind];response=[z for z in r['responses'] if z['family']==family and z['noise_sigma']==noise and z['kind']==kind];complete=[z for z in rr if z['finite_complete'] and z['solver']['success']];task_scores=[]
    for tag in sorted({z['tag'] for z in rr}):
     selected=[z for z in rr if z['tag']==tag];ok=all(z['finite_complete'] and z['solver']['success'] for z in selected);task_scores.append(dict(tag=tag,median=float(np.median([z['coordinate_mean_nrmse'] for z in selected])) if ok else None,all_complete=ok))
    goodtasks=[z['median'] for z in task_scores if z['all_complete']];groups.append(dict(family=family,noise_sigma=noise,kind=kind,expected_rollouts=len(rr),complete_rollouts=len(complete),trajectory_finite_median=float(np.median([z['coordinate_mean_nrmse'] for z in complete])) if complete else None,all_tasks_complete=len(goodtasks)==4,task_median=float(np.median(goodtasks)) if len(goodtasks)==4 else None,field_median=float(np.median([z['coordinate_mean_nrmse'] for z in ff if z['finite_complete']])),response_finite_median=float(np.median([z['coordinate_mean_nrmse'] for z in response if z['finite_complete']])) if any(z['finite_complete'] for z in response) else None,task_scores=task_scores))
 lookup={(z['family'],z['noise_sigma'],z['kind']):z for z in groups};native_complete=all(lookup[f,n,kinds[0]]['complete_rollouts']==16 for f in ['fhn','hill'] for n in [0.,.03]);accuracy=all(lookup[f,n,kinds[0]]['task_median'] is not None and lookup[f,n,kinds[0]]['task_median']<=.3 for f in ['fhn','hill'] for n in [0.,.03]);comparisons=[]
 for f in ['fhn','hill']:
  for noise in [0.,.03]:
   native=lookup[f,noise,kinds[0]]['task_median']
   for kind in kinds[1:]:
    base=lookup[f,noise,kind]['task_median'];threshold=.75 if noise else 1.25;passed=native is not None and base is not None and native<=threshold*base;comparisons.append(dict(family=f,noise_sigma=noise,baseline=kind,native_median=native,baseline_median=base,required_ratio=threshold,passed=bool(passed)))
 passed=native_complete and accuracy and all(z['passed'] for z in comparisons);decision='CONTINUE_LOCAL_FOUNDATION_COMPOSITION_WITH_EXTERNAL_VALIDATION' if passed else 'GENERIC_LOCAL_FOUNDATION_COMPOSITION_ADVANTAGE_NOT_ESTABLISHED';dump(OUT/'DECISION83_V1.json',dict(utc=utc(),decision=decision,frozen_development_gate_passed=bool(passed),native_all_complete=native_complete,native_accuracy_gate=accuracy,comparisons=comparisons,groups=groups,qualified_final_directions=0,overall_goal='ACTIVE_UNACHIEVED'))
 def fmt(x):return 'failed/incomplete' if x is None else f'{x:.6g}'
 table='| Family | Noise | Method | Field median | Trajectory median (finite) | Task median (all complete) | Response median (finite) | Complete |\n|---|---:|---|---:|---:|---:|---:|---:|\n'
 for z in groups:table+=f"| {z['family']} | {z['noise_sigma']} | {z['kind']} | {fmt(z['field_median'])} | {fmt(z['trajectory_finite_median'])} | {fmt(z['task_median'])} | {fmt(z['response_finite_median'])} | {z['complete_rollouts']}/{z['expected_rollouts']} |\n"
 fail=[z for z in r['rollouts'] if not z['finite_complete'] or not z['solver']['success']];diagnosis='The frozen development gate passes; broader transfer and closest-method comparisons remain mandatory.' if passed else 'The generic composition route has not demonstrated its required advantage over ordinary local identification. Do not promote stitching itself to a novel world-model mechanism.'
 report=f'''# C83 — Larger-system simulation from local inferred fields

Decision: `{decision}`. {diagnosis} No final research direction is accepted.

## Research question and actual execution

Can a released low-dimensional field inference model, reused through target-specific local parent sets, form a larger autonomous simulator and propagate a prescribed intervention? C82 tested local primitives; this cycle actually composes a12-state coupled oscillator ring and a7-state nonlinear repression ring. There are two parameter settings in each family, two initial-context banks and clean/.03-noise conditions:16 tasks. Related settings and paired noise tasks are not16 independent datasets. Four context trajectories each have256 observations over[0,8].

All methods know physical coordinates and the same exact local parent graph. Ordinary polynomial regression is source-informed for the cubic oscillator family; fixed RFF128 supplies a different flexible control. Both fit weak state-increment equations and choose one ridge for the complete composed model from three candidates using observed-context coupled validation only. Refitting uses all256 observations. No future test state or derivative selects a model. FIM's large pretrained prior and its training cost are additional information, not a capacity/budget match.

Each composed model simulates two independently generated initial states under both free dynamics and `do(x0(t)=x0(0)+.5 sin(.8t))`. Nonintervened variables evolve under the fitted coupled field; no future factual state is supplied. Total192 trajectories,48 local-field records and96 paired intervention-response records. Fields are measured at128 context-union points and do not certify off-support accuracy. All finite medians below are accompanied by completion counts; task medians require every constituent simulation to complete.

## Results

The primary score averages per-coordinate RMSE/std(reference coordinate); clamped-coordinate0 is excluded. Field, trajectory and response denominators refer to different quantities and must not be interchanged. Noisy/clean conditions share clean trajectories; parameter settings share initial-state banks.

{table}

Failed or incomplete trajectories: {len(fail)}. All failure messages and partial/empty arrays remain in raw server artifacts and compact result records. True-field evaluation-solver checks cover64 trajectories; maximum coordinate-mean NRMSE is{max(z['score']['coordinate_mean_nrmse'] for z in r['true_solver_checks']):.6g}. Hence the reference/evaluation solver mismatch is measured rather than assumed negligible.

![Composition outcomes](figures/composition83_v1.png)

## Attribution and limitations

Local field error, global rollout error, paired intervention-response error and support-distance diagnostics are distinct evidence. Context-neighbor distances are descriptive; they do not prove identifiability or isolate distribution shift from feedback amplification. This screen cannot settle latent-state recovery, graph discovery, unseen sensors, unknown interventions or real-data transfer. We do not infer failure from crossing the model's original3D task limit alone. Existing modular/meta-learning and operator-composition work already covers generic composition. Native UNICON/GICON/CHOP were inspected, not executed; no claim of outperforming those methods is made.

No original-paper table reproduction is asserted. Zero neural weight updates were performed. Independent native batch qualification covers every local target in both families at17 points with the original helper. End-to-end native field replay passes on all16 tasks. Original70 source files and exact checkpoint hashes were checked before/after execution.

## Verification and reproducibility

Independent vectorized true equations reproduce context and reference trajectories, maximum reference-coordinate discrepancy{v['maximum_reference_error']:.6g}. All336 score records were independently recomputed, maximum difference{v['score_max_abs']:.6g}; {v['prediction_target_scalars']} prediction/target scalars checked. Selected ordinary coefficients use separate per-window quadrature and augmented least-squares verification: {len(v['independent_coefficients'])} local models,{sum(z['parameters'] for z in v['independent_coefficients'])} coefficients; maximum discrepancy{max(z['max_abs'] for z in v['independent_coefficients']):.6g}. Failed candidates remain visible.

Numerical screen wall time{r['wall_seconds']:.2f}s, CPU two threads with process-local E-core affinity; no shared runtime packages, hardware settings or unrelated jobs changed. All context, model and prediction payloads remain on server15TB storage. GitHub receives source, protocol, hashes, compact results, logs and this figure; no result payload synchronization to the local machine.

## Next decisive experiment

If the gate passes, test unseen mechanism families and system sizes under equal-information ordinary graph/identification baselines, then compare relevant native operator methods under compatible contracts. If it fails, retain this negative capability evidence and diagnose whether a differentiated uncertainty/transfer mechanism exists before adding architecture. Local-model composition alone is not a research contribution. The separate original-budget modular-meta-learning campaign continues without early reserved evaluation. Overall ACTIVE/UNACHIEVED.
'''
 write('COMPOSITION_AUDIT83_V1.md',report)
 texts={
 'literature_review.md':'See LITERATURE_BOUNDARIES83_V1.md and SOURCE_AUDIT83_V1.json. UNICON/GICON/CHOP full-paper contracts and actual public repositories were inspected; CHOP has models, checkpoints and recursive PDE evaluation. Generic frozen composition and residual correction are occupied. No external model was reproduced in this source-only audit.',
 'baseline_registry.md':'Executed: unchanged13M FIM target-local composed fields, ordinary local degree3 polynomial and RFF128 weak regression.16tasks,96 composed validation candidates,32 full composed weak refits,192 complete-or-failed trajectory records. Known exact parent graph for all. Source-informed polynomial family control. Pretrained computation not matched to ordinary fitting.',
 'experimental_results.md':table+'\nAll48 fields,192 trajectories,96 responses and96 validation candidates remain in outputs/RESULTS83_V1.json. Raw arrays stay on server. Gate: '+decision,
 'negative_evidence.md':diagnosis+'\nFailure counts and solver messages are preserved. Finite medians cannot conceal failed trajectories. A source-informed ordinary baseline is explicitly labeled. No toy score is upgraded to a broad unsolved-gap claim.',
 'hypothesis_space.md':'H1 useful local pretrained prior: frozen gate '+decision+'. H2 ordinary local identification explains apparent advantages: strong polynomial/RFF controls. H3 error/support departure in coupled simulation: field and rollout/support diagnostics distinguish measurements, but do not fully isolate causes. H4 implementation/solver mismatch: native interface, true integration and independent checks pass. Unknown-sensor/hidden-state extensions remain untested.',
 'research_ideas.md':'No qualified final research direction. Candidate lineage, overlap challenge and alternative observation-transfer route are in RESEARCH_BRIEF83_V1.md. C83 supplies larger-system simulation evidence, not novelty. Do not add a residual/selector/stitching component already present in CHOP and relabel it as innovation.',
 'final_research_direction.md':f'Research Question: can a low-dimensional pretrained field prior support larger controlled worlds through known local mechanisms? Evidence: actual16tasks/192simulations/96responses, decision {decision}. Existing Methods: modular/meta-learning, graph operators, CHOP, ordinary weak identification. Proposed Idea: no accepted differentiated mechanism. Falsification: ordinary target-local models explain the benefit or coupled simulation fails without a new remedy. Feasibility: existing verified13M checkpoint/CPU and single5090; raw payloads on15TB. Next Experiment: conditional external mechanism/size transfer if gate passes, otherwise isolate a distinct failure mechanism before new architecture. Qualified directions0; overall ACTIVE/UNACHIEVED.'}
 for name,body in texts.items():write(name,'# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n')
 for label,rows in [('field',r['fields']),('rollout',r['rollouts']),('response',r['responses'])]:
  keys=['tag','family','noise_sigma','kind','finite_complete','mse','nrmse','coordinate_mean_nrmse']+(['initial_index','clamp'] if label=='rollout' else ['initial_index'] if label=='response' else [])
  with (OUT/f'{label}_metrics83_v1.csv').open('x',encoding='utf-8',newline='') as f:writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows([{k:z.get(k) for k in keys} for z in rows])
 fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained');colors=['#0072B2','#009E73','#D55E00'];labels=['FIM local','Weak polynomial3','Weak RFF128'];ticks=[];names=[]
 for gi,(family,noise) in enumerate([(f,n) for f in ['fhn','hill'] for n in [0.,.03]]):
  ticks.append(gi*4+1);names.append(f'{family.upper()} / noise{noise:g}')
  for ki,(kind,color,label) in enumerate(zip(kinds,colors,labels)):
   xbase=gi*4+ki
   for ax,rows in zip(axes,[r['fields'],r['rollouts']]):
    selected=[z for z in rows if z['family']==family and z['noise_sigma']==noise and z['kind']==kind];good=[z for z in selected if z['finite_complete'] and z.get('solver',{}).get('success',True)];values=[z['coordinate_mean_nrmse'] for z in good]
    if values:ax.scatter(xbase+np.linspace(-.13,.13,len(values)),values,s=16,color=color,alpha=.6,label=label if gi==0 else None);ax.plot([xbase-.25,xbase+.25],[np.median(values)]*2,color=color,lw=2)
    failed=len(selected)-len(good)
    if failed:ax.text(xbase,-.1,f'{failed} fail',transform=ax.get_xaxis_transform(),ha='center',color=color,fontsize=7)
 for ax,title in zip(axes,['Local field: context-support diagnostic','Coupled trajectories: free and clamped']):ax.set_yscale('log');ax.set_xticks(ticks,names,rotation=15,ha='right',fontsize=8);ax.set_title(title);ax.set_ylabel('Mean coordinate NRMSE (log)');ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=8);figure=storage('artifact')/'composition83_v1.png';fig.savefig(figure,dpi=170);plt.close(fig)
 dump(OUT/'CLOSURE83_V1.json',dict(utc=utc(),decision=decision,results_sha256=sha(OUT/'RESULTS83_V1.json'),verification_sha256=sha(OUT/'VERIFICATION83_V1.json'),decision_sha256=sha(OUT/'DECISION83_V1.json'),figure_path=str(figure),figure_sha256=sha(figure),field_records=48,rollout_records=192,response_records=96,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0))
 with (PROJECT/'CURRENT_RESEARCH_STATUS_V83.md').open('x',encoding='utf-8') as f:f.write('# C83 current research status\n\n'+diagnosis+'\n\n'+table+f'\nDecision: {decision}. No final research direction accepted. Original-budget campaign remains separate. All raw payloads remain on server; GitHub only. Overall ACTIVE/UNACHIEVED.\n')
 files=['PROTOCOL83_V1.md','RESEARCH_BRIEF83_V1.md','LITERATURE_BOUNDARIES83_V1.md','COMPOSITION_AUDIT83_V1.md','README.md']+list(texts);files+=['scripts/'+p.name for p in sorted((ROOT/'scripts').glob('*.py'))];files+=['outputs/'+n for n in ['DATA_MANIFEST83_V1.json','SOURCE_AUDIT83_V1.json','OPENING83_V1.json','RESULTS83_V1.json','VERIFICATION83_V1.json','DECISION83_V1.json','CLOSURE83_V1.json','field_metrics83_v1.csv','rollout_metrics83_v1.csv','response_metrics83_v1.csv','run83_v1_launch.json','run83_v1_exit.json','run83_v1_stdout.log']];dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);pub=[]
 for rel in sorted(files):
  b=(ROOT/rel).read_text(encoding='utf-8-sig').encode();assert all(z.encode() not in b for z in json.loads(os.environ.get('C83_PRIVATE_EXPORT_PATTERNS_JSON','[]'))),(rel,'private export pattern');p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='utf-8'))
 b=figure.read_bytes();rel='figures/composition83_v1.png';p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='binary'));dump(dest/'manifest.json',dict(utc=utc(),files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),large_data_weights_arrays_included=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0));print(json.dumps(dict(status='SNAPSHOT_READY',decision=decision,groups=groups,manifest_sha256=sha(dest/'manifest.json'),files=len(pub))),flush=True)
if __name__=='__main__':main()
