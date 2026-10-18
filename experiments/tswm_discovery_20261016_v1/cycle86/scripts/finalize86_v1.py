from common86_v1 import *
import csv,io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PUB=O/'publication_v1';FIG=RAW/'publication_figures_v1'
ARMS=['whole_filter','resampled_ic','prefix_all'];LABELS=['Whole-trajectory filter','Same mechanisms / resampled IC','Unselected finite prefixes'];COLORS=['#D55E00','#0072B2','#009E73']
def read(p):return json.loads(p.read_text())
def put(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(data.encode('utf-8') if isinstance(data,str) else data)
def scrub(v):
    if isinstance(v,dict):return {k:scrub(x) for k,x in v.items()}
    if isinstance(v,list):return [scrub(x) for x in v]
    if isinstance(v,str):
        n=v.replace('\\','/')
        for name,p in [('project',ROOT),('artifact',CONFIG['artifact_root']),('data',CONFIG['data_root']),('checkpoint',CONFIG['checkpoint_root']),('cache',CONFIG['cache_root'])]:
            prefix=str(p).replace('\\','/').rstrip('/')
            if n.lower().startswith(prefix.lower()+'/'):return 'server-'+name+':'+n[len(prefix)+1:]
    return v
def jput(p,v):put(p,json.dumps(scrub(v),indent=2,allow_nan=False))
def savefig(fig,name):
    p=FIG/(name+'.png');fig.savefig(p,dpi=160,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig);return p
def main():
    r=read(O/'RESULT86_V1.json');v=read(O/'VERIFICATION86_V1.json');p=read(O/'paired_v1'/'RESULT86_PAIRED_V1.json');pv=read(O/'paired_v1'/'VERIFICATION86_PAIRED_V1.json');j=read(O/'JOINT_DEPENDENCE86_V1.json');jv=read(O/'JOINT_VERIFICATION86_V1.json')
    for record,path in [(v,O/'RESULT86_V1.json'),(pv,O/'paired_v1'/'RESULT86_PAIRED_V1.json'),(jv,O/'JOINT_DEPENDENCE86_V1.json')]:assert record['status']=='PASS' and record['result_sha256']==sha(path)
    conditions=[q for population in r['populations'] for q in population['conditions']];FIG.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for threshold,color in [(None,'#777777'),(3.,'#D55E00'),(6.,'#0072B2')]:
        for col,key in enumerate(['fraction','positive_growth_fraction']):
            ax=axes[col]
            for seed in SEEDS:
                rows=sorted([q for q in conditions if q['seed']==seed and q['threshold']==threshold],key=lambda z:z['paths']);ax.plot([1,2,3],[q['distribution'][key] for q in rows],marker='o',color=color,alpha=.55,label=('No threshold' if threshold is None else 'Threshold '+str(int(threshold))) if seed==SEEDS[0] else None)
            ax.set_xticks([1,2,3]);ax.set_xticklabels(['1','4','16']);ax.set_xlabel('Sibling paths per mechanism');ax.set_ylim(0,1.05);ax.grid(alpha=.2);ax.set_ylabel('Retained mechanism fraction' if col==0 else 'Positive-growth mechanism fraction')
    axes[0].legend(frameon=False,fontsize=9);fig.suptitle('Whole-realization retention changes the mechanism population');fig.text(.5,-.03,'Same4096 mechanisms and common initial paths per seed. Three independent pools. Exact flows; no nonfinite paths.',ha='center',fontsize=9);fig.tight_layout();plots=[savefig(fig,'sampler_population86_v1')]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    ordinary=next(q for q in p['results'] if q['label']=='linear_2026861')['summary']['all'];persistence=next(q for q in p['results'] if q['label']=='persistence')['summary']['all']
    for row,kind in enumerate(['generator','direct']):
        for col,key in enumerate(['mean_mse','mean_nrmse']):
            ax=axes[row,col]
            for index,arm in enumerate(ARMS):
                values=[q['summary']['all'][key] for q in p['results'] if q.get('architecture')==kind and q.get('arm')==arm];ax.scatter(np.arange(3)*.04+index-.04,values,color=COLORS[index],s=35);ax.plot([index-.2,index+.2],[np.mean(values)]*2,color=COLORS[index],lw=2)
            ax.axhline(ordinary[key],color='black',ls='--',lw=1,label='Local ridge ID');ax.axhline(persistence[key],color='#888888',ls=':',lw=1,label='Persistence');ax.set_xticks(range(3));ax.set_xticklabels(['Whole filter','Resampled IC','Finite prefixes']);ax.set_yscale('log');ax.set_ylabel('Mean MSE' if col==0 else 'Mean trajectory NRMSE');ax.set_title(kind.capitalize()+' predictor');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8);fig.suptitle('Paired factual learning: gains depend on the metric');fig.text(.5,-.015,'Points: three paired training populations/seeds. One shared new evaluation pool (1024 mechanisms). Equal1024-window and3000-update budgets.',ha='center',fontsize=8.5);fig.tight_layout();plots.append(savefig(fig,'paired_learning86_v1'))
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,kind in enumerate(['generator','direct']):
        for row,key in enumerate(['mean_mse','mean_nrmse']):
            ax=axes[row,col]
            for index,arm in enumerate(ARMS):
                values=[]
                for seed in SEEDS:
                    vv=[next(q[key] for q in j['results'] if q['population']==pop and q['label']==f'{seed}_{kind}_{arm}') for pop in ['retained','crossed']];values.append(vv);ax.plot([0,1],vv,color=COLORS[index],alpha=.4,marker='o',lw=.8)
                ax.plot([0,1],np.mean(values,axis=0),color=COLORS[index],lw=2,label=LABELS[index])
            values=[next(q[key] for q in j['results'] if q['population']==pop and q['label']=='linear_2026861') for pop in ['retained','crossed']];ax.plot([0,1],values,color='black',ls='--',label='Local ridge ID');ax.set_xticks([0,1]);ax.set_xticklabels(['Retained pairing','Permuted pairing']);ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.set_ylabel('Mean MSE' if row==0 else 'Mean trajectory NRMSE')
            if row==0:ax.set_title(kind.capitalize()+' predictor')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.04),frameon=False,fontsize=9);fig.suptitle('New query population: fixed marginals, changed pairing',y=1.1);fig.text(.5,-.01,'Same1024 mechanisms and same1024 initial states in both queries; shared noise. Later trajectory support is not matched.',ha='center',fontsize=9);fig.tight_layout();plots.append(savefig(fig,'crossed_queries86_v1'))
    table=['| Architecture | Train seed | Whole MSE | Resampled-IC MSE | Prefix MSE | Whole mean NRMSE | Resampled mean NRMSE | Prefix mean NRMSE |','|---|---:|---:|---:|---:|---:|---:|---:|'];effects=[]
    for kind in ['generator','direct']:
        for seed in SEEDS:
            rows=[next(q['summary']['all'] for q in p['results'] if q['label']==f'{seed}_{kind}_{arm}') for arm in ARMS];table.append('| '+kind+' | '+str(seed)+' | '+' | '.join(f"{q[key]:.7g}" for key in ['mean_mse','mean_nrmse'] for q in rows)+' |');effects.append(dict(architecture=kind,seed=seed,resampled_vs_whole_mse_reduction=1-rows[1]['mean_mse']/rows[0]['mean_mse'],resampled_vs_whole_mean_nrmse_reduction=1-rows[1]['mean_nrmse']/rows[0]['mean_nrmse'],prefix_vs_whole_mse_reduction=1-rows[2]['mean_mse']/rows[0]['mean_mse'],prefix_vs_whole_mean_nrmse_reduction=1-rows[2]['mean_nrmse']/rows[0]['mean_nrmse']))
    cap=[]
    for fit in p['fits']:
        old=min(z['val_mse'] for z in fit['trace'] if z['step']<=2400);late=min(z['val_mse'] for z in fit['trace'] if z['step']>2400);cap.append(dict(seed=fit['seed'],architecture=fit['architecture'],arm=fit['arm'],last600_val_improvement_fraction=(old-late)/old,improving_more_than_half_percent=(old-late)/old>.005))
    summary=dict(utc=utc(),source_sampler_commit=r['source']['commit'],paired_training_fits=len(p['fits']),source_qualified=sum(q['source_qualified'] for q in p['fits']),source_qualification_is_only_persistence_control=True,update_cap_diagnostic=cap,unselected_eval_mechanisms=1024,crossed_eval_mechanisms=1024,eval_prediction_scalars=pv['prediction_scalars']+jv['prediction_scalars'],sampler_trajectory_scalars=v['verified_trajectory_scalars'],effects=effects,ordinary_baseline_effective_unique_configs=1,released_checkpoint_tested=False,new_method_validated=False,accepted_research_directions=0,candidate_status='NEEDS_NATIVE_MODEL_EVIDENCE_AND_NOVELTY_DIFFERENTIATION',overall_goal='ACTIVE_UNACHIEVED',local_result_payload_files_written=0)
    PUB.mkdir(exist_ok=False)
    for file in C.rglob('*'):
        if file.is_file() and 'outputs' not in file.relative_to(C).parts and '__pycache__' not in file.parts and file.suffix in ['.py','.md','.json']:put(PUB/file.relative_to(C),file.read_bytes())
    for file in FIG.iterdir():put(PUB/'figures'/file.name,file.read_bytes())
    for name,record in [('SUMMARY86_V1.json',summary),('SAMPLER_RESULT86_PUBLIC_V1.json',r),('SAMPLER_VERIFICATION86_V1.json',v),('PAIRED_RESULT86_PUBLIC_V1.json',p),('PAIRED_VERIFICATION86_V1.json',pv),('JOINT_DEPENDENCE86_PUBLIC_V1.json',j),('JOINT_VERIFICATION86_V1.json',jv),('SOURCE_PUBLIC_HTTPS86_V1.json',read(O/'SOURCE_PUBLIC_HTTPS86_V1.json')),('PRE_EVAL_SELECTION86_PUBLIC_V1.json',read(O/'paired_v1'/'SELECTION_SEALED86_V1.json'))]:jput(PUB/'evidence'/name,record)
    for name,rows in [('sampler_conditions86_v1.csv',conditions),('paired_effects86_v1.csv',effects),('joint_query_results86_v1.csv',j['results'])]:
        flat=[{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in rows];stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat);put(PUB/'evidence'/name,stream.getvalue())
    allrows=[]
    for population,record in [('primary',p),('joint',j)]:
        d=np.load(record['eval_array_path'] if population=='primary' else record['array_path'],allow_pickle=False)
        for q in record['results']:
            prefix=q['label'] if population=='primary' else q['population']+'_'+q['label']
            for i,(m,n,f) in enumerate(zip(d[prefix+'_mse'],d[prefix+'_nrmse'],d[prefix+'_failed'])):allrows.append(dict(evaluation=population,query_population='unselected' if population=='primary' else q['population'],model=q['label'],mechanism_index=i,mse=float(m),nrmse=float(n),failed=bool(f)))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows);put(PUB/'evidence'/'per_mechanism_scores86_v1.csv',stream.getvalue())
    report=f'''# C86: a sampling effect is verified; a new world-model method is not

The pinned whole-realization retention rule changes the mechanism population even when every path is finite. In a paired small-model study, resampling initial conditions while holding selected mechanism IDs fixed improves unselected-query raw MSE across both tested architectures and all three training seeds. That benefit is not universal: retained-cohort errors can rise, and on crossed queries the direct predictor's normalized errors worsen despite lower raw MSE. Ordinary local ridge identification is stronger than the neural models on the unselected pool. Final accepted directions:0.

## Sampler intervention

![Sampler population](figures/sampler_population86_v1.png)

Three pools contain4096 mechanisms each. Nine paired conditions vary sibling count1/4/16 and thresholdNone/3/6, with identical underlying mechanisms and prefix paths. All physical flows are finite, so the threshold effect is separate from numerical failure. The actual retained PathGenerator.solve_equations body is executed, but its solver is explicitly replaced by exact linear flow. This does not reproduce the full FIM generator or its checkpoint pretraining population. Public source SHA and method AST are independently verified.

For independent initial states, the accepted mechanism density is proportional to its original density times the single-path acceptance probability raised to the sibling count. This elementary selection identity is not novel theory. A weak snapshot-only probe shows a small selected-only association in some conditions, not a trained world model's actual shortcut.

## Matched-budget factual learning

![Paired learning](figures/paired_learning86_v1.png)

{chr(10).join(table)}

All18 fits beat persistence on their own source VAL population. This is a weak competence check: it does not establish convergence or equal architecture quality. {sum(x['improving_more_than_half_percent'] for x in cap)}/18 fits improve their source VAL minimum by more than0.5% in the last600 updates. Selected/completed costs and every checkpoint trace are retained. The generator architecture has a family-specific symmetric-linear transition assumption and still trails the direct predictor; do not turn this into a general architecture ranking.

Both architectures receive the same1024 retained short windows and3000 updates per arm, with matched initialization and index schedules. The resampled-IC arm preserves whole-filter mechanism IDs and changes their initial-condition pairing. Prefix retention changes the mechanism population as well. Simulator-attempt budgets and physical state distributions are not matched. New evaluation seed2026864 contains1024 common mechanisms, including701 that satisfy the original complete-path filter and323 that do not. Those subgroup labels use privileged future information for diagnosis only; they are never fitter inputs or selection criteria.

The ordinary control's unselected mean MSE is {ordinary['mean_mse']:.8g}, with mean NRMSE {ordinary['mean_nrmse']:.8g}. All three source-selected ridge values coincide, so they form one effective prediction baseline, not three independent successes. Prefix retention improves raw MSE but does not consistently improve average per-trajectory NRMSE; both metrics and all subgroups are retained.

## Fixed marginals, changed pairing

![Crossed queries](figures/crossed_queries86_v1.png)

The second new seed2026865 supplies1024 mechanisms and their original accepted initial states. Permuting those same initial states keeps both marginals exactly fixed and changes their joint association. Identical noise is reused. Later trajectory support and available information can differ, so this is not a full-support match. No model is retrained or reselected. Lower crossed raw MSE for resampled-IC training accompanies worse direct-predictor NRMSE; this tradeoff prevents a universal robustness claim.

## Verification, scope and decision

Independent NumPy networks replay all neural predictions; independent least squares replays ordinary controls. All198 source-checkpoint losses and data/selection hashes are checked. Physical paths are compared with SciPy matrix exponentials. Verified prediction scalars:{summary['eval_prediction_scalars']}; sampler trajectory scalars:{summary['sampler_trajectory_scalars']}. These are numerical-check counts, not independent scientific tasks. Three paired training seeds share each of two evaluation pools. Individual trajectories do not turn this into multiple unrelated datasets.

No released foundation-model checkpoint is tested in this cycle, and the exact FIM-ODE pretraining-data lineage remains unverified. The inspected source is an SDE/ODE data-generation sampler, not proof of a particular checkpoint's history. Therefore no original-paper error, causal attribution of C83 failures, posterior-calibration theorem, new architecture requirement or final research direction is claimed.

The candidate is now specifically about training-sampler selection and mechanism/initial-state dependence, rather than generic poor forecasting. Censored identification, validity correction in simulation-based inference, initial-state resampling and controlled-generator audits are prior art. A possible new contribution would need actual native world-model evidence of a non-ignorable selection/coarsening interaction and a robust correction under clear budget constraints. Read PRIOR_ART_AND_REPAIRS86_V1.md and final_research_direction.md before extending it. Overall research objective remains ACTIVE / UNACHIEVED.
'''
    put(PUB/'experimental_results.md',report)
    negative='''# Negative evidence and unresolved limits

* Ordinary local ridge identification outperforms all18 neural fits on the unselected evaluation population. No new predictor is needed to solve this family.
* Raw MSE improvements do not establish uniform normalized-error improvements. The direct predictor's crossed-query NRMSE is worse after IC resampling, and selected-cohort accuracy can also worsen. These rows must not be omitted.
* Source qualification is only a persistence comparison. The generator model is much weaker than the direct model, and some fits still improve near the update cap. Adequate convergence and architecture parity are not established.
* Selection deliberately changes the effective training distribution. An unselected downstream population differs from it; this is not automatically a defect or novelty. The matched-marginal crossed control narrows one confound but does not match future state support.
* Snapshot association is small and disappears in some sibling-count conditions. It does not prove that a native foundation model uses snapshot shortcuts.
* Only one symmetric-linear autonomous family and two small existing architectures are tested. No nonlinear controlled or real system, universal pretraining, planning, action effect or partial-observation benchmark result is established.
* The exact sampler is pinned; its use for a specific released checkpoint is not. C83's FIM-ODE failures remain causally unexplained by this cycle.
* Selection/censoring statistics, validity correction, initial-state augmentation and data audits are established. The candidate still needs a concrete differentiator and native-model evidence. Accepted final directions0.
''';put(PUB/'negative_evidence.md',negative)
    put(PUB/'README.md','''# C86 controlled pretraining-sampler audit

Start with experimental_results.md and final_research_direction.md. This contains a sampler-method diagnostic and small paired factual-learning controls, not a reproduced foundation-model result.

Use Python3.10, Torch2.8, NumPy1.26, SciPy1.15 and Matplotlib in a task-local environment. Configure the project-parent STORAGE_LOCATION_V1.json with artifact_root/data_root/checkpoint_root/cache_root on the external disk. Put the pinned OpenFIM source tree under cache_root/cycle80/OpenFIM_cee2bb53_v1. Run sample86_v1.py, then capture86_v1.py (or train86_v1.py directly), then joint_dependence86_v1.py. The protocols were frozen at their documented stages. Fresh output/artifact/checkpoint folders are mandatory; existing evidence is never overwritten.

Run verify86_v1.py if public HTTPS is reachable from the server. The retained v2 verifier uses the independently verified public-source receipt described in PRIOR_ART_AND_REPAIRS86_V1.md, with unchanged scientific checks. Run verify_paired86_v1.py and verify_joint86_v1.py, then finalize86_v1.py. Scripts are in scripts/. All models use CPU two threads; the separate C71 campaign is not altered.

All raw populations, observations, checkpoints, predictions and logs remain on the server's external disk. Compact verified scores, source/protocol hashes, three figures and code are published. No local result-payload synchronization occurred.
''')
    status='''# Current research status — C86

Overall objective: ACTIVE / UNACHIEVED. Final accepted directions:0.

C85's calibration-method candidate is rejected in its controlled subcase: an ordinary composition passes all16 Brusselator conditions. C86 develops a different, source-grounded candidate: synthetic trajectory selection changes the mechanism population and initial-state/mechanism association. A pinned sampler diagnostic and18 matched-budget small-model fits are complete and independently verified. Resampling ICs helps raw error, but metrics trade off and ordinary identification remains stronger. No released checkpoint defect or final novelty is established.

Read cycle86/experimental_results.md, cycle86/negative_evidence.md, cycle86/PRIOR_ART_AND_REPAIRS86_V1.md and cycle86/final_research_direction.md. Next decisive work is native-model training-sampler provenance and a competent paired nonlinear controlled-family comparison, not another generic architecture proposal. C71's original-budget campaign remains separate and is not declared complete here.

Raw data/models/predictions/logs remain on the external server disk. GitHub is the only result-publication destination; no local payload synchronization.
''';put(ROOT/'CURRENT_RESEARCH_STATUS_V86.md',status)
    files=[]
    for file in sorted(PUB.rglob('*')):
        if file.is_file():
            b=file.read_bytes();files.append(dict(path=file.relative_to(PUB).as_posix(),server_snapshot_file=str(file),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),encoding='binary' if file.suffix=='.png' else 'utf-8'))
    write(PUB/'manifest.json',dict(schema='C86_PUBLICATION_V1',utc=utc(),files=files,summary=summary,local_result_payload_files_written=0));closure=dict(status='READY_FOR_PUBLICATION_NOT_RESEARCH_COMPLETION',utc=utc(),manifest_sha256=sha(PUB/'manifest.json'),files=len(files),summary=summary,figures=[dict(path=str(file),sha256=sha(file)) for file in plots],local_result_payload_files_written=0);write(O/'CLOSURE86_V1.json',closure);print(json.dumps({k:v for k,v in closure.items() if k!='summary'},indent=2),flush=True)
if __name__=='__main__':main()
