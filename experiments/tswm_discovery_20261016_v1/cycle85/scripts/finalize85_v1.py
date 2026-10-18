"""Assemble immutable server-only scientific evidence; do not infer research completion."""
from common85_v1 import *
import csv,io,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PUB=O/'publication_v1'
METHODS=['frozen','native_frozen_alignment','joint_output_error','polynomial3','rff128']
LABELS=['Frozen','Native align + anchors','Joint output-error','Polynomial ridge','RFF ridge']
COLORS=['#888888','#D55E00','#0072B2','#009E73','#CC79A7']
def scrub(value):
    if isinstance(value,dict):return {k:scrub(v) for k,v in value.items()}
    if isinstance(value,list):return [scrub(v) for v in value]
    if isinstance(value,str):
        norm=value.replace('\\','/')
        for role,root in [('project',str(ROOT)),('artifact',CONFIG['artifact_root']),('data',CONFIG['data_root']),('checkpoint',CONFIG['checkpoint_root']),('cache',CONFIG['cache_root'])]:
            key=root.replace('\\','/').rstrip('/')
            if norm.lower().startswith(key.lower()+'/'):return 'server-'+role+':'+norm[len(key)+1:]
    return value
def put(path,b):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(b.encode() if isinstance(b,str) else b)
def load(path):return json.loads(path.read_text())
def jput(path,obj):put(path,json.dumps(obj,indent=2,allow_nan=False))
def main():
    assert load(O/'WORKER_EXIT85_V1.json')['exit_code']==0
    original=load(O/'RESULT85_V1.json');primary_v=load(O/'VERIFICATION85_V1.json');assert primary_v['status']=='PASS' and primary_v['result_sha256']==sha(O/'RESULT85_V1.json')
    stage_entries=[('primary',O,original,primary_v)]
    caproot=O/'capacity_v2';cap=load(caproot/'RESULT85_V1.json');assert load(caproot/'WORKER_EXIT85_V1.json')['exit_code']==0 and cap['closed_cases']==0
    longroot=O/'source_polish_v3';long=load(longroot/'RESULT85_V1.json');assert load(longroot/'WORKER_EXIT85_V1.json')['exit_code']==0
    long_v=load(longroot/'VERIFICATION85_V3.json');assert long_v['status']=='PASS' and long_v['result_sha256']==sha(longroot/'RESULT85_V1.json')
    stage_entries.append(('source_polish_v3',longroot,long,long_v))
    cases=[];allrows=[]
    for stage,path,result,verify in stage_entries:
        for p in sorted(path.glob('CASE85_*_V1.json')):
            r=load(p);r['stage']=stage;cases.append(r)
            for row in r['rows']:allrows.append(dict(family=r['family'],seed=r['seed'],case=r['case'],stage=stage,**row))
    assert len(cases)==original['closed_cases']+long['closed_cases']
    source_records=[dict(stage='primary',**r) for r in original['sources']]+[dict(stage='capacity_v2',**r) for r in cap['sources']]+[dict(stage='source_polish_v3',**r) for r in long['sources']]
    passed=sum(c['selected_existing_suffices'] for c in cases)
    table=['| Family | Seed | Change | Validation-selected existing repair | H128 NRMSE | Response error | Gate |','|---|---:|---|---|---:|---:|---|']
    for r in sorted(cases,key=lambda q:(q['family'],q['seed'],SHIFT_CASES.index(q['case']))):
        name=r['best_existing_by_validation'];s=r['summary'][name]
        table.append(f"| {r['family']} | {r['seed']} | {r['case']} | {name} | {s['128']['median_nrmse']:.6g} | {s['response']['median_error']:.6g} | {'PASS' if r['selected_existing_suffices'] else 'UNRESOLVED'} |")
    summary=dict(utc=utc(),completed_cases=len(cases),planned_cases=16,selected_existing_satisfies_cases=passed,source_training_records=len(source_records),original_source_runs=4,capacity_source_runs=2,source_continuations=len(long['sources']),target_neural_candidates=sum(len(list(path.glob('FIT85_*_V1.json'))) for _,path,_,_ in stage_entries),ordinary_target_coefficient_fits=len(cases)*24,ordinary_source_coefficient_fits=len(cases)*2,eval_prediction_trajectories=len(cases)*5*16,eval_action_response_pairs=len(cases)*5*8,metric_rows=len(allrows),source_hashes=primary_v['native_source_hashes'],verified_prediction_scalars=primary_v['total_prediction_scalars']+long_v['total_prediction_scalars'],new_method_tested=False,qualified_final_directions=0,scope='Two low-dimensional fully observed controlled systems; same-information calibration access; distinct neural capacities/costs and source-training controls disclosed')
    reports=RAW/'publication_figures_v1';reports.mkdir(parents=True,exist_ok=False)
    figure,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ri,family in enumerate(['duffing','brusselator']):
        group=[r for r in cases if r['family']==family]
        for col,key in enumerate(['128','response']):
            ax=axes[ri,col];finite_scores=[]
            for r in group:
                for method in METHODS:
                    v=r['summary'][method][key];value=v['median_nrmse'] if key=='128' else v['median_error']
                    if v['failed']==0 and np.isfinite(value) and value<1e12:finite_scores.append(max(value,1e-8))
            cap_plot=10**np.ceil(np.log10(max(finite_scores+[1.])*1.2))
            for mi,method in enumerate(METHODS):
                for r in group:
                    v=r['summary'][method][key];val=v['median_nrmse'] if key=='128' else v['median_error'];bad=v['failed']>0 or val>=1e12
                    seed_offset=-.015 if r['seed']==SEEDS[0] else .015
                    x=SHIFT_CASES.index(r['case'])+(mi-2)*.12+seed_offset
                    ax.scatter(x,cap_plot if bad else max(val,1e-8),c=COLORS[mi],marker='^' if bad else ('o' if r['seed']==SEEDS[0] else 'x'),s=36,alpha=.85,label=LABELS[mi] if ri==0 and col==0 and r is group[0] else None)
            ax.axhline(.2 if key=='128' else .3,color='#444444',ls=':',lw=1)
            ax.set_yscale('log');ax.set_ylim(min(finite_scores+[.01])*.6,cap_plot*1.5);ax.grid(axis='y',alpha=.2)
            ax.set_title(family.capitalize()+(' / H128 trajectory' if key=='128' else ' / action response'))
            ax.set_xticks(range(4));ax.set_xticklabels(['None','Observation','Dynamics','Joint']);ax.set_ylabel('NRMSE' if key=='128' else 'Normalized response error')
    handles,labels=axes[0,0].get_legend_handles_labels();figure.legend(handles,labels,loc='upper center',ncol=5,bbox_to_anchor=(.5,1.01),frameon=False,fontsize=9)
    figure.suptitle('Existing repairs under anchored shifts',y=1.06,fontsize=14)
    figure.text(.5,-.015,'Dots / crosses: two seeds. Triangles: rollout failures at plotting cap. Dotted lines: frozen diagnostic gates. No matched-compute claim.',ha='center',fontsize=9)
    figure.tight_layout();trajectory_fig=reports/'anchored_repairs85_v1.png';figure.savefig(trajectory_fig,dpi=160,bbox_inches='tight');figure.savefig(reports/'anchored_repairs85_v1.svg',bbox_inches='tight');plt.close(figure)
    sourcefig,sourceaxes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    slabels=[]
    for j,r in enumerate(source_records):
        slabels.append(f"{r['family'][:3]} {str(r['seed'])[-1]}\n{r['stage']}")
        for k,h in enumerate(['32','128']):sourceaxes[k].scatter(j,r['selected']['median'+h],marker='o' if r['qualified'] else 'x',color='#0072B2' if r['qualified'] else '#D55E00',s=45)
    for k,threshold in enumerate([.15,.35]):sourceaxes[k].axhline(threshold,color='black',ls=':',lw=1);sourceaxes[k].set_ylabel('Source median H'+('32' if k==0 else '128')+' NRMSE');sourceaxes[k].grid(axis='y',alpha=.2)
    sourceaxes[1].set_xticks(range(len(slabels)));sourceaxes[1].set_xticklabels(slabels,fontsize=8)
    sourcefig.suptitle('Source qualifications: original failures retained');sourcefig.text(.5,-.01,'Blue: qualified. Orange: unqualified. Stages change capacity / source rollout training; they are not an isolated causal ablation.',ha='center',fontsize=9)
    sourcefig.tight_layout();source_fig=reports/'source_controls85_v1.png';sourcefig.savefig(source_fig,dpi=160,bbox_inches='tight');sourcefig.savefig(reports/'source_controls85_v1.svg',bbox_inches='tight');plt.close(sourcefig)
    PUB.mkdir(exist_ok=False)
    for p in sorted(C.rglob('*')):
        if p.is_file() and 'outputs' not in p.relative_to(C).parts and '__pycache__' not in p.parts and p.suffix in ['.md','.py','.json','.ps1']:put(PUB/p.relative_to(C),p.read_bytes())
    for p in reports.iterdir():put(PUB/'figures'/p.name,p.read_bytes())
    jput(PUB/'evidence'/'SUMMARY85_V1.json',summary);jput(PUB/'evidence'/'SOURCE_CONTROLS85_PUBLIC_V1.json',scrub(source_records))
    public_cases=[{k:v for k,v in r.items() if k!='rows'} for r in cases];jput(PUB/'evidence'/'CASE_SUMMARIES85_PUBLIC_V1.json',scrub(public_cases))
    jput(PUB/'evidence'/'VERIFICATION85_PRIMARY_V1.json',scrub(primary_v));jput(PUB/'evidence'/'VERIFICATION85_SOURCE_POLISH_V3.json',scrub(long_v))
    seals=[]
    for stage,path,_,_ in stage_entries:
        for p in sorted(path.glob('SELECTION_SEALED85_*_V1.json')):seals.append(dict(stage=stage,original_seal_sha256=sha(p),**load(p)))
    jput(PUB/'evidence'/'PRE_EVAL_SELECTIONS85_PUBLIC_V1.json',scrub(seals))
    csvtext=io.StringIO();writer=csv.DictWriter(csvtext,fieldnames=list(allrows[0]) if allrows else ['family','seed','case','stage','method','kind','horizon','path','score','failed']);writer.writeheader();writer.writerows(allrows);put(PUB/'evidence'/'metric_rows85_v1.csv',csvtext.getvalue())
    report=f'''# C85 results: existing repairs in an anchored controlled subcase

Completed cases: {len(cases)} /16. A repair chosen using target VAL alone satisfies the frozen H128 trajectory and action-response criteria in {passed}/{len(cases)} completed cases. Qualified new research directions:0. No new method was trained.

{chr(10).join(table)}

![All methods and both seeds](figures/anchored_repairs85_v1.png)

The full five-method table is machine-readable in evidence/CASE_SUMMARIES85_PUBLIC_V1.json; all {len(allrows)} metric rows are retained. There are {summary['eval_prediction_trajectories']} independent model/plan rollouts and {summary['eval_action_response_pairs']} scored action contrasts, not that many independent scientific tasks. H32 scores reuse trajectory prefixes and the two plans share their first32 steps.

## Source training and controls

The original64-wide/2-latent sources qualified on Duffing but not Brusselator. Larger128-wide/4-latent source fits also failed the unchanged short-horizon gate. A separately documented source-only32-step output-training control used the same data and unchanged qualification thresholds before any Brusselator EVAL generation. All ancestor failures and additional training costs are retained. Changing capacity, learning rate, optimizer state and source loss horizon is not an isolated causal ablation of any one factor.

![Source qualification controls](figures/source_controls85_v1.png)

There are {len(source_records)} source training records: four original fits, two capacity fits and {len(long['sources'])} source continuations, over four family/seed cells. Actual selected and completed update counts and parameter counts are in evidence/SOURCE_CONTROLS85_PUBLIC_V1.json. The source sizes and objectives differ by family; this is not a matched-compute architecture comparison.

## Verification and interpretation

All selected neural/dictionary predictions were replayed; {summary['verified_prediction_scalars']} prediction scalars were checked, scores independently recomputed, selected ridge coefficients checked with augmented least squares, and physical trajectories checked with independently implemented DOP853 dynamics. No model was reselected on EVAL. Both verification records are published.

The ordinary controls use a low-dimensional fully observed state, and degree-three features are favorable to these dynamics. The sensor mapping is nonlinear and withheld from models, but remains invertible. These conditions differ materially from hidden-state, high-dimensional and universal cross-family modeling. Neither successful ordinary repairs nor a poor neural score establishes a universal result. A native aligner failing after actual dynamics change is outside its frozen-dynamics premise.

This screen does not establish a need for a new calibration mechanism in cases passed by existing repairs. Unresolved cases require source, coverage, observability, objective and model-selection analysis before becoming gap evidence. Overall research goal remains ACTIVE / UNACHIEVED.
'''
    put(PUB/'experimental_results.md',report)
    negative=f'''# C85 negative evidence and limits

* Validation-selected existing repairs pass both frozen criteria in {passed}/{len(cases)} completed cases. Any such case cannot support a claim that a distinct new mechanism is necessary.
* Source-only training/capacity/rollout controls were required before interpreting Brusselator transfer. Initial source qualification failures are implementation/training evidence, not a research gap.
* Neural trajectory and response rankings may differ; an ordinary repair succeeding on both prevents that discrepancy alone from establishing novelty.
* Source architectures, updated parameter counts and costs differ. Target methods have the same accessible calibration data, but are not compute/parameter matched. No efficiency or architecture superiority claim is made.
* Two seeds and two small controlled mechanisms do not establish stable real-world, partially observed or universal-model generalization. The polynomial dictionary has structural advantages here; RFF and ordinary joint adaptation remain independent controls.
* All predeclared completed predictions and failures are retained. No EVAL hyperparameter tuning, new method, paper reproduction success rate or accepted final research direction is claimed.
'''
    put(PUB/'negative_evidence.md',negative)
    readme='''# C85: existing repairs under anchored observation/dynamics changes

Read experimental_results.md, negative_evidence.md and final_research_direction.md. This is an original controlled falsification screen using existing methods, not a new-method paper or original robot-benchmark reproduction.

For a fresh reproduction, configure STORAGE_LOCATION_V1.json in the project parent with external artifact_root, data_root, checkpoint_root and cache_root directories. Use Python3.10, Torch2.8, NumPy1.26 and SciPy1.15; Matplotlib is needed only for export. Run scripts/bootstrap_source85_v1.py, then scripts/qualify85_v1.py, then scripts/capture85_v1.py. The separate capacity and source-polish captures reproduce the documented source-only repairs. Their guards enforce the recorded qualification branches; a different runtime may qualify a source earlier, in which case do not override a gate merely to force the same branch. Run verify85_v1.py for the original stage and verify85_v3.py source_polish for the final source-polish stage. All outputs/artifact directories must be fresh; existing evidence is protected against overwriting.

The frozen protocol and both source amendments disclose the information, update budgets and checks. Raw source/target arrays, selected weights, prediction arrays and raw logs remain on the server. Their hashes and compact verified metrics are published; no local result-payload synchronization was performed. Source and target neural sizes differ between families, so no matched-compute claim is supported.
'''
    put(PUB/'README.md',readme)
    status=f'''# C85 current research judgment

The explicitly anchored calibration screen completed {len(cases)}/16 cases. Validation-selected existing repairs satisfy trajectory and action-response gates in {passed}/{len(cases)} completed cases. This supplies negative evidence for needing a new mechanism in those low-dimensional fully observed settings, not a universal theorem about adaptation.

Initial source qualification failures, a capacity repair and a source-only longer-rollout control are retained. Independent verification checked {summary['verified_prediction_scalars']} prediction scalars. No new method was tested, and no final research direction is accepted. Overall objective: ACTIVE / UNACHIEVED. Do not reinterpret source underfitting, insufficient information or expected frozen-dynamics failure as novelty.

All raw data/weights/arrays/logs stay on the server. Code, compact evidence and figures go to GitHub. C71's unchanged original-budget campaign remains separate; its completion is not implied here.
'''
    put(ROOT/'CURRENT_RESEARCH_STATUS_V85.md',status)
    files=[]
    for p in sorted(PUB.rglob('*')):
        if not p.is_file():continue
        b=p.read_bytes();files.append(dict(path=p.relative_to(PUB).as_posix(),server_snapshot_file=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),encoding='binary' if p.suffix=='.png' else 'utf-8'))
    write(PUB/'manifest.json',dict(schema='C85_PUBLICATION_V1',utc=utc(),files=files,summary=summary,local_result_payload_files_written=0))
    closure=dict(status='READY_FOR_PUBLICATION_NOT_RESEARCH_COMPLETION',utc=utc(),manifest_sha256=sha(PUB/'manifest.json'),files=len(files),summary=summary,figures=[dict(path=str(p),sha256=sha(p)) for p in [trajectory_fig,source_fig]],local_result_payload_files_written=0)
    write(O/'CLOSURE85_V1.json',closure);print(json.dumps(closure,indent=2),flush=True)

if __name__=='__main__':main()
