from common85_v1 import *
import csv,io
import matplotlib.pyplot as plt

def load(path):return json.loads(path.read_text())
def collect_controls():
    verify=load(O/'CONTROL_VERIFICATION85_V1.json');cv=load(O/'COMPOSED_VERIFICATION85_V1.json');assert verify['status']==cv['status']=='PASS'
    wroot=O/'weak_controls_v1';aroot=O/'attribution_v1';wr=load(wroot/'RESULT85_WEAK_V1.json');ar=load(aroot/'RESULT85_ATTRIBUTION_V1.json')
    assert sha(wroot/'RESULT85_WEAK_V1.json')==verify['weak_result_sha256'] and sha(aroot/'RESULT85_ATTRIBUTION_V1.json')==verify['attribution_result_sha256']
    groups={'weak':(wroot,'WEAK'),'attribution':(aroot,'ATTRIBUTION'),'composed1':(O/'composed_control_v1','COMPOSED'),'composed2':(O/'composed_control_v2','COMPOSED')};records={};seals=[];rows=[]
    for group,(folder,prefix) in groups.items():
        records[group]=[]
        for p in sorted(folder.glob(f'CASE85_{prefix}_*_V1.json')):
            r=load(p);records[group].append(r)
            if prefix=='COMPOSED':assert sha(p)==cv['case_file_sha256'][str(p.relative_to(O))]
            for row in r['rows']:rows.append(dict(stage=group,seed=r['seed'],case=r['case'],**row))
        for p in sorted(folder.glob(f'SELECTION_SEALED85_{prefix}_*_V1.json')):seals.append(dict(stage=group,original_seal_sha256=sha(p),**load(p)))
    assert [len(records[k]) for k in groups]==[8,12,8,8]
    composed=records['composed1']+records['composed2'];passed=sum(r['summary']['ordinary_val128']['suffices'] for r in composed);joint=[r for r in composed if r['case']=='joint']
    summary=dict(weak_cases=8,attribution_cases=12,composed_cases=16,ordinary_composed_val128_passed=passed,ordinary_composed_joint_val128_passed=sum(r['summary']['ordinary_val128']['suffices'] for r in joint),composed_new_eval_seed=2026854,control_verified_prediction_scalars=verify['weak']['prediction_scalars']+verify['attribution']['prediction_scalars']+cv['prediction_scalars'],control_metric_rows=len(rows),new_method_validated=False,qualified_final_directions=0,candidate_decision='REJECT_CALIBRATION_NECESSITY_IN_THIS_CONTROLLED_SUBCASE')
    assert len(joint)==4
    return dict(records=records,seals=seals,rows=rows,verification=verify,composed_verification=cv,coverage=load(O/'COVERAGE85_V1.json'),summary=summary)

def plot_controls(reports,controls):
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    groups=[('attribution','short_full_field_val_long','Short / full field','#D55E00'),('attribution','short_affine_residual_val_long','Short / affine residual','#E69F00'),('attribution','long_full_field_val_long','Long / full field','#CC79A7'),('attribution','long_affine_residual_val_long','Long / affine residual','#0072B2'),('composed','ordinary_val128','Long / calibration + residual','#009E73'),('composed','oracle_val128','Privileged sensor family','#777777')]
    cases=dict(controls['records']);cases['composed']=cases['composed1']+cases['composed2']
    for col,change in enumerate(['dynamics','joint']):
        for row,(key,threshold) in enumerate([('median128',.2),('median_response',.3)]):
            ax=axes[row,col]
            for j,(stage,label,name,color) in enumerate(groups):
                rr=sorted([r for r in cases[stage] if r['case']==change],key=lambda r:r['seed'])
                for k,r in enumerate(rr):
                    s=r['summary'][label];value=s[key];bad=s['failed128' if row==0 else 'failed_response']>0 or value>=1e12
                    ax.scatter(r['seed']-2026850+(j-2.5)*.09,3. if bad else value,marker='^' if bad else ('D' if stage=='composed' else 'o'),facecolors='none' if label.startswith('oracle') else color,edgecolors=color,s=35,label=name if col==0 and row==0 and k==0 else None)
            ax.axhline(threshold,color='black',ls=':',lw=1);ax.set_yscale('log');ax.set_ylim(.003,5);ax.set_xticks([1,2,3,4]);ax.grid(axis='y',alpha=.2);ax.set_ylabel('H128 NRMSE' if row==0 else 'Action-response error')
            if row==0:ax.set_title('Dynamics only' if change=='dynamics' else 'Observation + dynamics')
            if row==1:ax.set_xlabel('Seed index (4: first unopened in composed control)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.07),frameon=False,fontsize=9)
    fig.suptitle('A standard composition removes the apparent joint-shift gap',y=1.12,fontsize=13)
    fig.text(.5,-.01,'All points use the longer available VAL horizon. Filled diamonds: ordinary calibration composition. Hollow: privileged oracle. Triangles: failures.',ha='center',fontsize=8.5)
    fig.tight_layout();p=reports/'ordinary_composition85_v1.png';fig.savefig(p,dpi=160,bbox_inches='tight');fig.savefig(reports/'ordinary_composition85_v1.svg',bbox_inches='tight');plt.close(fig);return [p]

def control_report(controls):
    s=controls['summary'];composed=controls['records']['composed1']+controls['records']['composed2'];lines=['| Seed | Condition | Ordinary H128 NRMSE | Ordinary response | Ordinary gate | Oracle H128 / response |','|---:|---|---:|---:|---|---|']
    for r in sorted(composed,key=lambda r:(r['seed'],SHIFT_CASES.index(r['case']))):
        a=r['summary']['ordinary_val128'];b=r['summary']['oracle_val128'];lines.append(f"| {r['seed']} | {r['case']} | {a['median128']:.6g} | {a['median_response']:.6g} | {'PASS' if a['suffices'] else 'FAIL'} | {b['median128']:.6g} / {b['median_response']:.6g} |")
    return f'''## Decisive additional existing-method control

**The proposed necessity of a new calibration mechanism is rejected in this controlled subcase.** A standard calibration followed by source-field affine-residual identification, selected using VAL H128, passes the unchanged trajectory/action-response/no-failure gates in {s['ordinary_composed_val128_passed']}/16 Brusselator cases, including all {s['ordinary_composed_joint_val128_passed']}/4 joint-change seeds. Seed2026854 was unopened before this control; the preceding three seeds are post-screen development evidence. No new method or final research direction has been validated.

![Ordinary attribution and composed calibration](figures/ordinary_composition85_v1.png)

{chr(10).join(lines)}

This is not a test-selected best-of-methods result: the entire ordinary calibration/field configuration is selected on VAL before EVAL is generated. Main selection uses the available128-step VAL trajectory; diagnostic H32 selections, oracle results and every failure are also retained in the machine-readable evidence. A source field is first qualified on source VAL. Degree-three features suit this small fully observed system; the result is not universal superiority over latent world models.

### What the successive controls excluded

The original screen passed10/12 scored neural-source cases. Its two unresolved Brusselator dynamics/joint conditions were followed by standard weak-form regression: this did not repair the changed dynamics in either seed. A same256/128-transition-budget allocation control compared four64-step versus two128-step TRAIN paths, and two64-step versus one128-step VAL paths. Standard affine residual updates in source coordinates repaired dynamics-only errors; the tested raw-observation fits still failed joint changes on all three seeds. These are post-screen attribution controls, not independent confirmations of a novel gap.

A privileged physical-state coverage diagnostic found about97.9–98.1% of changed-dynamics EVAL states outside the convex hull of the eight static calibration anchors. The diagnostic never entered fitting or model selection. Convex-hull membership is neither an identifiability certificate nor proof that calibration must fail. The composed ordinary baseline subsequently succeeded despite this extrapolation, invalidating a necessity argument based on the earlier failures.

The final composition fits bidirectional affine or affine-plus-RBF calibration on only the same eight anchors, transforms target TRAIN into source coordinates, then fits the already tested weak source-field/affine residual. It uses the long allocation already described, not additional transitions or true target states/equations. Frozen-field options are allowed. Thus this result does not isolate calibration alone as a causal factor relative to the original screen, which also differed in source architecture and selection horizon; it establishes that an available existing repair suffices under the stated information/budget contract.

The family oracle knows the functional form of the sensor generator but estimates its parameters from the same eight anchors. It is explicitly privileged and never included in the ordinary16/16 count. Its worse seed1 dynamics score is retained; the oracle label is not a guarantee of optimal finite-sample estimation.

### Execution and verification

The first composed capture completed seeds1/2 and stopped before seed3 on an immutable-source-file creation error. The second source version reads that existing source and runs only seeds3/4 in a fresh folder. Both source versions, failure-log hash, completed records and exit statuses are retained. No scientific grid or gate changed in this repair.

Weak, factorial and composed checks independently verify {s['control_verified_prediction_scalars']} additional prediction scalars and {s['control_metric_rows']} metric rows. Source/residual coefficients are checked by augmented least squares, paired calibrations are re-fitted, simulations and scores reimplemented, and physical reference paths compared with DOP853. These counts describe correlated numerical checks, not independent tasks or evidence strength. See the published verification records, case hashes and pre-EVAL seals. All raw arrays, weights and logs remain on the server.

**Next decision:** retire this candidate as a reason to invent a new mechanism in these fixtures. Harder observation or dynamics settings are not automatically new research gaps. Any replacement needs a concrete scientific requirement, competent inference/identification baselines, a differentiator from existing world-model adaptation, and held-out evidence. Overall research objective remains ACTIVE / UNACHIEVED; accepted directions0.
'''

def export_controls(pub,controls,put,jput,scrub):
    public={k:[{a:b for a,b in r.items() if a!='rows'} for r in rr] for k,rr in controls['records'].items()}
    jput(pub/'evidence'/'CONTROL_SUMMARIES85_PUBLIC_V1.json',scrub(public));jput(pub/'evidence'/'CONTROL_PRE_EVAL_SELECTIONS85_PUBLIC_V1.json',scrub(controls['seals']))
    for key,name in [('verification','CONTROL_VERIFICATION85_V1.json'),('composed_verification','COMPOSED_VERIFICATION85_V1.json'),('coverage','COVERAGE85_PUBLIC_V1.json')]:jput(pub/'evidence'/name,scrub(controls[key]))
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(controls['rows'][0]));w.writeheader();w.writerows(controls['rows']);put(pub/'evidence'/'control_metric_rows85_v1.csv',stream.getvalue())
    logs={}
    for stage in ['weak_controls_v1','attribution_v1','composed_control_v1','composed_control_v2']:logs[stage]=dict(raw_log_sha256=sha(O/stage/'worker85_v1.log'),exit=load(O/stage/'WORKER_EXIT85_V1.json'))
    jput(pub/'evidence'/'CONTROL_EXECUTION85_PUBLIC_V1.json',logs)
