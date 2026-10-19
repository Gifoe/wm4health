"""Descriptive post-evaluation analysis; changes no fit, selection or score."""
from common47 import *
import csv

def main():
 out=R/'outputs/eval_v1';verification=read(out/'verification.json');assert verification['status']=='PASS_FROZEN_FINITE_GRID_SCREEN';assert not (out/'analysis.json').exists();summary=read(out/'summary.json');rows=read(out/'all_rows.json');groups={(g['model'],g['rho']):g for g in summary['groups']};fits=[read(p/'fit_summary.json') for p in sorted((R/'outputs/fits').iterdir())];assert len(fits)==16
 tables=[];comparisons=[];diagnostics=[];bundles={}
 for ent in read(out/'model_bundles.json'):
  assert sha(out/ent['path'])==ent['sha256'];bundles[ent['model']]={k:v for k,v in np.load(out/ent['path']).items()}
 keys=['energy','mse','coverage90','nll_per65']
 for case in CASES:
  gp=np.load(R/'data'/case/'auditor_generating_parameters.npz');alpha=gp['prior'].copy();occupation=np.zeros_like(alpha)
  for t in range(T):occupation+=alpha;alpha=alpha@gp['P']
  occupation/=T;grid=gp['grid'];truef=-.6*grid if case=='ou' else 1.2*np.tanh(1.7*grid)-.75*grid;sig=.7 if case=='ou' else .55
  for method in METHODS+['known_process_reference']:
   tags=[case+'__known_process_reference'] if method=='known_process_reference' else [f'{case}__{method}__seed{s}' for s in SEEDS]
   for rho in [.25,1.,2.]:tables.append({'case':case,'method':method,'rho':rho,'n_training_initializations':0 if method=='known_process_reference' else 2,**{k:float(np.mean([groups[(tag,rho)][k] for tag in tags])) for k in keys}})
   for tag in tags:
    b=bundles[tag];diagnostics.append({'case':case,'model':tag,'method':method,'drift_occupancy_RMSE':float(np.sqrt(occupation@((b['drift']-truef)**2))),'clock_rate_occupancy_RMSE':float(np.sqrt(occupation@((b['rate']-gp['rate'])**2))),'diffusion':float(b['diffusion']),'diffusion_absolute_error':abs(float(b['diffusion'])-sig)})
  # Cluster by actual physical path; retain paired policies and fitted models.
  rng=np.random.default_rng(20264799+(case=='bistable'));indices=rng.integers(256,size=(2000,256));arrays={}
  for method in METHODS:
   for seed_value in SEEDS:
    for rho in [.25,1.,2.]:
     tag=f'{case}__{method}__seed{seed_value}';rs=sorted([r for r in rows if r['model']==tag and r['rho']==rho],key=lambda x:x['origin']);assert [r['origin'] for r in rs]==list(range(256))
     for k in keys:arrays[method,seed_value,rho,k]=np.array([r[k] for r in rs])
  for method in ['mean_nll','full_laplace','full_hybrid']:
   for metric in ['energy','mse','nll_per65']:
    d=np.stack([arrays[method,s,rho,metric]-arrays['full_nll',s,rho,metric] for s in SEEDS for rho in [.25,1.,2.]]).mean(0);ci=np.quantile(d[indices].mean(1),[.025,.975]);comparisons.append({'case':case,'candidate':method,'reference':'full_nll','metric':metric,'candidate_minus_reference':float(d.mean()),'conditional_path_bootstrap_CI95':ci.tolist(),'seed_mean_differences':{str(s):float(np.mean([arrays[method,s,rho,metric]-arrays['full_nll',s,rho,metric] for rho in [.25,1.,2.]])) for s in SEEDS},'CI_scope':'conditional on these two fitted initializations; physical paths resampled within this carrier, all policy contexts retained; not training-population uncertainty'})
 coupling=read(out/'coupling.json');coupling_diag=[]
 for row in coupling:
  if row['oracle']:continue
  ref=next(c for c in coupling if c['oracle'] and c['case']==row['case'] and c['rho']==row['rho']);p=np.array(ref['initial_bin_probability']);diff=np.array(row['conditional_count_state_covariance'])-np.array(ref['conditional_count_state_covariance']);coupling_diag.append({'model':row['model'],'case':row['case'],'rho':row['rho'],'weighted_covariance':row['probability_weighted_covariance'],'reference_weighted_covariance':ref['probability_weighted_covariance'],'conditional_bin_covariance_RMSE':float(np.sqrt(p@(diff**2)))})
 result={'status':'DESCRIPTIVE_ANALYSIS_COMPLETE_NOT_A_FRAMEWORK_VALIDATION','utc':now(),'fits':16,'formal_updates':sum(f['updates'] for f in fits),'sum_fit_wall_seconds':sum(f['elapsed_seconds'] for f in fits),'selected_at_final':sum(f['selected_at_final_update'] for f in fits),'hard_caps':sum(f['hard_cap_hit'] for f in fits),'tables':tables,'paired_comparisons':comparisons,'mechanism_diagnostics':diagnostics,'coupling_diagnostics':coupling_diag,'fit_summaries':fits,'verification_sha256':sha(out/'verification.json'),'source_sha256':sha(__file__),'accepted_final_directions':0};dump(out/'analysis.json',result)
 with (out/'method_policy_means.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'mean_nll':'#8c8c8c','full_nll':'#2864a5','full_laplace':'#d17c20','full_hybrid':'#3c8c63','known_process_reference':'#812c73'};labels={'mean_nll':'Mean-clock NLL','full_nll':'Full-clock NLL','full_laplace':'Laplace only','full_hybrid':'NLL + Laplace','known_process_reference':'Known-process reference'};figdir=R/'outputs/figures_v1';figdir.mkdir(exist_ok=True);exports=[]
 fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
 for i,case in enumerate(CASES):
  for j,metric in enumerate(['energy','nll_per65']):
   ax=axes[i,j]
   for method in METHODS+['known_process_reference']:
    values=[next(t[metric] for t in tables if t['case']==case and t['method']==method and t['rho']==rho) for rho in [.25,1.,2.]];ax.plot([.25,1.,2.],values,'o--' if method=='known_process_reference' else 'o-',color=colors[method],label=labels[method],linewidth=1.5,markersize=4)
   ax.set_title(case.capitalize());ax.set_ylabel('Future energy / sqrt(32)' if metric=='energy' else 'Joint observed NLL / 65');ax.grid(alpha=.2);ax.set_xticks([.25,1.,2.]);ax.set_xlabel('Known observation-density multiplier')
 axes[0,0].legend(fontsize=8);fig.suptitle('Clock/state objective screen: two fitted initializations per learned arm');fig.text(.5,.015,'Paired observation policies share physical EVAL paths. Fixed 1D grid and known emission; no real-domain claim.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.045,1,.95])
 for ext in ['png','svg','pdf']:
  p=figdir/('clock_objective_screen.'+ext);fig.savefig(p,dpi=180);exports.append({'path':p.name,'sha256':sha(p)})
 plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for i,case in enumerate(CASES):
  reference=bundles[case+'__known_process_reference'];grid=reference['grid'];occupation=np.zeros_like(grid,dtype=float);gp=np.load(R/'data'/case/'auditor_generating_parameters.npz');alpha=gp['prior'].copy()
  for t in range(T):occupation+=alpha;alpha=alpha@gp['P']
  occupation/=T;keep=occupation>1e-4
  for method in METHODS:
   for seed_index,s in enumerate(SEEDS):
    b=bundles[f'{case}__{method}__seed{s}']
    for j,key in enumerate(['drift','rate']):axes[i,j].plot(grid[keep],b[key][keep],color=colors[method],alpha=.85,linestyle='-' if seed_index==0 else ':',label=labels[method] if seed_index==0 else None)
   for j,key in enumerate(['drift','rate']):axes[i,j].set_xlabel('Physical grid state');axes[i,j].set_ylabel(key);axes[i,j].grid(alpha=.2)
  for j,key in enumerate(['drift','rate']):axes[i,j].plot(grid[keep],reference[key][keep],color=colors['known_process_reference'],linewidth=2,linestyle='--',label=labels['known_process_reference']);axes[i,j].set_title(case.capitalize()+' '+key)
  for j,method in enumerate(METHODS):
   values=[next(d['diffusion'] for d in diagnostics if d['model']==f'{case}__{method}__seed{s}') for s in SEEDS];axes[i,2].scatter([j-.06,j+.06],values,color=colors[method],s=28)
  axes[i,2].axhline(float(reference['diffusion']),color=colors['known_process_reference'],linestyle='--');axes[i,2].set_xticks(range(4),['Mean','NLL','Laplace','Hybrid']);axes[i,2].set_ylabel('Diffusion amplitude');axes[i,2].set_title(case.capitalize()+' diffusion');axes[i,2].grid(axis='y',alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle('Auditor mechanism diagnostics after all selections and EVAL opening');fig.text(.5,.015,'Curves shown where true average state probability > 1e-4. Drift/clock RMSE uses full true occupation, not curve trimming.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.045,1,.95])
 for ext in ['png','svg','pdf']:
  p=figdir/('learned_mechanisms.'+ext);fig.savefig(p,dpi=180);exports.append({'path':p.name,'sha256':sha(p)})
 plt.close(fig);dump(figdir/'manifest.json',{'visual_QA':'PENDING','exports':exports,'analysis_sha256':sha(out/'analysis.json')});print(json.dumps({k:v for k,v in result.items() if k in ['status','fits','formal_updates','sum_fit_wall_seconds','selected_at_final','hard_caps','paired_comparisons']}),flush=True)
if __name__=='__main__':main()
