"""Analyze completed fit and independently verified covariance diagnostic; no test data."""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 train=ROOT/'outputs/train/NP_MTS3_seed2021';fit=json.loads((train/'results.json').read_text());assert fit['training_complete']
 pointer=json.loads((train/'best_checkpoint.json').read_text());assert pointer['step']==fit['best_step']
 diagnostic=ROOT/f"outputs/diagnostics/final_step{fit['best_step']:06d}"
 result=json.loads((diagnostic/'results.json').read_text());verification=json.loads((diagnostic/'independent_verification.json').read_text())
 assert verification['passed'] and verification['results_sha256']==sha(diagnostic/'results.json')
 assert result['checkpoint']['sha256']==pointer['sha256'] and not result['smoke_only']
 with np.load(diagnostic/'arrays.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
 assert sha(diagnostic/'arrays.npz')==result['arrays_sha256']
 records={r['control']:r for r in result['rows']};out=ROOT/'outputs/analysis';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'results.json').exists()
 nll={name:.5*(np.log(2*np.pi*a[name+'_variance'].astype(np.float64))+
                    (a['target'].astype(np.float64)-a['native_mean'].astype(np.float64))**2/a[name+'_variance'].astype(np.float64))
      for name in records}
 changes=[]
 for name in ['dense_independent','dense_joint']:
  delta=nll[name].mean(axis=(1,2))-nll['native'].mean(axis=(1,2))
  changes.append({'control':name,'NLL_difference':float(delta.mean()),'origin_NLL_difference_quantiles':np.quantile(delta,[0,.25,.5,.75,1]).tolist(),
                  'fraction_origins_lower_NLL':float(np.mean(delta<0)),
                  'mean_variance_relative_change_signed':float(np.mean(a[name+'_variance']/a['native_variance']-1)),
                  'mean_variance_relative_change_absolute':float(np.mean(np.abs(a[name+'_variance']/a['native_variance']-1))),
                  'max_variance_relative_change_absolute':float(np.max(np.abs(a[name+'_variance']/a['native_variance']-1)))})
 refs=json.loads((ROOT/'outputs/references/results.json').read_text())['rows'];selected=next(r for r in fit['validation'] if r['step']==fit['best_step'])
 assert np.isclose(selected['NLL'],records['native']['NLL'],rtol=1e-5,atol=1e-5)
 summary={'fit':{k:fit[k] for k in ['last_step','best_step','best_validation_NLL','elapsed_seconds','stopping_reason','peak_cuda_allocated_bytes']},
          'selected_native':selected,'references':refs,'posthoc_controls':records,'paired_descriptive_changes':changes,
          'validation_origins':len(a['origins']),'horizon':a['target'].shape[1],'checkpoint':pointer,
          'diagnostic_path':diagnostic.relative_to(ROOT).as_posix(),'diagnostic_sha256':sha(diagnostic/'results.json'),
          'verifier_sha256':sha(diagnostic/'independent_verification.json'),'analysis_script_sha256':sha(pathlib.Path(__file__)),
          'interpretation_limits':['One seed and one transfer series; no original paper reproduction or generality claim.',
             'Validation selected the checkpoint. No independent confirmatory test or unbiased model-ranking claim.',
             'Ordinary references are not capacity/compute matched; covariance controls preserve the same trained weights and mean.',
             'Modified covariance summaries can be outside the variance decoder training distribution.',
             'The shared initial factorized belief is assumed; this is not the true learned joint posterior.',
             'Budget termination does not prove convergence or sufficient training for every downstream claim.'],
          'no_test_evaluation':True,'accepted_novel_directions':0,'objective_achieved':False}
 (out/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 colors={'native':'#263238','dense_independent':'#2878B5','dense_joint':'#D35D3E'}
 labels={'native':'Native','dense_independent':'Dense fast state; reset cross-scale covariance','dense_joint':'Dense fast/context joint'}
 fig,axes=plt.subplots(1,3,figsize=(11,3.2))
 steps=[r['step'] for r in fit['validation']]
 for axis,key,title in zip(axes,['NLL','MAE_standardized','RMSE_standardized'],['Validation Gaussian NLL','Validation MAE (standardized)','Validation RMSE (standardized)']):
  axis.plot(steps,[r[key] for r in fit['validation']],color=colors['native'],marker='o',ms=3,label='Native MTS3')
  for r,c in zip(refs,['#B0BEC5','#78909C','#7D5DA8']):axis.axhline(r[key],color=c,ls='--',lw=1,label=r['model'])
  axis.axvline(fit['best_step'],color='#607D8B',ls=':',lw=.8);axis.set(xlabel='Optimizer updates',title=title)
 axes[0].legend(fontsize=7,frameon=False)
 fig.suptitle('One native MTS3 pilot: NP, seed2021, history + deterministic calendar',fontsize=10)
 fig.text(.5,.01,'Validation checkpoint selection; final20% test untouched. Reference models have different capacity/training cost.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.07,1,.91])
 for ext in ['png','svg']:fig.savefig(out/('training_curve.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,6.8));h=np.arange(1,a['target'].shape[1]+1);d=a['A'].shape[0]//2
 native_trace=a['native_latent_covariance_summary'][:,:,:2*d].sum(-1)
 for name in ['dense_independent','dense_joint']:
  trace=a[name+'_latent_covariance_summary'][:,:,:2*d].sum(-1)
  axes[0,0].plot(h,100*(trace/native_trace-1).mean(0),color=colors[name],label=labels[name])
  axes[0,1].plot(h,100*(a[name+'_variance']/a['native_variance']-1).mean(axis=(0,2)),color=colors[name],label=labels[name])
  delta=nll[name].mean(axis=(1,2))-nll['native'].mean(axis=(1,2))
  axes[1,0].plot(np.arange(1,len(delta)+1),delta,color=colors[name],lw=1,alpha=.8,label=labels[name])
 for name in records:
  coverage=(np.abs(a['target']-a['native_mean'])<=1.959963984540054*np.sqrt(a[name+'_variance'])).mean(axis=(0,2))
  axes[1,1].plot(h,coverage,color=colors[name],lw=1,label=labels[name])
 for axis in axes[0]:axis.axhline(0,color='gray',lw=.7,ls=':');axis.set_xlabel('Forecast horizon')
 axes[0,0].set(title='Change in latent covariance trace',ylabel='Relative change from native (%)')
 axes[0,1].set(title='Change in decoded predictive variance',ylabel='Relative change from native (%)')
 axes[1,0].axhline(0,color='gray',lw=.7,ls=':');axes[1,0].set(title='Paired origin-wise NLL changes',xlabel='Chronological validation origin',ylabel='NLL(control) - NLL(native)')
 axes[1,1].axhline(.95,color='gray',ls=':',lw=.9);axes[1,1].set(title='Marginal95% interval coverage',xlabel='Forecast horizon',ylabel='Coverage',ylim=(0,1.03))
 axes[0,0].legend(fontsize=7,frameon=False);axes[1,1].legend(fontsize=7,frameon=False)
 fig.suptitle('Learned covariance diagnostic: selected frozen model',fontsize=11)
 fig.text(.5,.01,'Post-hoc covariance controls; identical native means; decoder not retrained. Descriptive validation results only.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.045,1,.95])
 for ext in ['png','svg']:fig.savefig(out/('covariance_diagnostic.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
 print(json.dumps({'fit':summary['fit'],'paired_descriptive_changes':changes,'objective_achieved':False}),flush=True)
if __name__=='__main__':main()
