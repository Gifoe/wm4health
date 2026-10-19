"""Diagnostic plots from verified arrays/scores; no model or paper-result claim."""
import pathlib,sys,json,hashlib,os
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
os.environ['MPLCONFIGDIR']=str(R/'outputs/matplotlib_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O=R/'outputs/figures_v1';assert not O.exists();O.mkdir();v=read(R/'outputs/verification_v1.json');q=read(R/'outputs/native_qualification_v1/qualification.json');assert v['status']=='PASS' and v['qualification_sha256']==sha(R/'outputs/native_qualification_v1/qualification.json');raw=R/'outputs/native_qualification_v1/linear_sampling.npz';assert sha(raw)==v['raw_sampling_sha256'];z=np.load(raw);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'ncdssm45'})
fig,ax=plt.subplots(1,2,figsize=(8.6,3.5));colors={'native_z':'#C97726','ordinary_joint_z':'#168C86'};labels={'native_z':'Native marginal-time draws','ordinary_joint_z':'Ordinary joint Markov draws'};dt=z['future_times']-z['past_times'][-1];cov=z['exact_covariance_blocks'];target=np.array([cov[t,t,0,0] for t in range(3)]);ax[0].plot(dt,target,'k--',label='Exact joint Gaussian',lw=1.4)
for key in ['native_z','ordinary_joint_z']:
 y=z[key];variance=y[:,:,0].var(0,ddof=1);error=np.sqrt(2/(len(y)-1))*variance;ax[0].errorbar(dt,variance,yerr=error,marker='o',ms=4,lw=1,color=colors[key],label=labels[key])
ax[0].set(xlabel='Time after last observation',ylabel='Marginal variance, state 1',title='Timewise variances agree');ax[0].legend(frameon=False,fontsize=7,loc='lower right');positions=np.arange(2);width=.24;truth=np.diag(cov[0,0]+cov[2,2]-cov[0,2]-cov[2,0]);ax[1].bar(positions-width,truth,width,color='#3C3C3C',label='Exact joint law')
for index,key in enumerate(['native_z','ordinary_joint_z']):
 y=z[key];var=np.var(y[:,2]-y[:,0],axis=0,ddof=1);error=np.sqrt(2/(len(y)-1))*var;ax[1].bar(positions+index*width,var,width,yerr=error,color=colors[key],label=labels[key],capsize=2)
ax[1].set(xticks=positions,xticklabels=['State 1','State 2'],ylabel='Variance of last minus first future state',title='The temporal contrast distinguishes the laws');ax[1].legend(frameon=False,fontsize=7);fig.suptitle('NCDSSM sampling-contract control: same parameters and filtered history',fontsize=11);fig.text(.5,.005,'Known linear numerical fixture; N = 16,384 per sampler. Error bars: one Gaussian MC standard error. No trained-model comparison.',ha='center',fontsize=7);fig.tight_layout(rect=[0,.035,1,.94]);files=[]
for suffix in ['png','svg','pdf']:
 p=O/f'sampling_contract.{suffix}';fig.savefig(p,dpi=180,metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else None);files.append(p)
plt.close(fig)
C=R.parent/'cycle44';quality=read(C/'outputs/quality_v1/quality_summary.json');verified=read(C/'outputs/quality_verification_v1.json');assert verified['status']=='PASS' and sha(C/'outputs/quality_v1/quality_summary.json')==verified['quality_summary_sha256'];fig,ax=plt.subplots(1,2,figsize=(8.6,3.5));families=['lgssm','ar_mixture','spf_native','spf_condition_trainable'];palette=['#2474A6','#168C86','#C97726','#91619F']
for panel,case in enumerate(['gaussian','nonlinear']):
 for family,color in zip(families,palette):
  rows=[r for r in quality['rows'] if r['case']==case and r['family']==family];xs=[1000*read(C/'outputs/pilot_v1'/r['model']/'pilot.json')['fine_seconds_per_draw'] for r in rows];ys=[r['joint_energy_mean'] for r in rows];ax[panel].scatter(xs,ys,s=25,color=color,label=family,alpha=.85)
 ax[panel].set(xscale='log',xlabel='VAL fine sampler + all functions + transfer (ms/draw)',ylabel='Held-out joint energy (lower is better)',title=case.capitalize());ax[panel].grid(alpha=.2)
ax[0].legend(frameon=False,fontsize=7);fig.suptitle('C44 controlled quality/cost screen, both fitted initializations shown',fontsize=11);fig.text(.5,.005,'Different family objectives, representations and fit budgets. This is a utility control, not an isolated architectural ranking.',ha='center',fontsize=7);fig.tight_layout(rect=[0,.035,1,.94])
for suffix in ['png','svg','pdf']:
 p=O/f'controlled_quality_cost.{suffix}';fig.savefig(p,dpi=180,metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else None);files.append(p)
plt.close(fig);manifest={'status':'GENERATED_PENDING_VISUAL_QA','source_sha256':sha(pathlib.Path(__file__)),'source_verification_sha256':sha(R/'outputs/verification_v1.json'),'quality_verification_sha256':sha(C/'outputs/quality_verification_v1.json'),'files':{p.name:sha(p) for p in files}};(O/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'plots':2,'files':len(files),'visual_QA':'pending'}))
