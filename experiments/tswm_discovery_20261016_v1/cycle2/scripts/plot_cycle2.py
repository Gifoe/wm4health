"""Deterministic evidence figures; same statistical source as the ledgers.

Adapts the CCFA small-multiple recipe to Matplotlib with Okabe-Ito colors.
No generated illustration, invented uncertainty or method topology.
"""
import csv,hashlib,json,pathlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';FIG=OUT/'figures'
METHODS=['native','exact_same_readout','generative_ssm','known_observer_oracle']
LABELS=['Native RP-GSSM','Retained F, frozen RP','Ordinary emission SSM','Known-likelihood oracle*']
COLORS=['#0072B2','#E69F00','#009E73','#666666'];MARKERS=['o','s','^','x']
def main():
 FIG.mkdir(exist_ok=True,parents=True);results=[];inputs={}
 for folder in ['native_alarm_extended','generative_alarm']:
  files=sorted((OUT/folder).glob('*_evaluation.json'));assert len(files)==6,(folder,len(files))
  for f in files:results.append(json.loads(f.read_text()));inputs[f.relative_to(ROOT).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
 rows=[]
 for result in results:
  for r in result['metrics']:
   if r['method'] not in METHODS:continue
   rows.append({'dataset':result['dataset'],'seed':result['seed'],**r})
 with (FIG/'alarm_figure_data.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titlesize':11})
 fig,axes=plt.subplots(2,2,figsize=(11,7.6));handles=[]
 panels=[('state_dependent_alarm','moment_gaussian_95_coverage','a  Registered forecast condition',16),('independent_alarm_control','moment_gaussian_95_coverage','b  Independent-alarm control',16),('state_dependent_alarm','mean_predicted_abs_state_gt2','c  Single-alarm diagnosis',1),('state_dependent_alarm','expected_tail_brier','d  Registered tail-risk score',16)]
 for ax,(kind,key,title,L) in zip(axes.flat,panels):
  Hs=[0,1,8,32] if L==1 else [1,8,32];x=np.arange(len(Hs))
  for method,label,color,marker in zip(METHODS,LABELS,COLORS,MARKERS):
   means=[];sds=[]
   for H in Hs:
    vs=[r[key] for r in rows if r['dataset']==kind and r['method']==method and r['prefix']==L and r['horizon']==H and r['subset']=='alarm_no_anchor']
    assert len(vs)==3;means.append(statistics.mean(vs));sds.append(statistics.stdev(vs))
   line=ax.errorbar(x,means,yerr=sds,color=color,marker=marker,linestyle='--' if method=='known_observer_oracle' else '-',capsize=2,linewidth=1.4,label=label)
   if ax is axes[0,0]:handles.append(line)
  ax.set_title(title,loc='left');ax.set_xticks(x,Hs);ax.set_xlabel('Forecast horizon');ax.grid(axis='y',alpha=.2)
  if 'coverage' in key:ax.axhline(.95,color='#222222',linewidth=.8,linestyle=':');ax.set_ylim(.8,1.);ax.set_ylabel('Moment-Gaussian 95% coverage')
  elif 'brier' in key:ax.set_ylim(bottom=0);ax.set_ylabel('Expected tail Brier score (lower better)')
  else:
   target=[statistics.mean(r['mean_realized_or_noise_expected_abs_state_gt2'] for r in rows if r['dataset']==kind and r['method']=='native' and r['prefix']==L and r['horizon']==H and r['subset']=='alarm_no_anchor') for H in Hs]
   ax.plot(x,target,':',color='#222222',linewidth=1.,label='Empirical / noise-expected rate');ax.set_ylim(bottom=0);ax.set_ylabel('Predicted probability of |state| > 2')
 fig.legend(handles,LABELS,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.995));fig.subplots_adjust(top=.87,bottom=.13,hspace=.42,wspace=.28)
 fig.text(.06,.035,'Three initializations, one training/test data draw. Error bars: initialization SD.\nAll panels condition on an alarm without a direct measurement. *Oracle knows the true observation likelihood.',fontsize=9)
 for ext in ['png','pdf','svg']:fig.savefig(FIG/('alarm_baseline_audit.'+ext),dpi=180)
 plt.close(fig)
 tail=json.loads((OUT/'rpm_tail_probe.json').read_text())['records'];compat=json.loads((OUT/'native_compatible_rpm_probe.json').read_text())['records']
 fig,axes=plt.subplots(1,2,figsize=(10,3.9))
 sets=[([r for r in tail if r['recognition_variance']==3.], 'a  General Gaussian recognition'),([r for r in compat if r['common_full_recognition_variance']==.6 and r['rare_full_recognition_variance']==.999],'b  Native-compatible fixed recognition')]
 for ax,(items,title) in zip(axes,sets):
  epskey='reference_rare_probability';items=sorted(items,key=lambda r:r[epskey],reverse=True);xx=np.array([-np.log10(r[epskey]) for r in items])
  if 'ordinary_gaussian_vi' in items[0]:curves=[('Exact posterior',[r['exact_variance'] for r in items]),('Ordinary Gaussian VI',[r['ordinary_gaussian_vi'][-1]['variance'] for r in items]),('Gaussian auxiliary bound',[r['interior_q_variance'] for r in items])]
  else:curves=[('Exact posterior',[r['exact_posterior_variance'] for r in items]),('Ordinary Gaussian VI',[r['ordinary_gaussian_vi_variance'] for r in items]),('Native surrogate',[r['native_posterior_variance'] for r in items])]
  for (label,values),color,marker in zip(curves,['#222222','#009E73','#0072B2'],MARKERS):ax.plot(xx,values,color=color,marker=marker,label=label)
  ax.set_title(title,loc='left');ax.set_xlabel('-log10(reference rare probability)');ax.set_ylabel('Posterior variance');ax.set_ylim(0,3.25);ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False,loc='lower right')
 fig.subplots_adjust(bottom=.25,wspace=.3,top=.87);fig.text(.07,.055,'Fixed-model counterexamples, not trained-model outcomes. Only panel a has the small-TV construction.\nNo sampling uncertainty: deterministic numerical integration; all control cells remain in the source JSON.',fontsize=9)
 for ext in ['png','pdf','svg']:fig.savefig(FIG/('rpm_fixed_model_audit.'+ext),dpi=180)
 plt.close(fig)
 (FIG/'figure_provenance.json').write_text(json.dumps({'inputs':inputs,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'statistics':'means and SD across three initializations, not independent datasets'},indent=2),encoding='utf-8');print('saved two figures, three formats each, CSV and provenance')
if __name__=='__main__':main()
