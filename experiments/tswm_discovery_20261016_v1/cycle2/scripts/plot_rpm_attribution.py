"""Show all existing repairs on the original primary condition."""
import csv,hashlib,json,pathlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';FIG=OUT/'figures'
def main():
 methods=[('native_alarm_extended','native','Native surrogate'),('native_alarm_extended','exact_same_readout','Frozen ratio correction'),('rpm_exact','positive_site_exact_ML','Positive-site RPM, numerical ML'),('rpm_exact','full_gaussian_exact_ML','Full Gaussian RPM, numerical ML'),('generative_alarm','generative_ssm','Ordinary emission SSM'),('native_alarm_extended','known_observer_oracle','Known-likelihood oracle*')]
 rows=[];data=[]
 for folder,method,label in methods:
  items=[]
  for path in sorted((OUT/folder).glob('state_dependent_alarm*_evaluation.json')):
   result=json.loads(path.read_text())
   for r in result['metrics']:
    if r['method']==method and r['prefix']==16 and r['horizon']==1 and r['subset']=='alarm_no_anchor':items.append({'seed':result['seed'],'label':label,**r})
  assert len(items)==3,(method,len(items));data.append(items);rows.extend(items)
 FIG.mkdir(exist_ok=True,parents=True)
 with (FIG/'rpm_attribution_data.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,3,figsize=(12.2,4.4),sharey=True);colors=['#0072B2','#E69F00','#56B4E9','#CC79A7','#009E73','#666666']
 for ax,key,title in zip(axes,['moment_gaussian_95_coverage','expected_tail_brier','moment_gaussian_nll'],['95% coverage','Tail Brier score (lower better)','Gaussian NLL (lower better)']):
  for i,(items,color) in enumerate(zip(data,colors)):
   vals=[r[key] for r in items];shown=[statistics.mean(vals)] if i==5 else vals
   ax.scatter(shown,np.full(len(shown),i)+np.linspace(-.065,.065,len(shown)) if len(shown)>1 else [i],s=18,color=color,alpha=.8)
   ax.plot(statistics.mean(vals),i,marker='|',markersize=13,color='#222222')
  ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_ylim(5.5,-.5)
 axes[0].set_yticks(range(6),[m[2] for m in methods]);axes[0].axvline(.95,linestyle=':',color='#222222',lw=.8)
 fig.subplots_adjust(left=.27,right=.98,top=.86,bottom=.21,wspace=.27)
 fig.text(.04,.055,'Registered primary: prefix 16, forecast horizon 1, alarm with missing measurement (292 test episodes).\nLearned-model dots: three initializations on one data draw; black tick: mean. *Oracle: deterministic, known likelihood.',fontsize=9)
 for ext in ['png','pdf','svg']:fig.savefig(FIG/('rpm_training_attribution.'+ext),dpi=180)
 plt.close(fig);print('saved attribution plot and complete source CSV')
if __name__=='__main__':main()
