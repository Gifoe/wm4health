"""Diagnostic scientific figure; all values are read from raw measured results."""
import runtime_profile
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 rows=json.loads((ROOT/'outputs/action_equivalence_pilot/results.json').read_text());assert len(rows)==27
 colors=['#246A91','#D88C00','#B34250'];arms=['prediction','hinge','full'];labels=['Prediction only','+ native hinge','+ hinge / frozen IDM'];kinds=['null','linear','deadzone']
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,key,title in zip(axes,['factual_MSE','paired_response_MSE','offline_candidate_regret_mean'],['Held-out physical prediction error','Paired physical response error','Offline candidate regret']):
  for j,arm in enumerate(arms):
   vals=np.array([[r['test_final'][key] for r in rows if r['arm']==arm and r['kind']==k] for k in kinds]);mean=vals.mean(1);sd=vals.std(1,ddof=1);x=np.arange(3)+(j-1)*.23
   ax.bar(x,mean,.22,color=colors[j],label=labels[j]);ax.errorbar(x,mean,yerr=sd,fmt='none',color='#303030',capsize=3,lw=.8)
   for xx,vv in zip(x,vals):ax.scatter(np.full(3,xx),vv,color='black',s=8,zorder=3,alpha=.6)
  ax.set_xticks(np.arange(3),['Zero effect','Linear effect','Action deadzone']);ax.set_yscale('log');ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Mean squared error / cost (log scale)')
 handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=3,frameon=False)
 fig.suptitle('Controlled native-objective transfer: larger action separation can add false effects',fontsize=13)
 out=ROOT/'outputs/figures';out.mkdir(parents=True,exist_ok=True)
 fig.savefig(out/'action_objective_probe.png',dpi=180);fig.savefig(out/'action_objective_probe.pdf');plt.close(fig)
 print('Saved scientific diagnostic figure; dots are three initialization seeds on shared cohorts, bars mean, error bars sample SD. Fixed encoder/reduced architecture; not native-paper reproduction.')
if __name__=='__main__':main()
