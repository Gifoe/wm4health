import runtime_profile
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 summary=json.loads((ROOT/'outputs/real_controls_summary.json').read_text())['rows'];quant=json.loads((ROOT/'outputs/timer_quantization/result.json').read_text());fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),layout='constrained')
 arms=['ridge_no_clock','ridge_oracle_future_dt','HGB_no_clock','HGB_oracle_future_dt'];names=['Ridge','Ridge + actual future time','HGB','HGB + actual future time'];colors=['#286A91','#78A6BD','#A65045','#D69D8C'];groups=['held-out walks','fast waves','slow waves']
 for ax,key,title in zip(axes[:2],['one_step_normalized_MSE','rollout16_normalized_MSE'],['One-step measured RPM error','16-step measured RPM error']):
  for j,(arm,name,col) in enumerate(zip(arms,names,colors)):
   vals=[next(r[key] for r in summary if r['group']==g and r['arm']==arm) for g in groups];ax.bar(np.arange(3)+(j-1.5)*.19,vals,.18,label=name,color=col)
  ax.set_xticks(np.arange(3),['Held-out walks','Fast waves','Slow waves']);ax.set_yscale('log');ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('MSE / 3000² (recording-weighted mean)')
 rs=quant['error_contribution'];x=np.arange(2);axes[2].bar(x-.17,[r['low_resolution_row_fraction'] for r in rs],.32,label='Fraction of rows',color='#67815F');axes[2].bar(x+.17,[r['low_resolution_share_of_HGB_no_clock_squared_error'] for r in rs],.32,label='Fraction of squared error',color='#A65045');axes[2].set_ylim(0,1.24);axes[2].set_yticks([0,.25,.5,.75,1.]);axes[2].set_xticks(x,['Multiple-target regimes','Single-target regimes']);axes[2].set_title('Low-resolution sensor blocks dominate error',fontsize=11);axes[2].legend(frameon=False,fontsize=8,loc='upper center');axes[2].spines[['top','right']].set_visible(False)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=9);fig.suptitle('Real wind-tunnel controls: separate prediction, time availability and measurement artifacts',fontsize=13)
 out=ROOT/'outputs/figures';out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'real_data_controls.png',dpi=180);fig.savefig(out/'real_data_controls.pdf');plt.close(fig);print('Saved actual-result diagnostic figure; future time arms are privileged comparators, not deployable simulation.')
if __name__=='__main__':main()
