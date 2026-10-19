"""Plot completed numerical diagnostics; no new selection or fitted method."""
import runtime_profile
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
def main():
 out=ROOT/'outputs/figures';out.mkdir(parents=True,exist_ok=True)
 fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
 rows=[json.loads(s) for s in (ROOT/'outputs/child_native_A_r1/train.jsonl').read_text().splitlines()]
 axes[0,0].plot([r['global_step'] for r in rows],[r['validation']['mcc_spearman'] for r in rows],marker='.',color='#326B95');axes[0,0].set(xlabel='Optimizer update',ylabel='Validation Spearman MCC',title='A. Complete native A configuration (one run)',ylim=(0,1))
 noise=load('outputs/generator_noise_audit/results.json')
 for name,label,color in [('native','Original generator','#B55341'),('independent_noise_control','Noise-index control','#326B95')]:
  axes[0,1].plot(range(1,8),np.maximum(0,noise[name]['innovation_cov_eigenvalues']),marker='o',label=label,color=color)
 axes[0,1].set(xlabel='Ordered covariance eigenvalue',ylabel='Innovation variance',title='B. Shared-noise generator audit');axes[0,1].legend(frameon=False,fontsize=8)
 rec=load('outputs/signal_recoverability/results.json')['rows']
 for variant,label,color in [('native_A','Original generator','#B55341'),('independent_noise_A','Noise-index control','#326B95')]:
  selected=[r for r in rec if r['variant']==variant];axes[1,0].plot([r['context_width'] for r in selected],[r['mean_spearman'] for r in selected],marker='o',label=label,color=color)
 axes[1,0].set(xlabel='Context width',ylabel='Mean validation Spearman',title='C. Supervised OLS recoverability certificate',ylim=(.94,1.002),xticks=[1,3,5]);axes[1,0].legend(frameon=False,fontsize=8)
 density=load('outputs/conditional_density/results.json')
 for name,label,color in [('native_inverse','Native inverse MLP','#B55341'),('normalized_gaussian','Ordinary Gaussian','#326B95')]:
  vals=np.array([[t['proper_normalized_test_NLL'] for t in r['test']] for r in density if r['kind']==name]);axes[1,1].errorbar([-2,0,2],vals.mean(0),yerr=vals.std(0,ddof=1),marker='o',capsize=3,label=label,color=color)
 axes[1,1].set(xlabel='Fixed conditioning value',ylabel='Proper test NLL (mean ± seed SD)',title='D. Six controlled density fits');axes[1,1].legend(frameon=False,fontsize=8)
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.savefig(out/'cycle6_diagnostics.png',dpi=180);fig.savefig(out/'cycle6_diagnostics.pdf');plt.close(fig)
 (out/'caption.md').write_text('Native source/config run uses disclosed manual orchestration; A is validation, not independent test. B fixes only innovation indexing. C uses supervised factor labels and is not a fair unsupervised-model comparison. D uses three initializations on one shared data cohort; error bars are seed SD, not confidence intervals. No novel gap is asserted.\n',encoding='utf-8')
 print('Wrote cycle6 diagnostics PNG/PDF and scope caption')
if __name__=='__main__':main()
