"""Render saved, verified aggregate metrics; no model or calibration selection."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'outputs/native_gate_attempt2/results.json').read_text())
out=ROOT/'outputs/analysis';assert not out.exists();out.mkdir(parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
labels=['Marginal','Total / sqrt(H)','Half contrast','DCT 1','DCT 2','Alternating']
keys=[None,'constant','half_contrast','dct1','dct2','alternating'];xx=np.arange(len(keys))
for j,(name,title,color) in enumerate([('native','Native independent','#4276a8'),
    ('bias_calibrated_diagonal','Calibrated diagonal','#e1a345'),('bias_calibrated_full','Calibrated full covariance','#479878')]):
    record=data['heldout_probe'][name]
    vals=[100*(record['marginal_coverage95'] if key is None else record['functionals'][key]['coverage95']) for key in keys]
    axes[0].bar(xx+(j-1)*.25,vals,.24,label=title,color=color)
axes[0].axhline(95,color='#333333',ls='--',lw=1);axes[0].set_ylim(0,104)
axes[0].set_xticks(xx,labels,rotation=24,ha='right');axes[0].set_ylabel('Nominal 95% interval coverage (%)')
axes[0].set_title('Pointwise calibration does not ensure trajectory calibration',fontsize=11)
axes[0].legend(loc='lower left',fontsize=8)
names=['native','temperature','bias_calibrated_diagonal','bias_calibrated_full']
vals=[data['heldout_probe'][k]['joint_nll_per_horizon'] for k in names]
axes[1].barh(range(4),vals,color=['#4276a8','#999999','#e1a345','#479878'])
axes[1].set_yticks(range(4),['Native','Temperature','Calibrated diagonal','Calibrated full'])
axes[1].set_xlim(-2.08,-1.75);axes[1].invert_yaxis();axes[1].set_xlabel('Joint NLL / horizon (lower is better)')
axes[1].set_title('Ordinary covariance already gives a substantial repair',fontsize=11)
for i,v in enumerate(vals):axes[1].text(v+.003,i,f'{v:.4f}',va='center',fontsize=9)
fig.suptitle('Frozen meta-ID: 256 unseen systems; calibration on 256 separate systems',fontsize=13)
fig.supxlabel('Native WH protocol includes whole-sequence normalization; no prospective raw-unit or novelty claim.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(out/('native_metaid_gate.'+ext),dpi=180)
plt.close(fig)
