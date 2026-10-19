"""Scientific descriptive figures, only from verified fixed analysis."""
import pathlib,sys,json
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'))
import runtime_profile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A=R/'outputs/analysis_v1';D=json.loads((A/'analysis.json').read_text(encoding='utf-8'))
assert D['status']=='completed' and not D['test_scored']
F=R/'outputs/figures_v1';assert not F.exists();F.mkdir()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
spec=[('quantile_mlp','iid','Q-MLP\nIID','#6088b4'),('quantile_mlp','analogue','Q-MLP\nrank','#6088b4'),
      ('quantile_mlp','analogue_calibrated','Q-MLP\nrank+CAL','#40678e'),
      ('kite_l1','native','KITE L1','#d99c4b'),('kite_l2','native','KITE L2','#d99c4b'),
      ('kite_l2','native_calibrated','KITE L2\n+CAL','#ba7630'),('hgb','analogue_joint','HGB\nanalogue','#649781')]
fig,axes=plt.subplots(2,2,figsize=(12,7.4),constrained_layout=True)
for row,data in enumerate(['wind','solar']):
    for col,metric in enumerate(['crps','daily_mean_crps']):
        ax=axes[row,col];values=[];errors=[];labels=[];colors=[]
        for model,arm,label,color in spec:
            rec=next(x for x in D['family_means'] if x['dataset']==data and x['model']==model and x['arm']==arm)
            values.append(rec[metric]);errors.append(rec[metric+'_seed_std'] or 0);labels.append(label);colors.append(color)
        x=np.arange(len(values));ax.bar(x,values,color=colors,width=.72,yerr=errors,capsize=3,linewidth=0)
        ax.set_xticks(x,labels,fontsize=8.5);ax.set_ylabel('Fair CRPS (normalized power)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title(data.capitalize()+(' — hourly marginals' if col==0 else ' — 24h mean power'))
fig.suptitle('Same information, exploratory VAL only; neural mean ± seed SD (2 seeds), HGB one fit',fontsize=11)
for suffix in ['png','pdf']:fig.savefig(F/f'conditional_path_controls.{suffix}')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),constrained_layout=True)
colors={'quantile_mlp':'#4b79aa','kite_l1':'#d18c36','kite_l2':'#478973'}
for ax,data in zip(axes,['wind','solar']):
    for kind in ['quantile_mlp','kite_l1','kite_l2']:
        for sd,style in [(2026371,'-'),(2026372,'--')]:
            folder=R/'outputs/learning_v2'/data/f'{kind}_{sd}'
            curve=json.loads((folder/'curve.json').read_text(encoding='utf-8'));info=json.loads((folder/'fit.json').read_text(encoding='utf-8'))
            ax.plot([x['step'] for x in curve],[x['val_crps'] for x in curve],style,color=colors[kind],label=f'{kind}, seed {sd%10}')
            selected=next(x for x in curve if x['step']==info['selected_step']);ax.scatter([selected['step']],[selected['val_crps']],s=24,color=colors[kind],zorder=4)
    ax.set_title(data.capitalize());ax.set_xlabel('Training updates');ax.set_ylabel('VAL marginal fair CRPS (64 paths)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,loc='best');fig.suptitle('All fixed-budget curves; dots mark selected checkpoints, TEST not scored',fontsize=11)
for suffix in ['png','pdf']:fig.savefig(F/f'learning_curves.{suffix}')
plt.close(fig)
print(json.dumps({'figures':4,'directory':str(F)}))
