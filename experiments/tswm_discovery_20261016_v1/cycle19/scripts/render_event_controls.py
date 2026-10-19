"""Descriptive figures from already frozen results, without fitting or selection."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/analysis';OUT.mkdir(parents=True,exist_ok=True)
event=json.loads((ROOT/'outputs/event_audit_attempt1/results.json').read_text())
attrib=json.loads((ROOT/'outputs/attribution/results.json').read_text())
sid=json.loads((ROOT/'outputs/system_id_control/results.json').read_text())
native=json.loads((ROOT.parent/'cycle18/outputs/native_demo/results.json').read_text())
systems=['lorenz63','chua','sprottM'];sids={r['task']:r for r in sid['rows']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');blue='#3375a2';red='#bb553b';green='#2b7b60'
ax=axs[0,0]
for i,name in enumerate(systems):
    vals=[event['systems'][name]['paths']['native_'+str(seed)]['versus_reference']['own_rank_diagnostic_all_offsets']['max_absolute_CDF_difference'] for seed in [2021,2022,2023]]
    ax.scatter(np.full(3,i)-.12,vals,c=blue,s=35,label='Native draws' if i==0 else None)
    ax.scatter(i+.12,sids[name+'_original']['event_comparison']['own_rank_diagnostic_all_offsets']['max_absolute_CDF_difference'],c=red,marker='D',s=48,label='Past-only library fit' if i==0 else None)
ax.set_xticks(range(3),['Lorenz63','Chua','SprottM']);ax.set_ylabel('Max absolute event-CDF error');ax.set_ylim(0,.23)
ax.set_title('(a) All-offset rank diagnostic; H = 10, 50, 100');ax.legend(fontsize=8)
ax=axs[0,1];case=event['systems']['chua'];h=event['horizons_samples']
ax.plot(h,case['reference']['own_rank_diagnostic_all_offsets']['first_passage_CDF'],'o-',c='#444444',label='Single released reference')
for i,seed in enumerate([2021,2022,2023]):
    ax.plot(h,case['paths']['native_'+str(seed)]['summaries']['own_rank_diagnostic_all_offsets']['first_passage_CDF'],'o--',alpha=.7,c=blue,label='Native draws' if i==0 else None)
ax.plot(h,sids['chua_original']['event_summaries']['own_rank_diagnostic_all_offsets']['first_passage_CDF'],'D-',c=red,label='Past-only library fit')
ax.set_xlabel('Horizon (sample indices)');ax.set_ylabel('Conditional upper-quantile hit probability');ax.set_title('(b) Chua original origin: exploratory path law');ax.legend(fontsize=8)
ax=axs[1,0]
styles={'short_native':(blue,'o','Native: last1028'), 'long_fixed_short_coordinates':(green,'s','Native: full2000, short scale'),
        'long_native_coordinates':('#a47827','^','Native: full2000, own scale')}
for condition,(color,marker,label) in styles.items():
    rows=[r for r in attrib['rows'] if r['condition']==condition]
    ax.scatter([r['global_metrics']['D_stsp'] for r in rows],[r['event_comparison']['own_rank_diagnostic_all_offsets']['CDF_difference'][2] for r in rows],c=color,marker=marker,s=45,label=label)
for task,color,label in [('chua_common_short',red,'Library: last1028'),('chua_common_long','#653b73','Library: full2000')]:
    r=sids[task];ax.scatter(r['global_metrics']['D_stsp'],r['event_comparison']['own_rank_diagnostic_all_offsets']['CDF_difference'][2],c=color,marker='D',s=55,label=label)
ax.axhline(0,c='#aaaaaa',lw=1);ax.set_xlabel('Native geometry KL (lower is better)');ax.set_ylabel('Rank-CDF difference at100');ax.set_title('(c) Chua common origin2000: same unseen future');ax.legend(fontsize=7,loc='upper right')
ax=axs[1,1]
for i,name in enumerate(systems):
    vals=[r['metrics']['D_stsp_native'] for r in native['rows'] if r['system']==name and 'seed' in r]
    no_noise=next(r for r in attrib['rows'] if r['system']==name and r['condition']=='zero_attention_noise')['global_metrics']['D_stsp']
    ax.scatter(np.full(3,i)-.12,vals,c=blue,s=35,label='Native draws' if i==0 else None)
    ax.scatter(i+.12,no_noise,c=green,marker='x',s=60,label='Attention noise set to zero' if i==0 else None)
ax.set_yscale('log');ax.set_xticks(range(3),['Lorenz63','Chua','SprottM']);ax.set_ylabel('Native geometry KL (log scale)');ax.set_title('(d) Removing exploration harms global reconstruction');ax.legend(fontsize=8)
fig.suptitle('Released DynaMix examples: event-law residuals weaken under ordinary controls',fontsize=13)
fig.supxlabel('Exploratory analyses of three public fixtures; draws are not independent systems.\nAll-offset and rank statistics are post-pilot diagnostics, not confirmatory tests or a new-method result.',fontsize=9)
for extension in ['png','svg','pdf']:fig.savefig(OUT/('event_law_controls.'+extension),dpi=170)
plt.close(fig)
print(str(OUT/'event_law_controls.png'))
