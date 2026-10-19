import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
colors=['#264653','#2a9d8f','#e9c46a','#e76f51']

def main():
    a=json.loads((OUT/'analysis_summary.json').read_text());r=a['synthetic_screen']
    order=['worldts_two_stage','worldts_e2e','worldts_fir','gru'];by={x['model']:x for x in r}
    labels=['Native\n2-stage','Native\ne2e','Ordinary\nFIR','Smaller\nGRU']
    fig,axes=plt.subplots(1,2,figsize=(10,3.2))
    for ax,key,title in zip(axes,['factual_mse','effect_nmse'],['Factual forecast error','Pulse-response error']):
        vals=[by[x][key]['mean'] for x in order];sd=[by[x][key]['sample_std'] for x in order]
        ax.bar(range(4),vals,color=colors,yerr=sd,capsize=3)
        ax.set_xticks(range(4),labels);ax.set_yscale('log');ax.set_ylabel(key.replace('_',' '));ax.set_title(title)
    fig.suptitle('Known FIR / GRU fixes the native short-support failure (3 training seeds)',fontsize=11)
    fig.tight_layout();fig.savefig(FIG/'action_memory_screen.png',dpi=180);fig.savefig(FIG/'action_memory_screen.pdf');plt.close(fig)

    ref=json.loads((OUT/'neural_range_refinement.json').read_text());fig,ax=plt.subplots(figsize=(7.2,3.5))
    x=np.arange(3)
    for offset,method,color,label in [(-.19,'coarse',colors[0],'7 angles'),(.19,'dense',colors[1],'256 angles')]:
        vals=[r['true_cf_coverage']*100 for r in ref['records'] if r['method']==method]
        ax.bar(x+offset,vals,width=.36,color=color,label=label)
    ax.axhline(100,color='gray',linestyle='--',linewidth=1,label='True-kernel oracle: 100%')
    ax.set_ylim(0,110);ax.set_xticks(x,['Seed 17','Seed 31','Seed 59']);ax.set_ylabel('True counterfactual contained (%)')
    ax.set_title('A denser coupling grid does not repair fitted-kernel error');ax.legend(ncols=3,loc='lower center',bbox_to_anchor=(.5,-.38),fontsize=9)
    fig.tight_layout();fig.savefig(FIG/'neural_range_falsification.png',dpi=180);fig.savefig(FIG/'neural_range_falsification.pdf');plt.close(fig)

    d=json.loads((OUT/'shared_parameter_confirmation.json').read_text());rows=d['summaries']
    models=['gauge_only','statistical_only_fixed_common_noise','joint_paired_tube','joint_shared_parameter_tube','oracle_known_kernel_family']
    names=['Coupling only','Statistics only\n(fixed noise)','Joint norm\ntube','Shared-parameter\nTaylor tube','True kernel\nfamily']
    chosen={r['method']:r for r in rows if r['training_n']==16384 and r['horizon']==64}
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.4))
    cs=[colors[0],colors[3],'#648c9b',colors[1],'#929292']
    for ax,key,title,scale in zip(axes,['whole_family_coverage','mean_width','sign_certifiable_fraction'],['Whole-family containment','Mean interval width','Intervals excluding zero'],[100,1,100]):
        vals=[chosen[m][key]*scale for m in models];ax.bar(range(5),vals,color=cs)
        if key=='sign_certifiable_fraction':
            invalid=[chosen[m]['wrong_signed_claim_fraction']*100 for m in models]
            ax.bar(range(5),invalid,facecolor='none',edgecolor='black',hatch='///',linewidth=.8,label='Invalid for true family')
            ax.legend(fontsize=7,loc='upper right')
        ax.set_xticks(range(5),names,rotation=30,ha='right',fontsize=8);ax.set_title(title)
        if scale==100:ax.set_ylim(0,110);ax.set_ylabel('%')
    fig.suptitle('Fresh confirmation: N=16,384; H=64; 64 independent fits × 32 paths',fontsize=11)
    fig.tight_layout();fig.savefig(FIG/'counterfactual_intervals.png',dpi=180);fig.savefig(FIG/'counterfactual_intervals.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(9.6,3.5))
    for method,label,c in [('joint_paired_tube','Joint norm tube','#648c9b'),('joint_shared_parameter_tube','Shared Taylor',colors[1]),('oracle_known_kernel_family','True-kernel family','#929292')]:
        rr=sorted([r for r in rows if r['horizon']==64 and r['method']==method],key=lambda r:r['training_n'])
        axes[0].plot([r['training_n'] for r in rr],[r['mean_width'] for r in rr],'o-',label=label,color=c)
        axes[1].plot([r['training_n'] for r in rr],[100*r['sign_certifiable_fraction'] for r in rr],'o-',label=label,color=c)
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('Randomized training transitions');ax.grid(alpha=.15)
    axes[0].set_yscale('log');axes[0].set_ylabel('Mean interval width');axes[1].set_ylabel('Certified sign (%)')
    axes[1].legend(fontsize=9);fig.suptitle('Coverage can be restored while useful decisions remain scarce',fontsize=11)
    fig.tight_layout();fig.savefig(FIG/'sample_size_tradeoff.png',dpi=180);fig.savefig(FIG/'sample_size_tradeoff.pdf');plt.close(fig)
    generic=[]
    for p in (OUT/'lirpa').glob('rep*_q*_H*.json'):
        if p.name.endswith('.partial.json'):continue
        item=json.loads(p.read_text())
        if item['horizon'] in [32,64]:generic.append(item)
    if generic:
        fig,ax=plt.subplots(figsize=(6.8,3.5))
        for H,color in [(32,colors[0]),(64,colors[1])]:
            items=sorted([r for r in generic if r['horizon']==H],key=lambda r:(r['rep'],r['query']))
            if not items:continue
            ratios=[r['results'][0]['width']/r['same_set_shared_width'] for r in items]
            ax.plot(np.arange(len(items)),ratios,'o-',color=color,label=f'H={H}: {len(items)} matched cases')
        ax.axhline(1,color='gray',linestyle='--',linewidth=1)
        ax.set_xlabel('Matched case (2 fits × 4 queries per horizon)')
        ax.set_ylabel('CROWN width / specialized Taylor width')
        ax.set_title('Generic verification is a necessary novelty control')
        ax.legend(fontsize=9);fig.tight_layout();fig.savefig(FIG/'generic_verifier_falsifier.png',dpi=180);fig.savefig(FIG/'generic_verifier_falsifier.pdf');plt.close(fig)
    print('WROTE_FIGURES',len(list(FIG.iterdir())))
if __name__=='__main__':main()
