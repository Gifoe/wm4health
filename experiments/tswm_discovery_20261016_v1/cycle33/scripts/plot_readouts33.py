"""Render verified values; this script does not fit or select a model."""
from native_common33 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    evaluation=ROOT/'outputs/test_evaluation_v1';analysis=ROOT/'outputs/analysis_v1'
    a=json.loads((analysis/'analysis.json').read_text())
    p=json.loads((evaluation/'provenance_closure.json').read_text());assert p['status']=='passed'
    assert a['evaluation_sha256']==p['evaluation_sha256']==sha(evaluation/'result.json')
    er=json.loads((evaluation/'result.json').read_text())
    out=ROOT/'outputs/figures_v1';assert not out.exists();out.mkdir(parents=True)
    palette=['#0072B2','#E69F00','#009E73','#CC79A7']
    labels=['Native mixed','Native prediction only','Wider local','Global MLP']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
    plotrows=[]
    for row,ds in enumerate(['NP','PJM']):
        for j,arm in enumerate(ARMS):
            hs=[json.loads((FITS/ds/f'{arm}_seed{s}'/'history.json').read_text()) for s in SEEDS]
            vals=np.array([[r['forecast_mae'] for r in h[1:]] for h in hs])
            mu=vals.mean(0);lo=vals.min(0);hi=vals.max(0);ep=np.arange(1,21)
            axs[row,0].plot(ep,mu,color=palette[j],label=labels[j],lw=1.6)
            axs[row,0].fill_between(ep,lo,hi,color=palette[j],alpha=.12,linewidth=0)
            scores=[next(r['mae'] for r in er['records'] if r['dataset']==ds and r['method']==arm and
                         r['seed']==s and r['selection']=='forecast') for s in SEEDS]
            for s,y in zip(SEEDS,scores):plotrows.append({'dataset':ds,'arm':arm,'seed':s,'test_mae':y})
            axs[row,1].scatter(j+np.array([-.09,0,.09]),scores,color=palette[j],s=20,zorder=3)
            axs[row,1].errorbar(j,np.mean(scores),yerr=np.std(scores,ddof=1),color=palette[j],fmt='_',ms=15,capsize=4,lw=1.4)
        reference=next(r['mae'] for r in er['records'] if r['dataset']==ds and r['method']=='original_stage2_reference')
        axs[row,1].axhline(reference,color='#555555',ls='--',lw=1,label='Original stage 2 reference')
        axs[row,0].set(xlabel='Decoder epoch',ylabel='Validation MAE',title=f'{ds}: fixed-budget training',xticks=[1,5,10,15,20])
        axs[row,1].set(ylabel='Test MAE',title=f'{ds}: validation-selected checkpoints',xticks=np.arange(4),xticklabels=['Mixed','Pred. only','Wider local','Global MLP'])
        for ax in axs[row]:ax.grid(axis='y',alpha=.2);ax.tick_params(axis='both',length=3)
    axs[0,0].legend(fontsize=8,frameon=False,ncol=2)
    axs[0,1].legend(fontsize=8,frameon=False)
    fig.suptitle('Frozen native WorldTS: ordinary readout attribution',fontsize=13)
    for suffix in ['png','svg','pdf']:fig.savefig(out/f'readout_attribution.{suffix}',dpi=180)
    plt.close(fig)
    pd.DataFrame(plotrows).to_csv(out/'plotted_test_values.csv',index=False)
    dump(out/'manifest.json',{'script_sha256':sha(__file__),'analysis_sha256':sha(analysis/'analysis.json'),
        'provenance_closure_sha256':sha(evaluation/'provenance_closure.json'),
        'files':[{'path':f.name,'sha256':sha(f)} for f in sorted(out.iterdir()) if f.is_file()]})
    (out/'CAPTION.md').write_text('左列为验证MAE：实线是3 decoder seeds均值，阴影为最小至最大，不是置信带；隐藏随机epoch0便于阅读。右列点为3 seeds，横线为均值、误差条为seed样本标准差；仅改变decoder初始化，stage1固定同一seed。虚线原stage2只是旧训练reference，预算/初始化不同。NP/PJM同为既有电力任务，不能声称跨领域泛化。',encoding='utf-8')
    print(json.dumps({'status':'rendered_verified_values','figures':3,'test_points':len(plotrows)}))
if __name__=='__main__':main()
