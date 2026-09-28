"""Six descriptive figures; case selection by support quantile, never by error."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from common import *
from score_support import summary,fit_scale,scaled

plt.rcParams.update({'font.size':10,'figure.dpi':130,'savefig.dpi':180})
PLOTS=ROOT/'plots';PLOTS.mkdir(exist_ok=True)

def save(fig,name):
    fig.tight_layout();fig.savefig(PLOTS/(name+'.png'));fig.savefig(PLOTS/(name+'.pdf'));plt.close(fig)

def main():
    scores=pd.read_csv(OUT/'support_scores.csv');fd=pd.read_csv(OUT/'failure_detection_metrics.csv')
    rc=pd.read_csv(OUT/'risk_coverage_metrics.csv');hz=pd.read_csv(OUT/'horizon_reliability_metrics.csv')
    sel=pd.read_csv(OUT/'selective_rollout_metrics.csv')
    focus=['U_ensemble','D_state_knn','D_action','D_joint_knn','D_cond','D_perp']
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,name in zip(axs.flat,focus):
        x=scores[name].to_numpy();y=scores.BIS_MAE_full_5min.to_numpy()
        cuts=np.quantile(x,np.linspace(0,1,11));xx=[];yy=[];lo=[];hi=[]
        for j in range(10):
            m=(x>=cuts[j])&(x<cuts[j+1] if j<9 else x<=cuts[j+1])&np.isfinite(y)
            if m.sum()==0:continue
            xx.append(float(np.median(x[m])))
            p=scores.loc[m,'subjectid'];means=pd.DataFrame({'p':p,'e':y[m]}).groupby('p').e.mean().to_numpy()
            yy.append(float(means.mean()))
            rng=np.random.default_rng(0);draw=means[rng.integers(len(means),size=(1000,len(means)))].mean(1)
            a,b=np.quantile(draw,[.025,.975]);lo.append(a);hi.append(b)
        ax.plot(xx,yy,'o-');ax.fill_between(xx,lo,hi,alpha=.2)
        ax.set(title=name,xlabel='Support/risk score',ylabel='Full 5-min BIS MAE')
    save(fig,'figure1_support_vs_error')
    names=['Random','U_ensemble','D_raw_physiology','D_state_knn','D_action',
           'D_joint_knn','D_cond','D_perp','U_plus_best_geometry']
    f=fd[(fd.label=='top20')&fd.signal.isin(names)].set_index('signal').loc[names]
    fig,axs=plt.subplots(1,2,figsize=(12,5));pos=np.arange(len(f))
    for ax,col,title in zip(axs,['auroc','auprc'],['High-error AUROC','High-error AUPRC']):
        ax.barh(pos,f[col]);ax.set_yticks(pos,names);ax.invert_yaxis();ax.set(xlim=(0,1),title=title)
    save(fig,'figure2_failure_detection')
    fig,ax=plt.subplots(figsize=(8,5))
    for name in names:
        d=rc[rc.signal==name].sort_values('coverage')
        ax.plot(d.coverage,d.bis_mae,marker='o',label=name)
    ax.set(xlabel='Retained coverage',ylabel='Equal-patient BIS MAE',title='Risk–coverage')
    ax.legend(fontsize=7,ncol=2);save(fig,'figure3_risk_coverage')
    fig,axs=plt.subplots(1,3,figsize=(12,4));x=hz.horizon_seconds/60
    for ax,col,label in zip(axs,['bis_mae','ensemble_variance','dynamic_support'],
                             ['BIS MAE','Ensemble BIS variance','Horizon-matched support distance']):
        ax.plot(x,hz[col]);ax.set(xlabel='Horizon (minutes)',ylabel=label)
    save(fig,'figure4_horizon_reliability')
    # Four cases from support quantiles, not selected using model errors.
    idx=np.load(ARRAYS/'test_index.npy');sub=np.load(ARRAYS/'test_subject.npy')
    support=scores.D_joint_knn.to_numpy();order=np.argsort(support)
    chosen=[];used=set()
    for q in [.1,.35,.65,.9]:
        center=int(q*(len(order)-1))
        for radius in range(len(order)):
            j=int(order[min(len(order)-1,center+radius)])
            if sub[j] not in used:
                chosen.append(j);used.add(sub[j]);break
    preds=np.stack([np.load(ARRAYS/f'test_pred_seed{s}.npy',mmap_mode='r')[:,:,0]
                    for s in CFG['seeds']]).mean(0)
    target=np.load(ARRAYS/'test_target.npy',mmap_mode='r')[:,:,0]
    action=np.load(ARRAYS/'test_action.npy',mmap_mode='r')
    dynamic=np.load(ARRAYS/'test_dynamic_support.npy',mmap_mode='r')
    norm=json.loads((SOURCE/'outputs/normalization.json').read_text())
    threshold=float(sel[(sel.signal=='Dynamic support')&(sel.target_validation_accepted_mae==3.5)].threshold.iloc[0])
    fig,axs=plt.subplots(4,3,figsize=(13,11),sharex='col')
    t=np.arange(1,F+1)*10
    for row,j in enumerate(chosen):
        accepted=np.flatnonzero(dynamic[j]>threshold);cut=int(accepted[0]) if len(accepted) else F
        axs[row,0].plot(t,target[j],label='Observed BIS');axs[row,0].plot(t,preds[j],label='Ensemble BIS')
        axs[row,0].axvline(cut*10,color='k',ls=':',alpha=.6)
        dose=np.expm1(action[j]*np.array(norm['action_std'])+np.array(norm['action_mean']))/10
        axs[row,1].plot(t,dose[:,0],label='PPF');axs[row,1].plot(t,dose[:,1],label='RFTN')
        axs[row,2].plot(t,dynamic[j],label='Support');axs[row,2].axhline(threshold,color='r',ls='--',label='Threshold')
        axs[row,0].set_ylabel(f'Case {idx[j,0]}\nBIS')
    for a,title in zip(axs[0],['BIS and accepted cutoff','Future administration (mL/10s)','Dynamic support score']):a.set_title(title)
    for a in axs[-1]:a.set_xlabel('Seconds after anchor')
    for a in axs[0]:a.legend(fontsize=7)
    save(fig,'figure5_representative_patients')
    # PCA is only a display of TRAIN joint features; all primary scores above
    # were computed in original space. Quadrants use test error descriptively.
    tr_z=np.load(ARRAYS/'train_z.npy',mmap_mode='r');te_z=np.load(ARRAYS/'test_z.npy',mmap_mode='r')
    tr_a=np.load(ARRAYS/'train_action.npy',mmap_mode='r');te_a=np.load(ARRAYS/'test_action.npy',mmap_mode='r')
    rng=np.random.default_rng(44);take=rng.choice(len(tr_z),size=12000,replace=False)
    zs=fit_scale(np.asarray(tr_z));acts=fit_scale(summary(np.asarray(tr_a)),robust=True)
    train_q=np.concatenate([scaled(np.asarray(tr_z[take]),zs),scaled(summary(np.asarray(tr_a[take])),acts)],1)
    test_q=np.concatenate([scaled(np.asarray(te_z),zs),scaled(summary(np.asarray(te_a)),acts)],1)
    pca=PCA(n_components=2,random_state=0).fit(train_q);base=pca.transform(train_q)
    err=scores.BIS_MAE_full_5min.to_numpy();supported=support<=np.quantile(support,.8)
    lowerror=err<np.quantile(err[np.isfinite(err)],.8)
    categories=[('supported, low error',supported&lowerror,'#277da1'),
                ('low support, high error',~supported&~lowerror,'#d62828'),
                ('low support, low error',~supported&lowerror,'#f77f00'),
                ('supported, high error',supported&~lowerror,'#6a4c93')]
    fig,ax=plt.subplots(figsize=(8,6));ax.scatter(base[:,0],base[:,1],s=2,c='lightgray',alpha=.15,label='TRAIN')
    for label,mask,color in categories:
        ii=np.flatnonzero(mask);ii=rng.choice(ii,size=min(len(ii),2000),replace=False)
        xy=pca.transform(test_q[ii]);ax.scatter(xy[:,0],xy[:,1],s=7,alpha=.45,label=f'{label} (n={mask.sum()})',c=color)
    ax.set(xlabel='PC1',ylabel='PC2',title='Joint support space: display only');ax.legend(fontsize=7)
    save(fig,'figure6_support_pca')
    print('FIGURES_COMPLETE',flush=True)

if __name__=='__main__':main()
