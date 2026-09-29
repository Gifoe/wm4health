"""Descriptive all-lambda 300-s geometry versus held-out forecast plot points."""
import json
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from evaluate import ROOT,OUT,CFG,SOURCE,ARR,Store,Windows,RSSM,to_device,seed_all,old,target,feat,weighted_r2,weighted_metrics

@torch.no_grad()
def extract_point(model,store,split):
    ds=Windows(store,split);device=next(model.parameters()).device;dz=[];pred=[]
    for b in DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True):
        p,z,r=model(to_device(b,device),'true',True)
        dz.append((r[:,-1]-z).cpu().numpy())
        if split=='test':pred.append(p[:,-1,0].cpu().numpy()*store.norm['state_std'][0]+store.norm['state_mean'][0])
    return np.concatenate(dz),np.concatenate(pred) if pred else None

def main():
    seed_all(0);store=Store();device='cuda' if torch.cuda.is_available() else 'cpu';selected=json.loads((OUT/'selected_lambda.json').read_text());rows=[]
    for kind in ('state','transition'):
        for lam in CFG['lambda_grid']:
            name='State-Grounded' if kind=='state' else 'Transition-Grounded'
            if lam==selected[kind]:
                x={s:np.asarray(feat(name,s,'dz'))[:,3] for s in ('train','val','test')}
                pred=np.asarray(feat(name,'test','prediction'))[:,29,0]
            else:
                ck=torch.load(ROOT/'checkpoints'/f'{kind}_{lam:g}.pt',map_location=device,weights_only=False)
                model=RSSM().to(device);model.set_action_stats(store);model.load_state_dict(ck['model']);model.eval()
                x={};pred=None
                for s in ('train','val','test'):
                    x[s],q=extract_point(model,store,s)
                    if s=='test':pred=q
            y={s:target(s,'BIS',3)[0] for s in x};m={s:target(s,'BIS',3)[1] for s in x}
            sc=StandardScaler().fit(x['train'][m['train']]);xt=sc.transform(x['train'][m['train']]);xv=sc.transform(x['val'][m['val']]);xe=sc.transform(x['test'])
            best=-np.inf;probe=None;alpha=None
            for a in (.1,10.,1000.):
                q=Ridge(alpha=a,solver='cholesky').fit(xt,y['train'][m['train']]);score=weighted_r2(y['val'][m['val']],q.predict(xv),np.asarray(old('val','subject'))[m['val']],np.ones(m['val'].sum(),bool))
                if score>best:best=score;probe=q;alpha=a
            r2=weighted_metrics(y['test'],probe.predict(xe),np.asarray(old('test','subject')),m['test'])['r2']
            actual=np.asarray(old('test','target'))[:,3,0];valid=np.asarray(old('test','fresh'))[:,3,0]
            sub=np.asarray(old('test','subject'));mae=float(np.mean([np.mean(np.abs(pred[(sub==s)&valid]-actual[(sub==s)&valid])) for s in np.unique(sub[valid])]))
            rows.append(dict(model=name,lambda_weight=lam,selected=lam==selected[kind],delta_bis_r2_300s=r2,bis_mae_300s=mae,alpha=alpha,val_r2=best))
            pd.DataFrame(rows).to_csv(OUT/'lambda_geometry_forecast.csv',index=False)
            print('LAMBDA_SCAN',kind,lam,r2,mae,flush=True)
    # Fixed baseline checkpoint and original frozen latent cache.
    r=pd.read_csv(OUT/'transition_probe_metrics.csv').query('model=="Baseline" and target=="BIS" and horizon_seconds==300').iloc[0]
    f=pd.read_csv(OUT/'forecast_metrics.csv').query('model=="Baseline" and target=="BIS" and horizon_seconds==300').iloc[0]
    rows.append(dict(model='Baseline',lambda_weight=0,selected=True,delta_bis_r2_300s=r.r2,bis_mae_300s=f.mae,alpha=r.alpha,val_r2=r.val_r2))
    pd.DataFrame(rows).to_csv(OUT/'lambda_geometry_forecast.csv',index=False)
    print('LAMBDA_SCAN_COMPLETE',flush=True)
if __name__=='__main__':main()
