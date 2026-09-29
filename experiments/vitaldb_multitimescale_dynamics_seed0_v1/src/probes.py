"""Frozen fast/slow branch probes; TRAIN fit, VAL alpha, equal-patient TEST R²."""
import json
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from models import *

OUT=ROOT/'outputs';ARRAYS=OUT/'arrays'
AUDIT=(ROOT/'../vitaldb_physiological_grounding_audit_seed0_v1').resolve()
def old(split,key):return np.load(AUDIT/'outputs/arrays'/f'{split}_{key}.npy',mmap_mode='r')
def make_targets(split):
    current=np.asarray(old(split,'current'));future=np.asarray(old(split,'target'))
    cm=np.asarray(old(split,'current_fresh'));fm=np.asarray(old(split,'fresh'))
    raw=np.asarray(old(split,'raw_history'));drug=np.asarray(old(split,'drug_history'))
    result={}
    for j,var in enumerate(('BIS','MAP')):
        result[f'future_delta_{var}_30s']=(future[:,0,j]-current[:,j],cm[:,j]&fm[:,0,j]&np.isfinite(future[:,0,j])&np.isfinite(current[:,j]))
        result[f'future_delta_{var}_60s']=(future[:,1,j]-current[:,j],cm[:,j]&fm[:,1,j]&np.isfinite(future[:,1,j])&np.isfinite(current[:,j]))
        result[f'future_delta_{var}_180s']=(future[:,2,j]-current[:,j],cm[:,j]&fm[:,2,j]&np.isfinite(future[:,2,j])&np.isfinite(current[:,j]))
        result[f'future_delta_{var}_300s']=(future[:,3,j]-current[:,j],cm[:,j]&fm[:,3,j]&np.isfinite(future[:,3,j])&np.isfinite(current[:,j]))
    for label,col in [('recent_BIS_delta_300s',12),('history_BIS_mean_300s',6),('history_MAP_mean_300s',8)]:
        y=raw[:,col];result[label]=(y,np.isfinite(y))
    for label,col in [('recent_PPF_action_delta_300s',6),('recent_RFTN_action_delta_300s',7),
                      ('history_PPF_action_mean_300s',2),('history_RFTN_action_mean_300s',3)]:
        y=drug[:,col];result[label]=(y,np.isfinite(y))
    return result

@torch.no_grad()
def extract(model,store,split,device):
    ds=Windows(store,split);n=len(ds)
    fast=np.lib.format.open_memmap(ARRAYS/f'{split}_fast.npy','w+',dtype='float32',shape=(n,32))
    slow=np.lib.format.open_memmap(ARRAYS/f'{split}_slow.npy','w+',dtype='float32',shape=(n,32))
    at=0
    for b in DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True):
        b=to_device(b,device);f,s=model.encode(b);m=len(f)
        fast[at:at+m]=f.cpu().numpy();slow[at:at+m]=s.cpu().numpy();at+=m
    fast.flush();slow.flush();print('PROBE_EXTRACT',split,n,flush=True)

def weighted_r2(y,p,subject,valid):
    m=valid&np.isfinite(y)&np.isfinite(p)
    # direct sufficient statistics; avoid interpreting overlapping windows as independent.
    ids=np.unique(subject[m]);stats=[]
    for s in ids:
        z=m&(subject==s);a=y[z].astype(float);b=p[z].astype(float)
        stats.append([a.mean(),np.mean(a*a),np.mean((a-b)**2)])
    q=np.asarray(stats);v=q.mean(0);return float(1-v[2]/(v[1]-v[0]**2))

def r2_with_ci(y,p,subject,valid):
    m=valid&np.isfinite(y)&np.isfinite(p);ids=np.unique(subject[m]);q=[]
    for s in ids:
        ix=m&(subject==s);a=y[ix].astype(float);b=p[ix].astype(float)
        q.append([a.mean(),np.mean(a*a),np.mean((a-b)**2),np.mean(np.abs(a-b))])
    q=np.asarray(q);v=q.mean(0);r2=1-v[2]/(v[1]-v[0]**2)
    rng=np.random.default_rng(0);draw=q[rng.integers(len(q),size=(1000,len(q)))].mean(1)
    rr=1-draw[:,2]/(draw[:,1]-draw[:,0]**2)
    return dict(r2=float(r2),r2_ci_low=float(np.nanquantile(rr,.025)),r2_ci_high=float(np.nanquantile(rr,.975)),
                mae=float(v[3]),patients=len(ids),windows=int(m.sum()))

def main():
    seed_all(0);store=Store();device='cuda' if torch.cuda.is_available() else 'cpu'
    model=MTDynamics().to(device);ck=torch.load(ROOT/'checkpoints/MT-Dynamics.pt',map_location=device,weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    for split in ('train','val','test'):extract(model,store,split,device)
    features={s:{'z_fast':np.asarray(np.load(ARRAYS/f'{s}_fast.npy',mmap_mode='r')),
                 'z_slow':np.asarray(np.load(ARRAYS/f'{s}_slow.npy',mmap_mode='r'))} for s in ('train','val','test')}
    for s in features:features[s]['concat']=np.concatenate([features[s]['z_fast'],features[s]['z_slow']],1)
    targets={s:make_targets(s) for s in features}
    subjects={s:np.asarray(old(s,'subject')) for s in features}
    rng=np.random.default_rng(0);train_sample=rng.choice(len(features['train']['z_fast']),size=min(CFG['probe_train_windows'],len(features['train']['z_fast'])),replace=False)
    rows=[]
    for label in targets['train']:
        ys={s:targets[s][label][0] for s in features};ms={s:targets[s][label][1] for s in features}
        for rep in ('z_fast','z_slow','concat'):
            x={s:np.nan_to_num(features[s][rep]) for s in features}
            tr=train_sample[ms['train'][train_sample]];valid_val=ms['val'];valid_test=ms['test']
            if min(len(tr),valid_val.sum(),valid_test.sum())<100:continue
            sc=StandardScaler().fit(x['train'][tr]);xt=sc.transform(x['train'][tr]);xv=sc.transform(x['val'][valid_val]);xe=sc.transform(x['test'])
            best=-np.inf;chosen=None;alpha=None
            for a in CFG['ridge_alpha_grid']:
                m=Ridge(alpha=a,solver='svd').fit(xt,ys['train'][tr])
                score=weighted_r2(ys['val'][valid_val],m.predict(xv),subjects['val'][valid_val],np.ones(valid_val.sum(),bool))
                if score>best:best=score;chosen=m;alpha=a
            pred=chosen.predict(xe)
            metrics=r2_with_ci(ys['test'],pred,subjects['test'],valid_test)
            rows.append(dict(representation=rep,target=label,alpha=alpha,val_r2=best,training_windows=len(tr),**metrics))
        print('PROBE',label,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'timescale_probe_metrics.csv',index=False)
    print('PROBES_COMPLETE',flush=True)
if __name__=='__main__':main()
