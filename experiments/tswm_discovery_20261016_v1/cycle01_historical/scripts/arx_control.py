"""Classical ARX(1,0) on identical TRAIN future-transition supervision."""
import json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from synthetic_data import split,pulse_truth
cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
def rollout(h,u,w):
    x=h[:,-1,0].copy();out=np.empty_like(u)
    for t in range(u.shape[1]):
        x=w[0]*x+w[1]*u[:,t,0]+w[2];out[:,t,0]=x
    return out
result=[]
for ex in ['smooth','white']:
    tr=split('train',cfg,ex);va=split('validation',cfg,ex)
    past=np.concatenate([tr['history'][:,-1:,0],tr['target'][:,:-1,0]],axis=1)
    A=np.stack([past,tr['future'][...,0],np.ones_like(past)],axis=-1).reshape(-1,3).astype('float64')
    yy=tr['target'].reshape(-1).astype('float64')
    candidates=[]
    for lam in [1e-6,1e-4,1e-2,1.]:
        w=np.linalg.solve(A.T@A+lam*np.eye(3),A.T@yy)
        v=float(np.mean((rollout(va['history'],va['future'],w)-va['target'])**2))
        candidates.append((v,lam,w))
    v,lam,w=min(candidates,key=lambda x:x[0]);data=split('discovery',cfg,ex)
    base=rollout(data['history'],data['future'],w);ps=[]
    for pos in [0,12,24]:
        u=data['future'].copy();u[:,pos,0]+=.5
        delta=rollout(data['history'],u,w)-base;truth=pulse_truth(data,cfg,pos,.5)
        ps.append({'position':pos,'effect_nmse':float(np.sum((delta-truth)**2)/np.sum(truth**2)),'late_tail_gain':float(np.sum(delta[:,pos+5:]*truth[:,pos+5:])/np.sum(truth[:,pos+5:]**2))})
    result.append({'model':'classical_ARX1','excitation':ex,'coefficients':w.tolist(),'validation_mse':v,'ridge':lam,'factual_mse':float(np.mean((base-data['target'])**2)),'pulses':ps,'supervision':'same train target transitions; no true parameters used to fit'})
(ROOT/'outputs/arx_discovery.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
