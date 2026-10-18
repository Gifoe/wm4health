from common88_v1 import *
import numpy as np
from plan_data88_v1 import observe,advance

Q=65;T=96;START=48;LAGS=4

def predict(obs,actions,weights):
    history=[obs[:,t].astype(np.float64) for t in range(Q)];pred=[]
    for t in range(Q-1,T):
        x=np.concatenate([np.ones((len(obs),1))]+[history[-1-l] for l in range(LAGS)]+[actions[:,t-l].astype(np.float64) for l in range(LAGS)],axis=-1)
        y=x@weights;assert np.isfinite(y).all();pred.append(y);history.append(y)
    return np.stack(pred,axis=1)

def main():
    manifest=read(O/'PLAN_DATA88_V1.json');data={}
    for split in ['train','val']:
        row=next(r for r in manifest['files'] if r['split']==split);assert sha(row['path'])==row['sha256']
        with np.load(row['path']) as z:data[split]={k:z[k] for k in z.files}
    train=data['train'];val=data['val'];obs=train['observations'].astype(np.float64);act=train['actions'].astype(np.float64)
    n=len(obs);steps=192-LAGS+1
    x=np.concatenate([np.ones((n,steps,1))]+[obs[:,LAGS-1-l:192-l] for l in range(LAGS)]+[act[:,LAGS-1-l:192-l] for l in range(LAGS)],axis=-1).reshape(n*steps,-1)
    y=obs[:,LAGS:193].reshape(n*steps,16);xx=x.T@x/len(x);xy=x.T@y/len(x);reg=np.eye(x.shape[1]);reg[0,0]=0
    vobs=val['observations'][:,START:START+T+1];va=val['actions'][:,START:START+T];truth=vobs[:,Q:];family=val['family'];choices=[];best=None
    for lam in [1e-6,1e-4,.01,.1,1.,10.]:
        weights=np.linalg.solve(xx+lam*reg,xy);pred=predict(vobs,va,weights);mse=float(np.mean((pred-truth)**2));choices.append(dict(ridge=lam,mse=mse))
        if best is None or mse<best[0]:best=(mse,lam,weights,pred)
    _,lam,weights,base=best
    # Edited branches are opened only after the ordinary ARX hyperparameter is selected.
    rows=[];arrays=dict(weights=weights,base=base,truth=truth,family=family)
    for cut in [80,88]:
        for sign in [-1,1]:
            a=va.copy();a[:,cut:,0]=sign*.8;target=truth.copy()
            for f in [0,1]:
                ids=np.flatnonzero(family==f);state=val['states'][ids,START+cut].copy()
                for t in range(cut,T):
                    state=advance(state,a[ids,t].astype(np.float64),val['parameters'][ids],f)
                    sensed=(observe(state)+val['observation_noise'][ids,START+t+1]).astype(np.float32)
                    target[ids,t-(Q-1)]=((sensed-val['center'])/val['scale']).astype(np.float32)
            edited=predict(vobs,a,weights);early=cut-(Q-1)
            for f in [0,1]:
                choose=family==f;delta=edited[choose]-base[choose];response=target[choose]-truth[choose]
                rows.append(dict(family=f,cut=cut,sign=sign,early_max=float(abs(delta[:,:early]).max()),early_rms=float(np.sqrt(np.mean(delta[:,:early]**2))),post_true_response_rms=float(np.sqrt(np.mean(response[:,early:]**2))),post_model_response_rms=float(np.sqrt(np.mean(delta[:,early:]**2))),post_response_error_rms=float(np.sqrt(np.mean((delta[:,early:]-response[:,early:])**2))),edited_mse=float(np.mean((edited[choose]-target[choose])**2))))
            arrays[f'cut{cut}_sign{sign}']=edited
    factual=[]
    for f in [0,1]:
        choose=family==f;mse=float(np.mean((base[choose]-truth[choose])**2));p=float(np.mean((vobs[choose,Q-1:Q]-truth[choose])**2));zero=float(np.mean(truth[choose]**2))
        factual.append(dict(family=f,mse=mse,persistence_ratio=mse/p,train_mean_ratio=mse/zero))
    root=Path(STORAGE['artifact_root'])/'cycle88';root.mkdir(parents=True,exist_ok=True);path=root/'ridge_predictions88_v1.npz'
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    write(O/'RIDGE_PLAN_RESULTS88_V1.json',dict(utc=utc(),status='ORDINARY_ARX4_DIAGNOSTIC_COMPLETE',training_rows=len(x),parameters=weights.size,selected_ridge=lam,validation_grid=choices,factual=factual,rows=rows,payload_path=str(path),payload_sha256=sha(path),selection_uses_edited_plans=False,held_out_eval_accessed=False,new_method=False))
    log(dict(stage='ordinary_arx4_closed',selected_ridge=lam,factual=factual))

if __name__=='__main__':main()
