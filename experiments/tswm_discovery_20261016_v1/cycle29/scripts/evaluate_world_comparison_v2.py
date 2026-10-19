"""Frozen independent-world evaluation; never feeds future outputs to a model."""
import runtime_profile
import argparse,csv,json,pathlib,sys,time
import numpy as np
import torch
from world_data import ROOT,OUT as DATA,sha,dump,C,H
from world_models_v2 import build,NativeAttention,scale_context
from world_models import scale_context as legacy_scale_context
from train_world_comparison_v2 import KINDS,SEEDS,setup,hashes
OUT=ROOT/'outputs/world_evaluation_v2'
PLANS=['factual','zero','step','pulse'];HORIZONS=[32,64,128]
def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
def load_test():
    result={}
    for family in ['lti','wh']:
        with np.load(DATA/f'test_{family}.npz',allow_pickle=False) as f:result[family]={k:f[k] for k in ['u','y','clean']}
    return result
def inputs(data,lo=0,hi=None,legacy_normalization=False):
    if hi is None:hi=len(data['y'])
    y=torch.tensor(data['y'][lo:hi],device='cuda');u=torch.tensor(data['u'][lo:hi],device='cuda');n=len(y)
    cy=y[:,0,:C,None];cu=u[:,0,:C,None];future=u[:,:,C:,None]
    cy=cy[:,None].expand(n,4,C,1).reshape(n*4,C,1);cu=cu[:,None].expand(n,4,C,1).reshape(n*4,C,1)
    return (legacy_scale_context if legacy_normalization else scale_context)(cy,cu,future.reshape(n*4,H,1))
@torch.no_grad()
def neural_predict(model,data,chunked=False,legacy_normalization=False):
    result=[];start=time.monotonic()
    for lo in range(0,len(data['y']),8):
        cy,cu,fu,_,ym,ys=inputs(data,lo,min(lo+8,len(data['y'])),legacy_normalization=legacy_normalization)
        if not chunked:prediction=model.simulate(model.encode(cy,cu),fu)
        else:
            output=[];yy=cy;uu=cu
            for off in range(0,H,32):
                part=fu[:,off:off+32];p=model(yy,uu,part);output.append(p);yy=torch.cat((yy,p),1)[:,-C:];uu=torch.cat((uu,part),1)[:,-C:]
            prediction=torch.cat(output,1)
        result.append((prediction*ys+ym).reshape(-1,4,H).cpu().numpy())
    torch.cuda.synchronize();return np.concatenate(result),time.monotonic()-start
def metric_rows(family,method,seed,pred,data):
    rows=[];clean=data['clean'][:,:,C:];noisy=data['y'][:,:,C:]
    for i in range(len(pred)):
        for plan in range(4):
            for horizon in HORIZONS:
                p=pred[i,plan,:horizon];target=clean[i,plan,:horizon];pe=p-pred[i,0,:horizon];te=target-clean[i,0,:horizon];den=te@te
                rows.append(dict(family=family,system=i,method=method,seed=seed,plan=PLANS[plan],horizon=horizon,clean_rmse=float(np.sqrt(np.mean((p-target)**2))),noisy_rmse=float(np.sqrt(np.mean((p-noisy[i,plan,:horizon])**2))),effect_rmse=float(np.sqrt(np.mean((pe-te)**2))),effect_slope=float(pe@te/den) if den>1e-8 else None,informative_effect=bool(den>1e-8),finite=bool(np.isfinite(p).all())))
    return rows
@torch.no_grad()
def ablation_predict(model,data,mode):
    cy,cu,fu,_,ym,ys=inputs(data);state,code=model.encode(cy,cu);n=len(data['y'])
    if mode=='zero':code=torch.zeros_like(code)
    else:
        perm=((torch.arange(n,device='cuda')+1)%n)[:,None]*4+torch.arange(4,device='cuda')[None,:]
        code=code[perm.reshape(-1)]
    prediction=[]
    for lo in range(0,len(cy),32):prediction.append(model.simulate((state[lo:lo+32],code[lo:lo+32]),fu[lo:lo+32]))
    return ((torch.cat(prediction)*ys+ym).reshape(n,4,H).cpu().numpy())
def repeat_encoded(encoded,n):return tuple(x.expand((n,)+x.shape[1:]).contiguous() for x in encoded)
@torch.no_grad()
def timings(model,data,method,seed,legacy_normalization=False):
    rows=[];cy,cu,fu,_,_,_=inputs(data,0,1,legacy_normalization=legacy_normalization);cy=cy[:1];cu=cu[:1];one=fu[:1]
    for n in [1,16,64]:
        queries=one.repeat(n,1,1)
        for _ in range(2):encoded=model.encode(cy,cu);model.simulate(repeat_encoded(encoded,n),queries)
        enc_times=[];sim_times=[];torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize();start=time.monotonic();encoded=model.encode(cy,cu);torch.cuda.synchronize();enc_times.append(time.monotonic()-start)
            enc=repeat_encoded(encoded,n);torch.cuda.synchronize();start=time.monotonic();model.simulate(enc,queries);torch.cuda.synchronize();sim_times.append(time.monotonic()-start)
        rows.append(dict(method=method,seed=seed,n_plans=n,horizon=H,compile_context_s_median=float(np.median(enc_times)),cached_simulation_s_median=float(np.median(sim_times)),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),scope='samecontext repeatedplan-shape timing; no accuracy claim'))
    return rows
def arx_fit(y,u,order,lam):
    ids=np.arange(order,len(y));x=np.column_stack([y[ids-j] for j in range(1,order+1)]+[u[ids-j] for j in range(order)])
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-8);z=np.column_stack((np.ones(len(x)),(x-mean)/scale));penalty=np.eye(z.shape[1])*lam;penalty[0,0]=0
    beta=np.linalg.solve(z.T@z/len(z)+penalty,z.T@y[ids]/len(z));return dict(order=order,lam=lam,mean=mean,scale=scale,beta=beta)
def arx_predict(fit,y,u,horizon):
    history=list(y);start=len(history);order=fit['order'];pred=[]
    for t in range(start,start+horizon):
        x=np.r_[[history[t-j] for j in range(1,order+1)],[u[t-j] for j in range(order)]]
        value=float(np.r_[1,(x-fit['mean'])/fit['scale']]@fit['beta']);history.append(value);pred.append(value)
    return np.array(pred)
def arx_all(data,family):
    predictions=[];validation=[];fits=[]
    for i in range(len(data['y'])):
        observed=data['y'][i,0,:C].astype('float64');pastu=data['u'][i,0,:C].astype('float64');candidates=[]
        for order in [1,2,4,8,16]:
            for lam in [1e-4,.01,1.]:
                fit=arx_fit(observed[:96],pastu[:96],order,lam);p=arx_predict(fit,observed[:96],pastu,32);mse=float(np.mean((p-observed[96:])**2))
                candidates.append((mse,order,lam));validation.append(dict(family=family,system=i,order=order,lam=lam,context_validation_mse=mse))
        chosen=min(candidates);fit=arx_fit(observed,pastu,chosen[1],chosen[2]);out=[]
        for plan in range(4):out.append(arx_predict(fit,observed,data['u'][i,plan],H))
        predictions.append(out);fits.append(dict(family=family,system=i,selected_validation_mse=chosen[0],**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()}))
    return np.array(predictions),validation,fits
def pretrained():
    path=ROOT.parent/'cycle16/outputs/external/metaid/ckpt_430_fine-tuned_3100000_lower_noise.pt'
    expected='e8324becad85ac610ec94f8ae09048938b343761a5bcd3967a86353a5bf28197';assert sha(path)==expected
    assert set(torch.serialization.get_unsafe_globals_in_checkpoint(path))<={'argparse.Namespace'}
    with torch.serialization.safe_globals([argparse.Namespace]):payload=torch.load(path,map_location='cpu',weights_only=True)
    from transformer_sim import Config,TSTransformer
    model=NativeAttention();model.core=TSTransformer(Config(**payload['model_args']));model.core.load_state_dict(payload['model'],strict=True);model.n_in=30
    return model.cuda().eval(),expected
def main():
    setup();training=ROOT/'outputs/world_training_v2/summary.json';assert training.exists(),'Do not inspect test performance before all validation-selected training is complete'
    summary=json.loads(training.read_text());assert summary['status']=='all_training_complete' and summary['runs']==12 and summary['hashes']==hashes()
    assert not OUT.exists();OUT.mkdir();data=load_test();rows=[];latency=[];files=[];source_checks=[];start=time.monotonic()
    for seed in SEEDS:
        for kind in KINDS:
            folder=ROOT/'outputs/world_training_v2'/f'{kind}_{seed}';state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert state['hashes']==hashes()
            model=build(kind).cuda().eval();model.load_state_dict(state['model'],strict=True)
            for family,test in data.items():
                for mode in (['direct','chunk32'] if kind=='native_attention64' else ['direct']):
                    name=kind if mode=='direct' else kind+'_chunk32';pred,elapsed=neural_predict(model,test,mode=='chunk32');assert np.isfinite(pred).all()
                    path=OUT/f'{family}_{name}_{seed}.npz';np.savez_compressed(path,prediction=pred);files.append(dict(path=path.name,sha256=sha(path),seconds=elapsed,best_weight_sha256=sha(folder/'best.pt')))
                    rows+=metric_rows(family,name,seed,pred,test)
                if kind in ['compiled64','conditional128']:
                    for mode in (['zero','swap'] if kind=='compiled64' else ['swap']):
                        name=kind+('_zero_operator' if mode=='zero' else ('_swap_operator' if kind=='compiled64' else '_swap_code'))
                        pred=ablation_predict(model,test,mode);assert np.isfinite(pred).all();path=OUT/f'{family}_{name}_{seed}.npz';np.savez_compressed(path,prediction=pred)
                        files.append(dict(path=path.name,sha256=sha(path),best_weight_sha256=sha(folder/'best.pt'),ablation_plan_sha256=sha(ROOT/'EVAL_MECHANISM_ABLATION_PLAN.md')))
                        rows+=metric_rows(family,name,seed,pred,test)
            latency+=timings(model,data['lti'],kind,seed);del model;torch.cuda.empty_cache()
            print(json.dumps(dict(phase='evaluated',kind=kind,seed=seed,rows=len(rows))),flush=True)
    validation=[];fits=[]
    for family,test in data.items():
        pred,vr,fs=arx_all(test,family);validation+=vr;fits+=fs;rows+=metric_rows(family,'arx_context_selected',None,pred,test)
        path=OUT/f'{family}_arx_context_selected.npz';np.savez_compressed(path,prediction=pred);files.append(dict(path=path.name,sha256=sha(path)))
    model,weight_sha=pretrained()
    for family,test in data.items():
        pred,elapsed=neural_predict(model,test,legacy_normalization=True)
        # Verify the cached representation wrapper against the original published forward.
        cy,cu,fu,_,_,_=inputs(test,0,1,legacy_normalization=True);qu=torch.cat((cu[:,-30:],fu),1);qy=torch.cat((cy[:,-30:],torch.zeros_like(fu)),1)
        with torch.no_grad():a=model(cy,cu,fu);b=model.core(cy,cu,qu,qy,30)[0]
        error=float((a-b).abs().max());assert error<2e-5;source_checks.append(dict(family=family,pretrained_wrapper_max_abs=error))
        rows+=metric_rows(family,'pretrained_native_icl',None,pred,test);path=OUT/f'{family}_pretrained_native_icl.npz';np.savez_compressed(path,prediction=pred);files.append(dict(path=path.name,sha256=sha(path),seconds=elapsed,weight_sha256=weight_sha))
    latency+=timings(model,data['lti'],'pretrained_native_icl',None,legacy_normalization=True)
    csvwrite(OUT/'per_system_metrics.csv',rows);csvwrite(OUT/'latency.csv',latency);csvwrite(OUT/'arx_context_validation.csv',validation);dump(OUT/'arx_fits.json',fits)
    result=dict(status='completed',n_metric_rows=len(rows),predictions=files,n_latency_rows=len(latency),training_summary_sha256=sha(training),data_manifest_sha256=sha(DATA/'manifest.json'),script_sha256=sha(pathlib.Path(__file__)),plan_sha256=sha(ROOT/'MODEL_COMPARISON_PLAN.md'),amendment_sha256=sha(ROOT/'NORMALIZATION_AMENDMENT.md'),source_checks=source_checks,pretrained_weight_sha256=weight_sha,elapsed_s=time.monotonic()-start,test_worlds_per_family=128,plans=PLANS,horizons=HORIZONS,accepted_directions=0,scope='New synthetic two-family general-model comparison; not author leaderboard reproduction or real-world validation')
    dump(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)
if __name__=='__main__':main()
