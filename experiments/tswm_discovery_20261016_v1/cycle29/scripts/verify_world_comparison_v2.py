"""All-metric rescore, checkpoint selection, and sampled neural prediction replay."""
import runtime_profile
import csv,hashlib,json,pathlib,time
import numpy as np
import torch
from world_data import ROOT,OUT as DATA,sha,dump
from world_models_v2 import build
from train_world_comparison_v2 import setup,hashes,KINDS,SEEDS,MAX_STEPS,BATCH,C,H as TRAIN_H
EVAL=ROOT/'outputs/world_evaluation_v2';OUT=ROOT/'outputs/world_verification';H=128
PLANS=['factual','zero','step','pulse']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def parse_name(path):
    stem=pathlib.Path(path).stem;family,tail=stem.split('_',1);seed=None
    for value in SEEDS:
        if tail.endswith('_'+str(value)):tail=tail[:-(len(str(value))+1)];seed=value;break
    return family,tail,seed
def rowkey(family,method,seed,system,plan,horizon):return (family,method,str(seed) if seed is not None else '',system,plan,horizon)
def ridge_lstsq(y,u,order,lam):
    xs=[];targets=[]
    for t in range(order,len(y)):
        xs.append([y[t-j] for j in range(1,order+1)]+[u[t-j] for j in range(order)]);targets.append(y[t])
    x=np.asarray(xs);mean=x.mean(0);scale=np.maximum(x.std(0),1e-8);z=np.c_[np.ones(len(x)),(x-mean)/scale]
    penalty=np.sqrt(lam)*np.eye(z.shape[1]);penalty[0,0]=0
    beta=np.linalg.lstsq(np.r_[z/np.sqrt(len(z)),penalty],np.r_[np.asarray(targets)/np.sqrt(len(z)),np.zeros(z.shape[1])],rcond=None)[0]
    return dict(order=order,mean=mean,scale=scale,beta=beta)
def free_arx(fit,observed,u,horizon):
    history=list(observed);start=len(history);out=[];order=fit['order']
    for t in range(start,start+horizon):
        features=np.asarray([history[t-j] for j in range(1,order+1)]+[u[t-j] for j in range(order)])
        value=fit['beta'][0]+((features-fit['mean'])/fit['scale'])@fit['beta'][1:];history.append(value);out.append(value)
    return np.asarray(out)
@torch.no_grad()
def replay(model,data,method,ids):
    result=[]
    for i in ids:
        cy=torch.tensor(data['y'][i,0,:C],device='cuda')[None,:,None].repeat(4,1,1);cu=torch.tensor(data['u'][i,0,:C],device='cuda')[None,:,None].repeat(4,1,1)
        fu=torch.tensor(data['u'][i,:,C:],device='cuda')[:,:,None]
        if method.endswith('_chunk32'):
            parts=[]
            for lo in range(0,H,32):
                u=fu[:,lo:lo+32];p=model(cy,cu,u);parts.append(p);cy=torch.cat((cy,p),1)[:,-C:];cu=torch.cat((cu,u),1)[:,-C:]
            value=torch.cat(parts,1)
        elif '_swap_' in method or method.endswith('_zero_operator'):
            h,code=model.encode(cy,cu)
            if method.endswith('_zero_operator'):code=torch.zeros_like(code)
            else:
                donor=(i+1)%len(data['y']);dy=torch.tensor(data['y'][donor,0,:C],device='cuda')[None,:,None].repeat(4,1,1);du=torch.tensor(data['u'][donor,0,:C],device='cuda')[None,:,None].repeat(4,1,1)
                _,code=model.encode(dy,du)
            value=model.simulate((h,code),fu)
        else:value=model(cy,cu,fu)
        result.append(value[:,:,0].cpu().numpy())
    return np.array(result)
def main():
    setup();start=time.monotonic();result=read(EVAL/'results.json');assert result['status']=='completed'
    data_check=read(OUT/'data_replay.json');assert data_check['status']=='passed' and data_check['data_manifest_sha256']==sha(DATA/'manifest.json')
    trainroot=ROOT/'outputs/world_training_v2';training=read(trainroot/'summary.json');assert training['hashes']==hashes() and training['runs']==12
    schedules={};checkpoints=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed+1000);idx=rng.integers(0,1024,size=(MAX_STEPS,BATCH),dtype=np.int64);starts=rng.integers(0,512-C-TRAIN_H+1,size=(MAX_STEPS,BATCH),dtype=np.int64)
        schedules[seed]=hashlib.sha256(idx.tobytes()+starts.tobytes()).hexdigest()
        for kind in KINDS:
            folder=trainroot/f'{kind}_{seed}';r=read(folder/'result.json');assert r['hashes']==hashes() and r['batch_schedule_sha256']==schedules[seed]
            assert r['best_validation_mse']==min(x['validation_mse'] for x in r['history'])
            assert r['best_weight_sha256']==sha(folder/'best.pt')
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert state['step']==r['best_step'] and state['validation_mse']==r['best_validation_mse']
            checkpoints.append(dict(kind=kind,seed=seed,steps=r['steps'],best_step=r['best_step'],best_validation_mse=r['best_validation_mse'],weight_sha256=r['best_weight_sha256'],underconverged_at_cap=r['underconverged_at_cap']))
    data={}
    for family in ['lti','wh']:
        with np.load(DATA/f'test_{family}.npz',allow_pickle=False) as z:data[family]={k:z[k] for k in ['u','y','clean']}
    with (EVAL/'per_system_metrics.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    by={rowkey(r['family'],r['method'],int(r['seed']) if r['seed'] else None,int(r['system']),r['plan'],int(r['horizon'])):r for r in rows};assert len(by)==len(rows)
    metric_max=0.;n=0;predictions={}
    for f in result['predictions']:
        path=EVAL/f['path'];assert sha(path)==f['sha256'];family,method,seed=parse_name(f['path'])
        with np.load(path,allow_pickle=False) as z:pred=z['prediction'].astype('float64')
        assert pred.shape==(128,4,128);predictions[(family,method,seed)]=pred
        truth=data[family]['clean'][:,:,128:].astype('float64');noisy=data[family]['y'][:,:,128:].astype('float64')
        for i in range(128):
            for j,plan in enumerate(PLANS):
                for horizon in [32,64,128]:
                    key=rowkey(family,method,seed,i,plan,horizon);row=by[key];n+=1
                    p=pred[i,j,:horizon];t=truth[i,j,:horizon];effect=p-pred[i,0,:horizon];target_effect=t-truth[i,0,:horizon]
                    values=dict(clean_rmse=np.linalg.norm(p-t)/np.sqrt(horizon),noisy_rmse=np.linalg.norm(p-noisy[i,j,:horizon])/np.sqrt(horizon),effect_rmse=np.linalg.norm(effect-target_effect)/np.sqrt(horizon))
                    den=target_effect@target_effect
                    if den>1e-8:values['effect_slope']=float(target_effect@effect/den)
                    assert (row['informative_effect']=='True')==(den>1e-8)
                    for k,v in values.items():
                        old=float(row[k]);err=abs(old-v)/(1+abs(v));assert err<2e-5,(key,k,old,v);metric_max=max(metric_max,float(err))
    assert n==len(rows)==result['n_metric_rows']
    with (EVAL/'arx_context_validation.csv').open(newline='',encoding='utf-8') as stream:arxrows=list(csv.DictReader(stream))
    arxvalidation={(r['family'],int(r['system']),int(r['order']),float(r['lam'])):float(r['context_validation_mse']) for r in arxrows}
    fits=read(EVAL/'arx_fits.json');arx_error=0.;arx_fits_recomputed=0
    for fit in fits:
        family=fit['family'];i=fit['system'];observed=data[family]['y'][i,0,:128].astype('float64');u=data[family]['u'][i,0,:128].astype('float64');candidates=[]
        for order in [1,2,4,8,16]:
            for lam in [1e-4,.01,1.]:
                f=ridge_lstsq(observed[:96],u[:96],order,lam);pred=free_arx(f,observed[:96],u,32);mse=float(np.mean((pred-observed[96:])**2));old=arxvalidation[(family,i,order,lam)]
                err=abs(old-mse)/(1+abs(mse));assert err<2e-6;arx_error=max(arx_error,err);arx_fits_recomputed+=1;candidates.append(mse)
        selected=arxvalidation[(family,i,fit['order'],fit['lam'])];assert selected<=min(candidates)+2e-6*(1+abs(min(candidates)))
        f=ridge_lstsq(observed,u,fit['order'],fit['lam']);arx_fits_recomputed+=1
        for plan in range(4):
            pred=free_arx(f,observed,data[family]['u'][i,plan],128);expected=predictions[(family,'arx_context_selected',None)][i,plan];err=float(np.max(np.abs(pred-expected)/(1+np.abs(expected))));assert err<2e-5;arx_error=max(arx_error,err)
    neural=[];ids=[0,31,127]
    for seed in SEEDS:
        for kind in KINDS:
            model=build(kind).cuda().eval();state=torch.load(trainroot/f'{kind}_{seed}'/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True)
            methods=[kind]
            if kind=='native_attention64':methods.append(kind+'_chunk32')
            if kind=='compiled64':methods+=['compiled64_zero_operator','compiled64_swap_operator']
            if kind=='conditional128':methods+=['conditional128_swap_code']
            for family in ['lti','wh']:
                for method in methods:
                    p=replay(model,data[family],method,ids);expected=predictions[(family,method,seed)][ids];err=float(np.max(np.abs(p-expected)/(1+np.abs(expected))));assert err<2e-4,(seed,kind,family,method,err)
                    neural.append(dict(seed=seed,method=method,family=family,worlds=ids,relative_max_error=err))
            del model;torch.cuda.empty_cache()
    # Nonzero generated recurrent update must equal a native GRUCell with the modified matrix.
    model=build('compiled64').cuda().eval();torch.manual_seed(20262999);h=torch.randn(3,64,device='cuda');u=torch.randn(3,1,device='cuda');delta=torch.randn(3,192,64,device='cuda')*.1;expected=[]
    with torch.no_grad():
        actual=model.advance(h,u,delta)
        for i in range(3):
            cell=torch.nn.GRUCell(1,64).cuda();cell.load_state_dict(model.cell.state_dict());cell.weight_hh.add_(delta[i]);expected.append(cell(u[i:i+1],h[i:i+1]))
        operator_error=float((actual-torch.cat(expected)).abs().max());assert operator_error<2e-6
    verification=dict(status='passed',results_sha256=sha(EVAL/'results.json'),training_summary_sha256=sha(trainroot/'summary.json'),data_replay_sha256=sha(OUT/'data_replay.json'),script_sha256=sha(pathlib.Path(__file__)),metric_rows_rescored=n,relative_metric_error_max=metric_max,arx_fits_recomputed=arx_fits_recomputed,arx_relative_error_max=arx_error,checkpoints=checkpoints,batch_schedules=schedules,neural_replays=neural,neural_relative_error_max=max(r['relative_max_error'] for r in neural),nonzero_operator_native_grucell_max_abs=operator_error,elapsed_s=time.monotonic()-start,scope='all metric rows + validation selection + independent schedule + ARX refits + sampled trained-neural replay; physics separately replayed by SciPy')
    dump(OUT/'model_comparison_v2.json',verification);print(json.dumps({k:v for k,v in verification.items() if k not in ['neural_replays','checkpoints']}),flush=True)
if __name__=='__main__':main()
