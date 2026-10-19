"""Matched-data cross-system baseline training; validation-only model selection."""
import runtime_profile
import copy,hashlib,json,math,pathlib,sys,time,random
import numpy as np
import torch
from world_models import build,scale_context,RecurrentWorld
from world_data import OUT as DATA,main as make_data,sha,dump,ROOT
OUT=ROOT/'outputs/world_training'
KINDS=['conditional96','compiled64','conditional128','native_attention64']
SEEDS=[2026291,2026292,2026293];MAX_STEPS=8192;BATCH=32;C=128;H=32

def setup():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
def hashes():return {p.name:sha(p) for p in [ROOT/'MODEL_COMPARISON_PLAN.md',ROOT/'scripts/world_data.py',ROOT/'scripts/world_models.py',pathlib.Path(__file__)]}
def read_data(split):
    blocks=[]
    for family in ['lti','wh']:
        with np.load(DATA/f'{split}_{family}.npz',allow_pickle=False) as f:blocks.append({k:f[k] for k in ['u','y']})
    return {k:torch.tensor(np.concatenate([b[k] for b in blocks]),device='cuda').unsqueeze(-1) for k in ['u','y']}
def batch(data,indices,starts=None):
    if starts is None:starts=torch.zeros(len(indices),device='cuda',dtype=torch.long)
    t=starts[:,None]+torch.arange(C+H,device='cuda')[None,:];i=indices[:,None]
    y=data['y'][i,t];u=data['u'][i,t]
    return scale_context(y[:,:C],u[:,:C],u[:,C:],y[:,C:])[:4]
@torch.no_grad()
def validate(model,data):
    model.eval();losses=[]
    for i in range(0,len(data['y']),BATCH):
        x=batch(data,torch.arange(i,min(i+BATCH,len(data['y'])),device='cuda'));pred=model(*x[:3]);losses.append(float((pred-x[3]).square().mean()))
    model.train();return float(np.mean(losses))

def critical_checks():
    torch.manual_seed(20262900);checks={}
    model=RecurrentWorld('compiled64').cuda();h=torch.randn(3,64,device='cuda');u=torch.randn(3,1,device='cuda');zero=torch.zeros(3,192,64,device='cuda')
    error=float((model.advance(h,u,zero)-model.cell(u,h)).abs().max());assert error<1e-6;checks['zero_update_grucell_max_abs']=error
    cy=torch.randn(3,C,1,device='cuda');cu=torch.randn_like(cy);fu=torch.randn(3,64,1,device='cuda')
    for kind in KINDS:
        model=build(kind).cuda().train();model.zero_grad(set_to_none=True);pred=model(cy,cu,fu[:,:32]);pred.square().mean().backward()
        grads={n:float(p.grad.abs().max()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None];assert not missing,(kind,missing)
        assert grads and all(np.isfinite(v) for v in grads.values())
        model.eval()
        with torch.no_grad():
            a=model(cy,cu,fu);alt=fu.clone();alt[:,32:]+=10;b=model(cy,cu,alt)
            prefix=float((a[:,:32]-b[:,:32]).abs().max());assert prefix<2e-5,(kind,prefix)
            if kind=='native_attention64':
                q_u=torch.cat((cu[:,-16:],fu),1);q_y=torch.cat((cy[:,-16:],torch.zeros_like(fu)),1)
                aa=model.core(cy,cu,q_u,q_y,16)[0];q_y[:,16:]=123.;bb=model.core(cy,cu,q_u,q_y,16)[0]
                native_error=float((a-aa).abs().max());placeholder=float((aa-bb).abs().max());assert native_error<2e-5 and placeholder==0
                checks['native_wrapper_mean_max_abs']=native_error;checks['native_target_placeholder_max_abs']=placeholder
        checks[kind]=dict(causal_prefix_max_abs=prefix,total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gradient_max=grads)
    dump(OUT/'critical_checks.json',checks);print(json.dumps(dict(phase='critical_checks_passed',counts={k:checks[k]['trainable_parameters'] for k in KINDS})),flush=True)

def train_one(kind,seed,train,val):
    folder=OUT/f'{kind}_{seed}'
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text());assert result['hashes']==hashes();print(json.dumps(dict(phase='already_complete',kind=kind,seed=seed)),flush=True);return result
    folder.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=build(kind).cuda();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed+1000);indices=rng.integers(0,len(train['y']),size=(MAX_STEPS,BATCH),dtype=np.int64);starts=rng.integers(0,512-C-H+1,size=(MAX_STEPS,BATCH),dtype=np.int64)
    # Share the complete batch schedule across model types for this seed.
    schedule_sha=hashlib.sha256(indices.tobytes()+starts.tobytes()).hexdigest()
    indices=torch.tensor(indices,device='cuda');starts=torch.tensor(starts,device='cuda')
    best=math.inf;best_step=0;stale=0;history=[];window_losses=[];begin=0;elapsed_before=0
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cuda',weights_only=True);assert state['hashes']==hashes();model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        begin=state['step'];best=state['best'];best_step=state['best_step'];stale=state['stale'];history=state['history'];elapsed_before=state['elapsed_s']
        # No dropout or stochastic forward; schedule and explicit step fully determine continuation.
        print(json.dumps(dict(phase='resume',kind=kind,seed=seed,step=begin)),flush=True)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();model.train()
    log=folder/'training.jsonl'
    with log.open('a',encoding='utf-8') as stream:
        for k in range(begin,MAX_STEPS):
            cy,cu,fu,target=batch(train,indices[k],starts[k]);opt.zero_grad(set_to_none=True);pred=model(cy,cu,fu);loss=(pred-target).square().mean()
            if not torch.isfinite(loss):raise ArithmeticError((kind,seed,k,'nonfinite loss'))
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise ArithmeticError((kind,seed,k,'nonfinite gradient'))
            opt.step();window_losses.append(float(loss.detach()))
            if (k+1)%512==0:
                vl=validate(model,val);torch.cuda.synchronize();elapsed=time.monotonic()-start+elapsed_before
                row=dict(step=k+1,training_mse=float(np.mean(window_losses)),validation_mse=vl,elapsed_s=elapsed,grad_norm=float(norm));window_losses=[]
                history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                if vl<best*.998:
                    best=vl;best_step=k+1;stale=0
                    torch.save({'model':model.state_dict(),'kind':kind,'seed':seed,'step':k+1,'validation_mse':vl,'hashes':hashes()},folder/'best.pt')
                else:stale+=1
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=k+1,best=best,best_step=best_step,stale=stale,history=history,elapsed_s=elapsed,hashes=hashes()),folder/'last.pt')
                dump(folder/'status.json',dict(status='training',kind=kind,seed=seed,best_step=best_step,best_validation_mse=best,**row))
                print(json.dumps(dict(phase='validation',kind=kind,seed=seed,best_step=best_step,stale=stale,**row)),flush=True)
                if k+1>=2048 and stale>=4:break
    torch.cuda.synchronize();elapsed=time.monotonic()-start+elapsed_before
    result=dict(status='completed',kind=kind,seed=seed,steps=k+1,best_step=best_step,best_validation_mse=best,stopped_by_patience=stale>=4,underconverged_at_cap=(k+1==MAX_STEPS and len(history)>=5 and history[-1]['validation_mse']<history[-5]['validation_mse']*.95),total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),batch_schedule_sha256=schedule_sha,hashes=hashes(),best_weight_sha256=sha(folder/'best.pt'),history=history,elapsed_s=elapsed,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(folder/'result.json',result);dump(folder/'status.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['history','hashes']}),flush=True)
    del opt,model;torch.cuda.empty_cache();return result

def main():
    setup();OUT.mkdir(parents=True,exist_ok=True)
    if not (DATA/'manifest.json').exists():make_data()
    data_manifest=json.loads((DATA/'manifest.json').read_text())
    assert data_manifest['plan_sha256']==sha(ROOT/'MODEL_COMPARISON_PLAN.md')
    for row in data_manifest['files']:assert sha(DATA/row['path'])==row['sha256']
    critical_checks();train=read_data('train');val=read_data('val');rows=[]
    # First complete a whole model comparison for seed1, then independent repeats.
    for seed in SEEDS:
        for kind in KINDS:rows.append(train_one(kind,seed,train,val))
    summary=dict(status='all_training_complete',runs=len(rows),rows=[{k:v for k,v in row.items() if k!='history'} for row in rows],data_manifest_sha256=sha(DATA/'manifest.json'),hashes=hashes(),test_not_loaded=True,environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name()))
    dump(OUT/'summary.json',summary);print(json.dumps(dict(phase='all_training_complete',runs=len(rows))),flush=True)
if __name__=='__main__':main()
