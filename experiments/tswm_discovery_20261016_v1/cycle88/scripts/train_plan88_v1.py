from common88_v1 import *
import argparse,time,numpy as np,torch
from plan_models88_v1 import make_model,qualify_sources,tensor_state_sha

torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE='cuda';Q=65;T=96;SEED=2026884

def load_split(name):
    manifest=read(O/'PLAN_DATA88_V1.json');row=next(r for r in manifest['files'] if r['split']==name)
    assert sha(row['path'])==row['sha256']
    with np.load(row['path']) as z:return {k:z[k] for k in ['observations','actions','family']}

def batch(data,indices,starts):
    x=np.stack([data['observations'][i,s:s+T] for i,s in zip(indices,starts)])
    y=np.stack([data['observations'][i,s+1:s+T+1] for i,s in zip(indices,starts)])
    a=np.stack([data['actions'][i,s:s+T] for i,s in zip(indices,starts)])
    x[:,Q:]=0;mask=np.ones((len(indices),T,1),bool);mask[:,Q:]=False
    return [torch.as_tensor(v,device=DEVICE) for v in [x,a,mask,y]]

@torch.no_grad()
def validate(model,data):
    model.eval();errors=[];persist=[];losses=[]
    for start in range(0,len(data['actions']),16):
        ids=np.arange(start,min(start+16,len(data['actions'])));x,a,m,y=batch(data,ids,np.full(len(ids),48));mean,var=model(x,a,m)[:2]
        y=y[:,Q-1:];mean=mean[:,Q-1:];var=var[:,Q-1:]
        assert torch.isfinite(mean).all() and torch.isfinite(var).all() and (var>0).all()
        errors.extend(((mean-y)**2).mean((1,2)).cpu().tolist());persist.extend(((x[:,Q-1:Q]-y)**2).mean((1,2)).cpu().tolist())
        losses.extend((.5*(torch.log(2*torch.pi*var)+(mean-y)**2/var)).mean((1,2)).cpu().tolist())
    e=np.array(errors);p=np.array(persist);families=[]
    for f in [0,1]:
        mask=data['family']==f;families.append(dict(family=f,mse=float(e[mask].mean()),persistence_mse=float(p[mask].mean()),persistence_ratio=float(e[mask].mean()/p[mask].mean())))
    return dict(mse=float(e.mean()),nll=float(np.mean(losses)),families=families)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['stock','lagged','acrkn'],required=True);parser.add_argument('--benchmark-only',action='store_true');args=parser.parse_args()
    qualify_sources();train=load_split('train');val=load_split('val');model=make_model(args.arm,SEED);initial=tensor_state_sha(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(SEED);rows=[];started=time.perf_counter();best=float('inf');B=16
    dest=Path(STORAGE['checkpoint_root'])/'cycle88'/args.arm;dest.mkdir(parents=True,exist_ok=True)
    receipt=O/f'PLAN_FIT88_V1_{args.arm}.json'
    if not args.benchmark_only:assert not receipt.exists() and not (dest/'selected88_v1.pt').exists(),'Existing fit; do not overwrite or duplicate'
    steps=3 if args.benchmark_only else 1000
    params=sum(p.numel() for p in model.parameters())
    for step in range(steps+1):
        if step in [0,250,500,750,1000] and not args.benchmark_only:
            row=dict(step=step,**validate(model,val),elapsed_seconds=time.perf_counter()-started);rows.append(row)
            if row['mse']<best:
                best=row['mse'];torch.save(dict(model=model.state_dict(),step=step,arm=args.arm,initial_sha256=initial,seed=SEED),dest/'selected88_v1.pt')
            log(dict(stage='plan_validation',arm=args.arm,parameters=params,initial_sha256=initial,**row))
        if step==steps:break
        ids=rng.integers(0,len(train['actions']),B);starts=rng.integers(0,97,B);x,a,m,y=batch(train,ids,starts)
        model.train();optimizer.zero_grad(set_to_none=True);mean,var=model(x,a,m)[:2];mean=mean[:,Q-1:];var=var[:,Q-1:];y=y[:,Q-1:]
        loss=.5*(torch.log(2*torch.pi*var)+(mean-y)**2/var).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
        if (step+1)%50==0:log(dict(stage='plan_fit_progress',arm=args.arm,step=step+1,nll=float(loss.detach()),grad_norm=float(norm),elapsed_seconds=time.perf_counter()-started))
    torch.cuda.synchronize()
    if args.benchmark_only:
        print(json.dumps(dict(arm=args.arm,steps=steps,seconds=time.perf_counter()-started,parameters=params,initial_sha256=initial,peak_gpu_bytes=torch.cuda.max_memory_allocated())),flush=True);return
    final=dest/'final88_v1.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=steps,arm=args.arm,seed=SEED),final)
    selected=dest/'selected88_v1.pt';chosen=min(rows,key=lambda r:r['mse']);qualify_sources()
    result=dict(utc=utc(),status='DEVELOPMENTAL_FIT_COMPLETE',arm=args.arm,seed=SEED,updates=steps,parameters=params,initial_sha256=initial,validation_trace=rows,selected=chosen,source_competent=all(r['persistence_ratio']<.9 for r in chosen['families']),selected_path=str(selected),selected_sha256=sha(selected),final_path=str(final),final_sha256=sha(final),elapsed_seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),data_manifest_sha256=sha(O/'PLAN_DATA88_V1.json'),protocol_sha256=sha(C/'PLAN_CAUSALITY_PROTOCOL88_V1.md'),matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,edited_plan_metrics_accessed_during_selection=False,held_out_eval_accessed=False)
    write(receipt,result);log(dict(stage='plan_fit_closed',arm=args.arm,source_competent=result['source_competent'],selected=chosen,elapsed_seconds=result['elapsed_seconds']))

if __name__=='__main__':main()
