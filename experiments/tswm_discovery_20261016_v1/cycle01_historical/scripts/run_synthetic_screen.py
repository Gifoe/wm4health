import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,copy,hashlib,json,pathlib,random,subprocess,sys,time
import numpy as np
import torch
from torch.nn import functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from synthetic_data import split,pulse_truth,simulate
from synthetic_models import make_model,stage1_loss
torch.set_num_threads(4)
if os.name=='nt' and os.cpu_count()>=32:
    import ctypes
    fn=ctypes.windll.kernel32.SetProcessAffinityMask
    fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];fn.restype=ctypes.c_int
    assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
def dump(x,p):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');tmp.replace(p)
def save(x,p):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def config():return json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
def batch_arrays(data):return tuple(torch.as_tensor(data[k],device='cuda') for k in ['history','future','target'])
def predict(model,h,u):
    out=[]
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(h),128):out.append(model(h[i:i+128],u[i:i+128]).cpu().numpy())
    return np.concatenate(out)
def fit(name,seed,excitation):
    cfg=config();key=f'{name}_s{seed}_{excitation}';folder=ROOT/'outputs/runs'/key
    folder.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=make_model(name,cfg).cuda();tr=split('train',cfg,excitation);va=split('validation',cfg,excitation)
    h,u,y=batch_arrays(tr);vh,vu,vy=batch_arrays(va)
    history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    stages=[1,2] if name=='worldts_two_stage' else [0]
    for stage in stages:
        if stage==1:model.core.configure_for_stage1()
        elif stage==2:model.core.configure_for_stage2()
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg['learning_rate'])
        best=float('inf');best_state=None;bad=0
        for epoch in range(cfg['max_epochs']):
            model.train()
            if stage==2:model.core.set_stage1_eval()
            order=torch.randperm(len(h),device='cuda');total=0.
            for idx in order.split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True)
                if stage==1:
                    z=model.core(h[idx],future_target=y[idx],x_future=u[idx]);loss=stage1_loss(z)
                elif stage==2:
                    with torch.no_grad():z=model.core(h[idx],future_target=y[idx],x_future=u[idx])
                    yp=model.core.decode(z['h_pred'],history_tokens=z['h_hist'])
                    yr=model.core.decode(z['h_target'],history_tokens=z['h_hist'])
                    loss=F.l1_loss(yp,y[idx])+.3*F.l1_loss(yr,y[idx])
                else:loss=F.mse_loss(model(h[idx],u[idx]),y[idx])
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                total+=float(loss.detach())*len(idx)
            model.eval()
            with torch.inference_mode():
                if stage==1:vl=float(stage1_loss(model.core(vh,future_target=vy,x_future=vu)))
                else:vl=float(F.mse_loss(model(vh,vu),vy))
            row={'stage':stage,'epoch':epoch+1,'train_loss':total/len(h),'validation_loss':vl,'elapsed_s':time.perf_counter()-start}
            history.append(row)
            if vl<best-1e-8:
                best=vl;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                save({'state':best_state,'config':cfg,'model':name,'seed':seed,'stage':stage,'validation_loss':best,'history':history},folder/f'stage{stage}_best.pt')
            else:bad+=1
            if epoch%10==0:print(key,json.dumps(row),flush=True)
            dump(history,folder/'training_curve.json')
            if bad>=cfg['patience']:break
        assert best_state is not None;model.load_state_dict(best_state,strict=True)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start
    data=split('discovery',cfg,excitation);dh,du,dy=batch_arrays(data)
    pred=predict(model,dh,du);base_error=((pred-data['target'])**2).mean((1,2))
    # A whole-plan positive shift checks sign, not timing; it is explicitly a
    # newly designed diagnostic, not the paper's quantile-shift MC score.
    shifted=predict(model,dh,du+.5)-pred
    result={'model':name,'seed':seed,'excitation':excitation,'scope':'discovery; no confirmation accessed',
       'factual_mse':float(base_error.mean()),'whole_plan_shift_positive_fraction':float((shifted.mean((1,2))>1e-6).mean()),
       'parameters':sum(p.numel() for p in model.parameters()),'train_seconds':train_seconds,
       'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),
       'best_validation_loss':best,'pulses':[]}
    arrays={'factual_errors':base_error}
    for pos in [0,12,24]:
        v=du.clone();v[:,pos]+=.5;delta=predict(model,dh,v)-pred
        truth=pulse_truth(data,cfg,pos,.5)
        late=np.arange(cfg['horizon'])>=pos+5
        rec={'position':pos,'effect_nmse':float(np.sum((delta-truth)**2)/np.sum(truth**2)),
             'late_tail_gain':float(np.sum(delta[:,late]*truth[:,late])/np.sum(truth[:,late]**2)),
             'pre_action_max_change':float(np.abs(delta[:,:pos]).max()) if pos else 0,
             'outside_native_support_max':float(np.abs(delta[:,pos+5:]).max())}
        result['pulses'].append(rec);arrays[f'effect_{pos}']=delta;arrays[f'truth_{pos}']=truth
    np.savez_compressed(folder/'discovery_predictions.npz',**arrays)
    dump(result,folder/'result.json');print('COMPLETED',json.dumps(result),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--seed',type=int);p.add_argument('--excitation',default='smooth');a=p.parse_args()
    if a.model:fit(a.model,a.seed,a.excitation);return
    cfg=config();root=ROOT/'outputs';(root/'logs').mkdir(parents=True,exist_ok=True)
    statuses=[]
    for name in cfg['initial_models']:
      for seed in cfg['train_seeds']:
        key=f'{name}_s{seed}_smooth';done=root/'runs'/key/'result.json'
        if done.exists():statuses.append({'run':key,'status':'reused_complete'});continue
        with (root/'logs'/f'{key}.log').open('w',encoding='utf-8') as log:
            ret=subprocess.run([sys.executable,'-u',str(pathlib.Path(__file__)),'--model',name,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        statuses.append({'run':key,'exit_code':ret.returncode,'completed':done.exists()})
        dump(statuses,root/'screen_status.json')
        if not done.exists():raise RuntimeError(f'{key}: exit {ret.returncode}; inspect its log before resuming')
    dump(statuses,root/'screen_status.json');print('SCREEN_COMPLETE',flush=True)
if __name__=='__main__':main()
