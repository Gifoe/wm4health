from common90_v1 import *
import torch,numpy as np,time
import csdi_core89_v1 as core
from train_csdi89_v1 import validate

def main():
    seed=int(sys.argv[1]);assert seed in [2026901,2026902,2026903]
    assert not (O/f'FIT90_{seed}_V1.json').exists();started=time.perf_counter()
    manifest=read(ROOT/'cycle89/outputs/CSDI_DATA89_V1.json');paths={r['split']:r for r in manifest['files']}
    for split in ['train','cal']:assert sha(paths[split]['path'])==paths[split]['sha256']
    with np.load(paths['train']['path']) as f:train=torch.as_tensor(f['X'].copy(),device='cuda')
    with np.load(paths['cal']['path']) as f:cal=torch.as_tensor(f['X'][np.r_[0:64,128:192]].copy(),device='cuda')
    model,meta=core.new_model(seed);opt=torch.optim.Adam(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[15000,18000],gamma=.1)
    dest=Path(STORAGE['checkpoint_root'])/f'cycle90/native_seed{seed}_v1';dest.mkdir(parents=True,exist_ok=True);cp=dest/'selected.pt';assert not cp.exists()
    best=float('inf');trace=[]
    for step in range(20001):
        if step in [0,1000,3000,6000,10000,15000,20000]:
            value=validate(model,cal);assert np.isfinite(value)
            row=dict(step=step,cal_denoising_mse=value,elapsed_seconds=time.perf_counter()-started);trace.append(row);log(dict(stage='full_budget_cal',seed=seed,**row))
            if value<best:
                best=value;torch.save(dict(model=model.state_dict(),step=step,seed=seed,config=core.config(),source=meta,protocol_sha256=sha(C/'PROTOCOL90_V1.md')),cp)
        if step==20000:break
        model.train();ix=torch.randint(len(train),(64,),device='cuda');x=train[ix];cm,tp=core.mask_and_time(len(x),'cuda');side=model.get_side_info(tp,cm)
        opt.zero_grad(set_to_none=True);loss=model.calc_loss(x,cm,torch.ones_like(cm),side,1);assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step();scheduler.step()
        if (step+1)%500==0:log(dict(stage='full_budget_progress',seed=seed,step=step+1,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-started))
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    result=dict(utc=utc(),status='FULL_BUDGET_NATIVE_FIT_COMPLETE_CAL_SCREEN_PENDING',seed=seed,updates=20000,selected_step=saved['step'],cal_denoising_mse=best,trace=trace,checkpoint_path=str(cp),checkpoint_sha256=sha(cp),source=meta,source_files_unchanged=all(sha(Path(meta['root'])/p)==h for p,h in meta['sha256'].items()),protocol_sha256=sha(C/'PROTOCOL90_V1.md'),data_manifest_sha256=sha(ROOT/'cycle89/outputs/CSDI_DATA89_V1.json'),eval_accessed=False,parameters=sum(p.numel() for p in model.parameters()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started)
    write(O/f'FIT90_{seed}_V1.json',result);log(dict(stage='full_budget_fit_closed',seed=seed,selected_step=saved['step'],elapsed_seconds=result['elapsed_seconds']))

if __name__=='__main__':main()
