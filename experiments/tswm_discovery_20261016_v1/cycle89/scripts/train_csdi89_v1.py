from csdi_core89_v1 import *
import time

@torch.no_grad()
def validate(model,x):
    model.eval();values=[]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(20268915)
        for t in range(50):
            for batch in x.split(64):
                cm,tp=mask_and_time(len(batch),'cuda');side=model.get_side_info(tp,cm)
                loss=model.calc_loss(batch,cm,torch.ones_like(cm),side,0,set_t=t)
                values.append(float(loss))
    return float(np.mean(values))

def main():
    started=time.perf_counter();assert not (O/'CSDI_PILOT89_V1.json').exists()
    qualification=read(O/'CSDI_QUALIFICATION89_V1.json');assert qualification['source_files_unchanged']
    data=read(O/'CSDI_DATA89_V1.json');paths={r['split']:r for r in data['files']}
    for split in ['train','cal']:assert sha(paths[split]['path'])==paths[split]['sha256']
    with np.load(paths['train']['path']) as f:train=torch.as_tensor(f['X'].copy(),device='cuda')
    with np.load(paths['cal']['path']) as f:cal=torch.as_tensor(f['X'][np.r_[0:64,128:192]].copy(),device='cuda')
    model,meta=new_model(20268913);opt=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[2250,2700],gamma=.1)
    dest=Path(STORAGE['checkpoint_root'])/'cycle89/csdi_v1';dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'selected.pt';assert not cp.exists();best=float('inf');trace=[]
    for step in range(3001):
        if step in [0,500,1000,2000,3000]:
            value=validate(model,cal);assert np.isfinite(value)
            row=dict(step=step,cal_denoising_mse=value,elapsed_seconds=time.perf_counter()-started)
            trace.append(row);log(dict(stage='csdi_cal_validation',**row))
            if value<best:
                best=value;torch.save(dict(model=model.state_dict(),step=step,config=config(),optimizer_seed=20268913,source=meta,protocol_sha256=sha(C/'CSDI_COUPLING_PROTOCOL89_V1.md')),cp)
        if step==3000:break
        model.train();ix=torch.randint(len(train),(64,),device='cuda');x=train[ix];cm,tp=mask_and_time(len(x),'cuda');side=model.get_side_info(tp,cm)
        opt.zero_grad(set_to_none=True);loss=model.calc_loss(x,cm,torch.ones_like(cm),side,1);assert torch.isfinite(loss)
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step();scheduler.step()
        if (step+1)%100==0:log(dict(stage='csdi_train_progress',step=step+1,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-started))
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    receipt=dict(utc=utc(),status='PILOT_TRAINING_COMPLETE_GENERATOR_COMPETENCE_UNASSESSED',updates=3000,optimizer_seed=20268913,selected_step=saved['step'],cal_denoising_mse=best,trace=trace,checkpoint_path=str(cp),checkpoint_sha256=sha(cp),source=meta,source_files_unchanged=all(sha(Path(meta['root'])/p)==h for p,h in meta['sha256'].items()),data_manifest_sha256=sha(O/'CSDI_DATA89_V1.json'),protocol_sha256=sha(C/'CSDI_COUPLING_PROTOCOL89_V1.md'),parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),eval_queries_accessed=False,accepted_research_directions=0)
    write(O/'CSDI_PILOT89_V1.json',receipt);log(dict(stage='csdi_pilot_train_closed',step=saved['step'],elapsed_seconds=receipt['elapsed_seconds']))

if __name__=='__main__':main()
