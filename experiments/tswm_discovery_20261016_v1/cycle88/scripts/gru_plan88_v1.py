from common88_v1 import *
import numpy as np,torch,time
from train_plan88_v1 import load_split,batch,validate
from plan_models88_v1 import tensor_state_sha

class OrdinaryGRU(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gru=torch.nn.GRU(19,192,batch_first=True);self.mean=torch.nn.Linear(192,16);self.variance=torch.nn.Linear(192,16)
    def forward(self,x,a,mask):
        hidden,_=self.gru(torch.cat([x,a,mask.float()],dim=-1))
        return self.mean(hidden),torch.nn.functional.softplus(self.variance(hidden))+1e-4

def main():
    torch.manual_seed(2026884);model=OrdinaryGRU().cuda();initial=tensor_state_sha(model);optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(2026884)
    train=load_split('train');val=load_split('val');start=time.perf_counter();rows=[];best=float('inf');dest=Path(STORAGE['checkpoint_root'])/'cycle88/gru';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'selected88_v1.pt').exists()
    for step in range(1001):
        if step in [0,250,500,750,1000]:
            row=dict(step=step,**validate(model,val));rows.append(row)
            if row['mse']<best:
                best=row['mse'];torch.save(dict(model=model.state_dict(),step=step,arm='gru'),dest/'selected88_v1.pt')
            log(dict(stage='ordinary_gru_validation',elapsed_seconds=time.perf_counter()-start,**row))
        if step==1000:break
        ids=rng.integers(0,len(train['actions']),16);starts=rng.integers(0,97,16);x,a,m,y=batch(train,ids,starts)
        model.train();optimizer.zero_grad(set_to_none=True);mean,var=model(x,a,m);mean=mean[:,64:];var=var[:,64:];y=y[:,64:]
        loss=.5*(torch.log(2*torch.pi*var)+(mean-y)**2/var).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
    final=dest/'final88_v1.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=1000),final)
    selected=dest/'selected88_v1.pt';fit=dict(utc=utc(),arm='gru',status='ORDINARY_GRU_FIT_COMPLETE',parameters=sum(p.numel() for p in model.parameters()),seed=2026884,updates=1000,initial_sha256=initial,selected=min(rows,key=lambda r:r['mse']),validation_trace=rows,selected_path=str(selected),selected_sha256=sha(selected),final_path=str(final),final_sha256=sha(final),elapsed_seconds=time.perf_counter()-start,edited_plan_metrics_accessed=False)
    write(O/'GRU_FIT88_V1.json',fit);log(dict(stage='ordinary_gru_fit_closed',parameters=fit['parameters'],selected=fit['selected'],elapsed_seconds=fit['elapsed_seconds']))

if __name__=='__main__':main()
