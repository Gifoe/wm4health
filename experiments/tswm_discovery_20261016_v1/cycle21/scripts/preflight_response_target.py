import runtime_profile
import copy,json,pathlib
import numpy as np
import torch
from torch import nn
import response_target_prototype as r
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
    generator=torch.Generator(device='cuda').manual_seed(20262101)
    x=torch.randn(256,12,generator=generator,device='cuda');x[:,2]=torch.abs(x[:,2])+.4
    class SyntheticTeacher:
        def predict(self,state,action):
            result=state.clone();result[:,0]+=state[:,2]*action[:,0];result[:,1]+=action[:,1];return result
    _,_,meta=r.targets(SyntheticTeacher(),dict(x=x),dict(x=x),2021)
    projection=np.asarray(meta['projection']);expected=np.diag([1.,1.]+[0.]*10)
    error=float(np.max(np.abs(projection-expected)));assert error<1e-6 and meta['selected_rank']==2
    data=r.generate('queueing',20262102,31);q=data['current'][:,0];p=data['current'][:,1];g=data['current'][:,2];u,v=data['action'].T
    truth=np.column_stack([np.maximum(0,.8*q+.7*g-(.8+.4*u)),np.maximum(0,.75*p+.2*q+.45*g-(.5+.3*v))])
    eq=float(np.max(np.abs(truth-data['future'][:,:2])));assert eq<1e-6 and np.isfinite(data['future']).all()
    y=torch.randn(256,12,generator=generator,device='cuda');u=torch.sign(torch.randn(256,2,generator=generator,device='cuda'))
    replay={}
    for method in r.METHODS:
        states=[]
        for _ in range(2):
            torch.manual_seed(2021);model=r.make(method);ema=copy.deepcopy(model.encoder).eval() if isinstance(model,r.Mop) else None
            if ema is not None:
                for p in ema.parameters():p.requires_grad_(False)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            for _ in range(3):
                loss,_=r.base_loss(model,x,y,u,ema);assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step()
            states.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        e=float((states[0]-states[1]).abs().max());assert e==0;replay[method]=e
    path=ROOT/'outputs/preflight.json';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    info=dict(pass_=True,action_span_projection_error=error,selected_rank=meta['selected_rank'],queue_equation_error=eq,
        deterministic_three_update_replays=replay,script_sha256=r.sha(ROOT/'scripts/response_target_prototype.py'),plan_sha256=r.sha(ROOT/'PROTOTYPE_PLAN.md'),torch=torch.__version__)
    info['pass']=info.pop('pass_');path.write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
