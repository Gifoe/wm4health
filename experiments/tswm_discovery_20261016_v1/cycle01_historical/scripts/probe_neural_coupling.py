import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import hashlib,json,pathlib,random,sys,time
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
if os.name=='nt' and os.cpu_count()>=32:
    import ctypes
    fn=ctypes.windll.kernel32.SetProcessAffinityMask
    fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];fn.restype=ctypes.c_int
    assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
def truth(x,a):
    A=x.new_tensor([[.86,.045],[-.03,.78]])
    D=x.new_tensor([[-.02,.02],[.01,.015]])
    b=x.new_tensor([.025,-.012])
    return x@A.T+(x@D.T)*a+b*a+.055*torch.tanh(1.4*x.flip(-1))
def rotate(e,a,theta):
    c=torch.cos(a*theta);s=torch.sin(a*theta)
    return torch.cat([c*e[...,:1]-s*e[...,1:],s*e[...,:1]+c*e[...,1:]],-1)
class GaussianSSM(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2))
    def forward(self,x,a):return self.net(torch.cat([x,a],-1))
def gen(n,seed,theta=.7):
    g=torch.Generator(device='cuda').manual_seed(seed)
    x=torch.randn((n,2),generator=g,device='cuda')*.6
    a=torch.rand((n,1),generator=g,device='cuda')*2-1
    e=torch.randn((n,2),generator=g,device='cuda')
    mu=truth(x,a);return x,a,mu+.1*rotate(e,a,theta),mu
def factual(n,H,seed):
    g=torch.Generator(device='cuda').manual_seed(seed)
    x0=torch.randn((n,2),generator=g,device='cuda')*.3
    eps=torch.randn((n,H,2),generator=g,device='cuda')
    states=[x0];a=torch.zeros((n,1),device='cuda')
    for t in range(H):states.append(truth(states[-1],a)+.1*eps[:,t])
    return torch.stack(states,1),eps
def roll(fn,x0,eps,angle):
    a=torch.ones((len(x0),1),device=x0.device);x=x0
    for t in range(eps.shape[1]):x=fn(x,a)+.1*rotate(eps[:,t],a,angle)
    return x[:,0]
def run():
    p=ROOT/'outputs/neural_coupling_plan.json';plan=json.loads(p.read_text());out=ROOT/'outputs/neural_coupling';out.mkdir(exist_ok=True)
    tx,ta,ty,tm=gen(plan['train_transitions'],98123)
    vx,va,vy,vm=gen(plan['validation_transitions'],98124)
    # Independent development/confirmation cohorts. Only DEV chooses the global angle.
    fd,ed=factual(plan['dev_episodes'],plan['horizon'],98125)
    fc,ec=factual(plan['confirmation_episodes'],plan['horizon'],98126)
    records=[];all_estimates=[];start=time.perf_counter()
    for seed in plan['training_seeds']:
        folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=GaussianSSM().cuda();opt=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
        best=float('inf');history=[];state=None
        for step in range(plan['gradient_steps']):
            idx=torch.randint(len(tx),(plan['batch_size'],),device='cuda')
            loss=((model(tx[idx],ta[idx])-ty[idx])**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if (step+1)%50==0:
                with torch.no_grad():v=float(((model(vx,va)-vy)**2).mean())
                history.append({'step':step+1,'validation_mse':v,'training_batch_mse':float(loss.detach())})
                if v<best:best=v;state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
        model.load_state_dict(state);model.eval()
        torch.save({'state':state,'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'validation_mse':best},folder/'best.pt')
        with torch.inference_mode():
            mean_rmse=float(torch.sqrt(((model(vx,va)-vm)**2).mean()))
            def abduct(f):
                x=f[:,:-1].reshape(-1,2);a=torch.zeros((len(x),1),device='cuda')
                return ((f[:,1:].reshape(-1,2)-model(x,a))/.1).reshape(len(f),-1,2)
            ad=abduct(fd);ac=abduct(fc)
            base_d=roll(model,fd[:,0],ad,0.)
            scores=[]
            for angle in plan['allowed_global_angle_grid']:
                scores.append(float((roll(model,fd[:,0],ad,angle)-base_d).abs().mean()))
            chosen=plan['allowed_global_angle_grid'][int(np.argmax(scores))]
            common=roll(model,fc[:,0],ac,0.)-fc[:,-1,0]
            witness=roll(model,fc[:,0],ac,chosen)-fc[:,-1,0]
            true_cf=roll(truth,fc[:,0],ec,plan['hidden_true_angle'])-fc[:,-1,0]
            family=torch.stack([roll(model,fc[:,0],ac,a)-fc[:,-1,0] for a in plan['allowed_global_angle_grid']])
            lo,hi=family.min(0).values,family.max(0).values
            flat=ac.reshape(-1,2);a=torch.ones((len(flat),1),device='cuda')
            # float64 identity check isolates numerical error from CF variation.
            old=.5*(flat.double()**2).sum(-1)
            new=.5*(rotate(flat.double(),a.double(),chosen)**2).sum(-1)
            null=float((old-new).abs().max())
            rec={'seed':seed,'validation_transition_mse':best,'mean_function_rmse':mean_rmse,
                 'selected_global_angle':chosen,'kernel_nll_max_abs_difference':null,
                 'heldout_sign_disagreement_fraction':float((common.sign()!=witness.sign()).float().mean()),
                 'heldout_mean_abs_witness_difference':float((common-witness).abs().mean()),
                 'common_seed_counterfactual_rmse':float(torch.sqrt(((common-true_cf)**2).mean())),
                 'witness_counterfactual_rmse':float(torch.sqrt(((witness-true_cf)**2).mean())),
                 'sampled_gauge_range_mean':float((hi-lo).mean()),
                 'sampled_family_contains_true_cf_fraction':float(((true_cf>=lo)&(true_cf<=hi)).float().mean()),
                 'mean_true_effect':float(true_cf.mean()),'n_confirmation':len(fc)}
            all_estimates.append(common.cpu().numpy())
            np.savez_compressed(folder/'confirmation_arrays.npz',common=common.cpu().numpy(),witness=witness.cpu().numpy(),truth=true_cf.cpu().numpy(),lo=lo.cpu().numpy(),hi=hi.cpu().numpy())
        records.append(rec);(folder/'training_curve.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps(rec),flush=True)
    result={'scope':'nonlinear Gaussian SSM; held-out sensitivity witnesses, NOT all-SCM coverage certificates',
            'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':records,
            'fitted_seed_cf_range_mean':float(np.ptp(np.stack(all_estimates),axis=0).mean()),
            'runtime_seconds':time.perf_counter()-start,
            'interpretation':'Correct conditional kernels still support correct prospective planning. This tests retrospective unit-level counterfactual sensitivity to structural noise assumptions.'}
    (ROOT/'outputs/neural_coupling_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('NEURAL_COUPLING_COMPLETE',result['runtime_seconds'],flush=True)
if __name__=='__main__':run()
