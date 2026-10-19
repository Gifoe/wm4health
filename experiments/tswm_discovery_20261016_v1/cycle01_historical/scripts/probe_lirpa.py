"""Unmodified pinned auto_LiRPA applied to the SAME retrospective uncertainty set.

Run in .venv_verifier (Python 3.11 / torch CPU), not the WorldTS training venv.
Parameter perturbation is one shared L2 ball; angle branching covers [-pi,pi].
No custom relaxation/operator or test-selected settings are used.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
if os.name=='nt':
    import ctypes
    fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import argparse,hashlib,json,pathlib,platform,sys,time
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/auto_LiRPA'))
from auto_LiRPA import BoundedModule,BoundedTensor,PerturbationLpNorm
from confidence_tube import family,simulate,mean,tubes,shared_parameter_tube
from run_confidence_screen import SIGMA
torch.set_num_threads(4);torch.set_default_dtype(torch.float64)

class RetrospectiveGraph(nn.Module):
    def __init__(self,w,P,factual):
        super().__init__();lam,U=np.linalg.eigh(P);S=(U*np.sqrt(lam))@U.T
        self.embedding=nn.Linear(12,12,bias=False)
        with torch.no_grad():self.embedding.weight.copy_(torch.kron(torch.tensor(S),torch.eye(2)))
        self.embedding.weight.requires_grad_(False)
        self.register_buffer('w_hat',torch.tensor(w.reshape(12)))
        self.f=factual.tolist()
    def forward(self,g,theta):
        w=self.embedding(g)+self.w_hat
        a00=w[:,0:1]+w[:,6:7];a01=w[:,2:3]+w[:,8:9]
        a10=w[:,1:2]+w[:,7:8];a11=w[:,3:4]+w[:,9:10]
        b0=w[:,4:5]+w[:,10:11];b1=w[:,5:6]+w[:,11:12]
        x0=g[:,0:1]*0.+self.f[0][0];x1=g[:,0:1]*0.+self.f[0][1]
        co=torch.cos(theta);si=torch.sin(theta)
        for t in range(len(self.f)-1):
            e0=self.f[t+1][0]-w[:,0:1]*self.f[t][0]-w[:,2:3]*self.f[t][1]-w[:,10:11]
            e1=self.f[t+1][1]-w[:,1:2]*self.f[t][0]-w[:,3:4]*self.f[t][1]-w[:,11:12]
            y0=a00*x0+a01*x1+b0+co*e0-si*e1
            y1=a10*x0+a11*x1+b1+si*e0+co*e1
            x0,x1=y0,y1
        return x0-self.f[-1][0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--rep',type=int,default=0);p.add_argument('--query',type=int,default=0);p.add_argument('--horizon',type=int,default=8);p.add_argument('--plan',default='lirpa_plan.json')
    a=p.parse_args();assert pathlib.Path(a.plan).name==a.plan
    pp=ROOT/'outputs'/a.plan;plan=json.loads(pp.read_text())
    folder=ROOT/'outputs/lirpa';folder.mkdir(exist_ok=True)
    saved=np.load(ROOT/f'outputs/feasible_witnesses/rep{a.rep}.npz');w,P=saved['w_hat'],saved['P'];f=saved['factual'][a.query:a.query+1,:a.horizon+1]
    model=RetrospectiveGraph(w,P,f[0]).eval();K=plan['angle_branches']
    g0=torch.zeros((K,12));theta=torch.linspace(-np.pi+np.pi/K,np.pi-np.pi/K,K)[:,None]
    # Independent numpy recursion validates the exact traced forward graph.
    residual=f[:,1:]-mean(w,f[:,:-1],0.)
    for angle in [0.,.7,-1.3]:
        actual=float(model(g0[:1],torch.tensor([[angle]])).item())
        expected=simulate(w,f[:,0],residual,1.,angle)[0,-1,0]-f[0,-1,0]
        assert abs(actual-expected)<1e-10,(actual,expected)
    # Known variance and the SAME family/ellipsoid as all specialized baselines.
    basic=tubes(w,P,SIGMA,f,.05,K);shared=shared_parameter_tube(w,P,SIGMA,f,.05,K)
    nominal_lo,nominal_hi=family(w,f);radius=basic['r'];results=[]
    started=time.perf_counter()
    bounded=BoundedModule(model,(g0,theta),bound_opts={'verbosity':0,'optimize_bound_args':{'iteration':plan['alpha_iterations'],'lr_alpha':.1}},device='cpu')
    parse_seconds=time.perf_counter()-started
    bg=BoundedTensor(g0,PerturbationLpNorm(norm=2,eps=radius))
    bt=BoundedTensor(theta,PerturbationLpNorm(norm=np.inf,eps=np.pi/K))
    for method in plan['methods']:
        start=time.perf_counter();lb,ub=bounded.compute_bounds(x=(bg,bt),method=method)
        lo=float(lb.detach().min());hi=float(ub.detach().max());elapsed=time.perf_counter()-start
        assert lo<=nominal_lo[0]+1e-8 and hi>=nominal_hi[0]-1e-8
        results.append({'method':method,'lo':lo,'hi':hi,'width':hi-lo,'compute_seconds':elapsed})
        (folder/f'rep{a.rep}_q{a.query}_H{a.horizon}.partial.json').write_text(json.dumps({'rep':a.rep,'query':a.query,'horizon':a.horizon,'results':results,'status':'partial_until_worker_complete','plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
        print(json.dumps(results[-1]),flush=True)
    result={'rep':a.rep,'query':a.query,'horizon':a.horizon,'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),
        'verifier_revision':'e3930353a3589f7d10879f1e71223d1eefb57196','code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu',
        'angle_branches':K,'confidence_radius':float(radius),'parse_seconds':parse_seconds,'results':results,
        'same_set_basic_width':float(nominal_hi[0]-nominal_lo[0]+2*basic['paired_radius'][0]),
        'same_set_shared_width':float(shared['hi'][0]-shared['lo'][0]),
        'scope':'classical general-purpose verifier baseline; unmodified source; same global-angle family and shared regression ellipsoid'}
    (folder/f'rep{a.rep}_q{a.query}_H{a.horizon}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('LIRPA_WORKER_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
