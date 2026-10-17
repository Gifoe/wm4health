"""Qualify the CUDA port and proper likelihood before any formal model update."""
from common57 import *
import math,numpy as np
import torch
from math57 import auxiliary,proper_loss,posterior,prior,update_running
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def independent(z,x,u,fixed):
    lam=z[3:5].exp();means=x@z[5:9].reshape(2,2).T+z[9:11]
    return auxiliary(z[:3],lam,means,means,u,fixed=fixed)

def main():
    pins=source_check();fixture=read(OUT/'NATIVE_FIXTURE57_V1.json')
    assert fixture['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    x=torch.tensor(fixture['x'],dtype=torch.float64);u=torch.tensor(fixture['actions'],dtype=torch.long)
    checks=[]
    for row in fixture['points']:
        z=torch.tensor(row['theta'],dtype=torch.float64,requires_grad=True)
        loss,p,_=independent(z,x,u,row['fixed_q']);g=torch.autograd.grad(loss,z)[0]
        errors={'loss':abs(float(loss.detach())-row['loss']),'gradient':float(np.max(abs(g.detach().numpy()-row['gradient'])))}
        for a,b in [('m','mean'),('C','cov'),('cross','cross')]:errors[b]=float(np.max(abs(p[a].detach().numpy()-np.asarray(row['moments'][b]))))
        assert max(errors.values())<2e-6,(row['fixed_q'],errors)
        checks.append({'theta':row['theta'],'fixed_q':row['fixed_q'],'errors':errors})
    assert torch.cuda.is_available();assert '5090' in torch.cuda.get_device_name(0)
    z=torch.tensor(fixture['points'][1]['theta'],dtype=torch.float32,device='cuda',requires_grad=True)
    loss,p,_=independent(z,x.float().cuda(),u.cuda(),False);grad=torch.autograd.grad(loss,z)[0]
    target=next(r for r in fixture['points'] if r['theta']==fixture['points'][1]['theta'] and not r['fixed_q'])
    cuda_errors={'loss':abs(float(loss.detach())-target['loss']),'gradient':float(np.max(abs(grad.detach().cpu().numpy()-target['gradient'])))}
    assert max(cuda_errors.values())<5e-4,cuda_errors
    # gamma=1 must give exactly the ordinary independent-reference gradient.
    z0=torch.tensor(fixture['points'][0]['theta'],dtype=torch.float64)
    reference=x.flip(0)*.7+.1
    def calc(running):
        z=z0.clone().requires_grad_();lam=z[3:5].exp();means=x@z[5:9].reshape(2,2).T+z[9:11];ur=reference@z[5:9].reshape(2,2).T+z[9:11]
        loss,_,logmean=auxiliary(z[:3],lam,means,ur,u)
        if running:
            logu=update_running(torch.full_like(logmean,-torch.inf),logmean,1.)
            loss,_,_=auxiliary(z[:3],lam,means,ur,u,log_running=logu)
        return torch.autograd.grad(loss,z)[0]
    gamma_error=float((calc(False)-calc(True)).abs().max());assert gamma_error<1e-10
    # Proper likelihood is checked using a separately assembled joint observation covariance.
    theta=z0[:3];lam=z0[3:5].exp();k=torch.tensor([.4,.7],dtype=torch.float64)
    actual=float(proper_loss(theta,lam,k,x,u));J,_,_,_=prior(theta,u)
    S=torch.linalg.inv(J).numpy()+np.kron(np.eye(x.shape[1]),np.diag(1/lam.numpy()))[None]
    y=(torch.sinh(k*x)/k).numpy().reshape(7,-1)
    jac=np.log(np.cosh((k*x).numpy())).sum((1,2))
    nll=[]
    for i in range(7):nll.append(.5*(12*math.log(2*math.pi)+np.linalg.slogdet(S[i])[1]+y[i]@np.linalg.solve(S[i],y[i]))-jac[i])
    proper_error=abs(actual-float(np.mean(nll)/6));assert proper_error<1e-9,proper_error
    assert source_check()==pins
    record={'utc':utc(),'status':'PASS_NATIVE_LOSS_MOMENT_GRADIENT_CUDA_AND_PROPER_LIKELIHOOD','native_comparisons':checks,'cuda_float32_errors':cuda_errors,'gamma1_gradient_error':gamma_error,'proper_joint_density_error':proper_error,'native_fixture_sha256':sha(OUT/'NATIVE_FIXTURE57_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'math_sha256':sha(ROOT/'scripts/math57.py'),'model_sha256':sha(ROOT/'scripts/models57.py'),'qualification_sha256':sha(__file__),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'native_source_sha256':pins,'formal_updates':0}
    dump(OUT/'QUALIFICATION57_V1.json',record)
    print(json.dumps({'status':record['status'],'native_max_error':max(max(r['errors'].values()) for r in checks),'cuda_errors':cuda_errors,'gamma1_gradient_error':gamma_error,'proper_density_error':proper_error}),flush=True)

if __name__=='__main__':main()
