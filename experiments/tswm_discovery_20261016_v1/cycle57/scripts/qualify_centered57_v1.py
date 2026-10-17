"""Qualify the analytic score difference and gamma-one compositional gradient."""
from common57 import *
import torch,numpy as np,math
from math57 import auxiliary,posterior,update_running
from relative57_v1 import relative_kernels
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False

def main():
    old=read(OUT/'QUALIFICATION57_V1.json');fixture=read(OUT/'NATIVE_FIXTURE57_V1.json')
    assert old['math_sha256']==sha(ROOT/'scripts/math57.py')
    x=torch.tensor(fixture['x'],dtype=torch.float64);a=torch.tensor(fixture['actions'],dtype=torch.long)
    checks=[]
    for row in fixture['points']:
        for independent in [False,True]:
            ref=x.flip(0)*.7+.1 if independent else x
            def compute(centered,running):
                z=torch.tensor(row['theta'],dtype=torch.float64,requires_grad=True)
                lam=z[3:5].exp();u=x@z[5:9].reshape(2,2).T+z[9:11];ur=ref@z[5:9].reshape(2,2).T+z[9:11]
                if not centered:value=auxiliary(z[:3],lam,u,ur,a,fixed=row['fixed_q'])[0]
                else:
                    post=posterior(z[:3],lam,u,a,fixed=row['fixed_q']);H,qf=relative_kernels(lam,u,ur,post)
                    logmean=torch.logsumexp(H,-1)-math.log(len(ref))
                    if running:
                        logu=update_running(torch.full_like(logmean,-torch.inf),logmean,1.)
                        term=(torch.exp(H-logu[...,None]).detach()*H).mean(-1)
                    else:term=logmean
                    value=(term+qf).mean()+post['kl'].mean()/x.shape[1]
                return float(value.detach()),torch.autograd.grad(value,z)[0].numpy()
            native_value,native_grad=compute(False,False)
            centered_value,centered_grad=compute(True,False)
            _,running_grad=compute(True,True)
            errors={'loss':abs(native_value-centered_value),'gradient':float(np.max(abs(native_grad-centered_grad))),'gamma1_gradient':float(np.max(abs(native_grad-running_grad)))}
            assert max(errors.values())<1e-9,errors
            checks.append({'fixed_q':row['fixed_q'],'independent_reference':independent,'errors':errors})
    assert not (OUT/'HELDOUT_OPEN_RECEIPT57_V1.json').exists()
    record={'utc':utc(),'status':'PASS_ANALYTIC_POSITIVE_RELATIVE_KERNEL_AND_GAMMA1_GRADIENT','checks':checks,'relative_math_sha256':sha(ROOT/'scripts/relative57_v1.py'),'unchanged_core_math_sha256':sha(ROOT/'scripts/math57.py'),'original_qualification_sha256':sha(OUT/'QUALIFICATION57_V1.json'),'failure_receipt_sha256':sha(OUT/'FAILED_ATTEMPT57_V1.json'),'clarification_sha256':sha(ROOT/'NORMALIZER_CLARIFICATION_V2.md'),'formal_new_updates':0,'heldout_opened':False,'source_files_sha256':source_check()}
    dump(OUT/'CENTERED_QUALIFICATION57_V1.json',record)
    print(json.dumps({'status':record['status'],'max_error':max(max(r['errors'].values()) for r in checks)}),flush=True)

if __name__=='__main__':main()
