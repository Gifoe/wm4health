from common57 import *
import torch,math,numpy as np
from math57 import posterior
from relative57_v1 import relative_kernels
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
fixture=read(OUT/'NATIVE_FIXTURE57_V1.json');row=fixture['points'][0]
z=torch.tensor(row['theta'],device='cuda',dtype=torch.float32,requires_grad=True)
x=torch.tensor(fixture['x'],device='cuda',dtype=torch.float32);a=torch.tensor(fixture['actions'],device='cuda',dtype=torch.long)
lam=z[3:5].exp();u=x@z[5:9].reshape(2,2).T+z[9:11]
post=posterior(z[:3],lam,u,a);H,qf=relative_kernels(lam,u,u,post)
loss=(torch.logsumexp(H,-1)-math.log(len(x))+qf).mean()+post['kl'].mean()/x.shape[1]
grad=torch.autograd.grad(loss,z)[0]
errors={'loss':abs(float(loss.detach())-row['loss']),'gradient':float(np.max(abs(grad.detach().cpu().numpy()-row['gradient'])))}
assert max(errors.values())<5e-4,errors
assert not (OUT/'HELDOUT_OPEN_RECEIPT57_V1.json').exists()
dump(OUT/'CENTERED_CUDA_QUALIFICATION57_V1.json',{'utc':utc(),'status':'PASS_CENTERED_CUDA_NATIVE_LOSS_AND_GRADIENT','errors':errors,'relative_math_sha256':sha(ROOT/'scripts/relative57_v1.py'),'script_sha256':sha(__file__),'source_files_sha256':source_check(),'formal_new_updates':0,'heldout_opened':False})
print(json.dumps({'status':'PASS_CENTERED_CUDA_NATIVE_LOSS_AND_GRADIENT','errors':errors}),flush=True)
