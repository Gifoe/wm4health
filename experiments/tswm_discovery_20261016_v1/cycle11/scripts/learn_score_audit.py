"""Controlled source-component learning study; no full-paper replication claim."""
import runtime_profile
import ast,contextlib,datetime,hashlib,json,math,pathlib,sys,time,traceback
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/score_learning';OUT.mkdir(parents=True,exist_ok=True)
class Tee:
 def __init__(self,*files):self.files=files
 def write(self,s):
  for f in self.files:f.write(s);f.flush()
 def flush(self):
  for f in self.files:f.flush()
def extract(path,name):
 text=path.read_text(encoding='utf-8');tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
 scope={'torch':torch,'nn':nn,'F':F,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
 return scope[name],ast.get_source_segment(text,node)
def mlp(inp,out):return nn.Sequential(nn.Linear(inp,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,out))
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=mlp(2,2);self.predictor=mlp(2,40);self.decoder=mlp(2,2)
 def params(self,z):
  value=self.predictor(z);mu=value[:,:16].reshape(-1,8,2);std=F.softplus(value[:,16:32].reshape(-1,8,2))+.03;weights=value[:,32:].softmax(-1);return mu,std,weights
def data(n,generator,device,rot):
 x=torch.randn(n,2,generator=generator,device=device);noise=torch.randn(n,2,generator=generator,device=device);sign=torch.where(torch.rand(n,1,generator=generator,device=device)<.25,-1.,1.);y=.9*(x@rot.T)*sign+math.sqrt(.19)*noise;return x,y
def weighted_ES(samples,weights,target):
 B,K,M,D=samples.shape
 # MC U-statistic is unbiased within a mixture component, including when M=4.
 first=(weights*((samples-target[:,None,None]).norm(dim=-1).mean(-1))).sum(-1)
 flat=samples.reshape(B,K*M,D);dist=torch.cdist(flat,flat).reshape(B,K,M,K,M)
 factor=torch.ones(K,M,K,M,device=samples.device,dtype=samples.dtype)/(M*M)
 same=torch.arange(K,device=samples.device)
 for k in range(K):factor[k,:,k,:]=(~torch.eye(M,device=samples.device,dtype=torch.bool)).to(samples.dtype)/(M*(M-1))
 second=(dist*factor[None]*weights[:,:,None,None,None]*weights[:,None,None,:,None]).sum((1,2,3,4))
 return (first-.5*second).mean()
def nll(mu,std,weights,z):
 lognormal=-.5*(((z[:,None]-mu)/std).square()+2*std.log()+math.log(2*math.pi)).sum(-1)
 return -torch.logsumexp(weights.clamp_min(1e-12).log()+lognormal,dim=1).mean()
def loss(model,reg,entropy,x,y,arm,draws=4):
 z=model.encoder(x);target=model.encoder(y);mu,std,weights=model.params(z)
 noise=torch.randn(len(x),8,draws,2,device=x.device);samples=mu[:,:,None]+std[:,:,None]*noise
 es=weighted_ES(samples,weights,target);regularizer=reg(torch.stack([z,target]));total=es+regularizer;aux=torch.zeros((),device=x.device)
 if arm=='energy_information':aux=2*(nll(mu,std,weights,target)-.5*(entropy(z)+entropy(target)));total=total+aux
 if arm in ['energy_observed','observed_only']:
  aux=weighted_ES(model.decoder(samples),weights,y);total=total+aux if arm=='energy_observed' else regularizer+aux
 return total,{'latent_energy':float(es.detach()),'sigreg':float(regularizer.detach()),'aux':float(aux.detach())}
def fold_np(x):
 radius=np.linalg.norm(x,axis=1,keepdims=True);return np.stack([x[:,0]**2-x[:,1]**2,2*x[:,0]*x[:,1]],axis=1)/np.maximum(radius,1e-15)
def probes(ztrain,ztest,xtrain,xtest):
 mean=ztrain.mean(0);std=ztrain.std(0).clip(1e-6);train=(ztrain-mean)/std;test=(ztest-mean)/std;result={}
 for key,estimator in [('ridge',Ridge(alpha=1e-4)),('extra_trees',ExtraTreesRegressor(n_estimators=100,min_samples_leaf=5,n_jobs=1,random_state=3901))]:
  estimator.fit(train,xtrain);pred=estimator.predict(test);mse=float(np.mean(np.sum((pred-xtest)**2,axis=1)));result[key]={'raw_state_MSE':mse,'one_step_raw_mean_MSE':mse*(.5*.9)**2}
 return result
def main():
 assert not (OUT/'results.json').exists(),'completed experiment must not be overwritten'
 assert torch.cuda.is_available();torch.set_num_threads(1);device='cuda';torch.backends.cuda.matmul.allow_tf32=False
 SIGReg,regbody=extract(ROOT/'sources/ac_mtm/module.py','SIGReg');entropy,entropybody=extract(ROOT/'sources/infoldm/implicit_nuisance/losses/ssl/entropy.py','logdet_entropy')
 (OUT/'native_SIGReg.py.txt').write_text(regbody,encoding='utf-8');(OUT/'native_logdet_entropy.py.txt').write_text(entropybody,encoding='utf-8')
 cfg={'seeds':[2021,2022,2023],'arms':['energy','energy_information','energy_observed','observed_only'],'updates':6000,'batch':256,'lr':.001,'components':8,'samples_per_component':4,'sigreg_weight':1.,'sigreg_projections':64,'sigreg_knots':17,'information_weight':2.,'rho':.9,'p_flip':.25,'angle':.3,'validation_seed':9101,'validation_n':8192,'probe_train_seed':9102,'probe_train_n':16384,'probe_test_seed':9103,'probe_test_n':8192,'checkpoint_selection':'last fixed update, no physical-readout/test selection','scope':'new controlled learning study using unchanged native source components; not full native method reproduction'}
 (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'native_SIGReg_SHA256':hashlib.sha256((ROOT/'sources/ac_mtm/module.py').read_bytes()).hexdigest(),'native_entropy_SHA256':hashlib.sha256((ROOT/'sources/infoldm/implicit_nuisance/losses/ssl/entropy.py').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 theta=.3;rot=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],device=device)
 vg=torch.Generator(device=device).manual_seed(9101);vx,vy=data(8192,vg,device,rot)
 prng=np.random.default_rng(9102);px=prng.normal(size=(16384,2));tx=np.random.default_rng(9103).normal(size=(8192,2))
 np.savez_compressed(OUT/'evaluation_states.npz',probe_train=px,probe_test=tx,validation_x=vx.cpu().numpy(),validation_y=vy.cpu().numpy())
 controls={'identity':probes(px,tx,px,tx),'exact_fold':probes(fold_np(px),fold_np(tx),px,tx)}
 rows=[];curves=[];started=time.time();torch.cuda.reset_peak_memory_stats()
 for seed in cfg['seeds']:
  for arm in cfg['arms']:
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Model().to(device);reg=SIGReg(knots=17,num_proj=64).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
   dg=torch.Generator(device=device).manual_seed(seed+10000);start=time.time()
   active=sum(p.numel() for p in model.encoder.parameters())+sum(p.numel() for p in model.predictor.parameters())+(sum(p.numel() for p in model.decoder.parameters()) if arm in ['energy_observed','observed_only'] else 0)
   for step in range(6000):
    x,y=data(256,dg,device,rot);opt.zero_grad();value,metrics=loss(model,reg,entropy,x,y,arm)
    if not torch.isfinite(value):raise RuntimeError(f'Nonfinite loss {seed} {arm} {step}')
    value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
    if step%500==0 or step==5999:
     curve={'seed':seed,'arm':arm,'update':step+1,'total':float(value.detach()),**metrics,'elapsed_seconds':time.time()-start};curves.append(curve);print('progress',json.dumps(curve),flush=True)
   torch.cuda.synchronize();training_seconds=time.time()-start;model.eval()
   with torch.no_grad():
    validation=[]
    for idx in range(0,8192,256):
     ll,mm=loss(model,reg,entropy,vx[idx:idx+256],vy[idx:idx+256],arm,draws=8);validation.append({'total':float(ll),**mm})
    ztrain=model.encoder(torch.tensor(px,device=device,dtype=torch.float32)).cpu().numpy();ztest=model.encoder(torch.tensor(tx,device=device,dtype=torch.float32)).cpu().numpy();neg=model.encoder(torch.tensor(-tx,device=device,dtype=torch.float32)).cpu().numpy()
   probe=probes(ztrain,ztest,px,tx);cov=np.cov(ztest.T);fiber=float(np.mean(np.sum((ztest-neg)**2,axis=1))/(2*np.trace(cov)));validation={k:float(np.mean([r[k] for r in validation])) for k in validation[0]}
   row={'seed':seed,'arm':arm,'active_trainable_parameters':active,'total_instantiated_parameters':sum(p.numel() for p in model.parameters()),'training_seconds':training_seconds,'validation':validation,'latent_covariance_eigenvalues':np.linalg.eigvalsh(cov).tolist(),'latent_mean':ztest.mean(0).tolist(),'sign_fiber_separation_ratio':fiber,'probes':probe}
   rows.append(row);torch.save({'state_dict':model.cpu().state_dict(),'config':cfg,'row':row},OUT/f'{arm}_seed{seed}.pt');np.savez_compressed(OUT/f'latents_{arm}_seed{seed}.npz',train=ztrain,test=ztest,negated_test=neg)
   (OUT/'partial_results.json').write_text(json.dumps({'rows':rows,'controls':controls},indent=2,allow_nan=False),encoding='utf-8');(OUT/'training_curves.json').write_text(json.dumps(curves,indent=2,allow_nan=False),encoding='utf-8');print('result',json.dumps(row),flush=True)
 result={'config':cfg,'rows':rows,'probe_controls':controls,'elapsed_seconds':time.time()-started,'max_cuda_memory_allocated':torch.cuda.max_memory_allocated(),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'No native full model, new method superiority, or novelty verdict is implied by this controlled study.'}
 assert len(rows)==12;(OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print('COMPLETE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':
 with (OUT/'stdout.log').open('a',encoding='utf-8') as stdout,(OUT/'stderr.log').open('a',encoding='utf-8') as stderr,contextlib.redirect_stdout(Tee(sys.stdout,stdout)),contextlib.redirect_stderr(Tee(sys.stderr,stderr)):
  try:main()
  except BaseException:traceback.print_exc();raise
