"""Native numerical/forecast contracts only. No proposed method or formal fit."""
import pathlib,sys,os,json,hashlib,datetime,traceback,warnings
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from torch import nn
from scipy.linalg import expm,solve_continuous_lyapunov
S=R/'source_archive/ncdssm';sys.path.insert(0,str(S/'src'))
from ncdssm.models import BaseLTI,NCDSSMNL
from ncdssm.models.components import AuxInferenceModel,GaussianOutput
from ncdssm.modules import MLP
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,obj):pathlib.Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')
O=R/'outputs/native_qualification_v1';assert not O.exists();O.mkdir(parents=True)
state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'comparative_fits':0,'discarded_updates':0};dump(O/'status.json',state)
try:
 manifest=json.loads((S/'SOURCE_TREE.json').read_text(encoding='utf-8'));assert manifest['revision']=='7212bba9616d81c948f507eef9eb1f7d5cf1954e'
 for item in manifest['files']:assert sha(S/item['path'])==item['sha256']
 assert not torch.cuda.is_initialized();torch.set_default_dtype(torch.float64);torch.manual_seed(202645100);F0=np.array([[-.3,.2],[-.1,-.4]]);Q0=np.diag([.07,.04]);H0=np.array([[1.,.3],[.2,1.]]);R0=np.diag([.05,.08]);m0=np.array([.15,-.12]);P0=np.diag([.3,.4]);past=np.array([0.,.2,.7]);future=np.array([1.2,1.8,2.7]);y=np.array([[[.2,-.1],[.1,.15],[-.05,.2]],[[.1,.1],[.2,.05],[.15,-.1]]]);mask=np.array([[1.,0.,1.],[1.,1.,1.]])
 base=BaseLTI(z_dim=2,y_dim=2,u_dim=0,F=torch.tensor(F0),Q=torch.tensor(np.diag(Q0)),H=torch.tensor(H0),R=torch.tensor(np.diag(R0)),mu0=torch.tensor(m0),Sigma0=torch.tensor(np.diag(P0)),integration_method='matrix_exp',integration_step_size=.05)
 def independent_filter(F0,y,mask):
  stationary=solve_continuous_lyapunov(F0,-Q0);all_m=[];all_p=[];all_lp=[]
  for values,observed in zip(y,mask):
   m=m0.copy();p=P0.copy();ms=[];ps=[];lp=0.
   for t,value in enumerate(values):
    if t:
     A=expm(F0*(past[t]-past[t-1]));qd=stationary-A@stationary@A.T;m=A@m;p=A@p@A.T+qd
    if observed[t]:
     innovation=value-H0@m;cov=H0@p@H0.T+R0;lp-=.5*(2*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+innovation@np.linalg.solve(cov,innovation));gain=np.linalg.solve(cov,H0@p).T;m+=gain@innovation;remaining=np.eye(2)-gain@H0;p=remaining@p@remaining.T+gain@R0@gain.T
    ms.append(m.copy());ps.append(p.copy())
   all_m.append(ms);all_p.append(ps);all_lp.append(lp)
  return np.asarray(all_m).transpose(1,0,2),np.asarray(all_p).transpose(1,0,2,3),np.asarray(all_lp)
 filtered=base.filter(torch.tensor(y),torch.tensor(mask),torch.tensor(past));means,covs,lps=independent_filter(F0,y,mask);actual_m=filtered['filtered_mus'].detach().numpy();L=filtered['filtered_LSigmas'].detach().numpy();actual_p=L@L.transpose(0,1,3,2);actual_lp=filtered['log_prob'].detach().numpy();filter_errors={'mean':float(np.max(np.abs(actual_m-means))),'covariance':float(np.max(np.abs(actual_p-covs))),'log_likelihood':float(np.max(np.abs(actual_lp-lps)))};assert max(filter_errors.values())<1e-8,filter_errors
 loss=-filtered['log_prob'].mean();loss.backward();gradient=float(base.dynamics.F.grad[0,0]);h=1e-5;plus=F0.copy();minus=F0.copy();plus[0,0]+=h;minus[0,0]-=h;expected_gradient=-(independent_filter(plus,y,mask)[2].mean()-independent_filter(minus,y,mask)[2].mean())/(2*h);gradient_error=abs(gradient-expected_gradient);assert gradient_error<5e-6,gradient_error
 with torch.no_grad():native=base.forecast(torch.tensor(y[:1]),torch.tensor(mask[:1]),torch.tensor(past),torch.tensor(future),num_samples=16384)['z_forecast'][:,0].numpy()
 stationary=solve_continuous_lyapunov(F0,-Q0);m=means[-1,0].copy();p=covs[-1,0].copy();true_means=[];true_covs=[];last=past[-1]
 for t in future:
  A=expm(F0*(t-last));qd=stationary-A@stationary@A.T;m=A@m;p=A@p@A.T+qd;true_means.append(m.copy());true_covs.append(p.copy());last=t
 joint=np.empty((3,3,2,2))
 for i in range(3):
  for j in range(3):joint[i,j]=expm(F0*(future[i]-future[j]))@true_covs[j] if i>=j else true_covs[i]@expm(F0.T*(future[j]-future[i]))
 rng=np.random.default_rng(202645101);z=means[-1,0]+rng.normal(size=(16384,2))@np.linalg.cholesky(covs[-1,0]).T;ordinary=[];last=past[-1]
 for t in future:
  A=expm(F0*(t-last));qd=stationary-A@stationary@A.T;z=z@A.T+rng.normal(size=z.shape)@np.linalg.cholesky((qd+qd.T)/2).T;ordinary.append(z.copy());last=t
 ordinary=np.stack(ordinary,1);sampling=[]
 for name,draws in [('native_marginal_time_samples',native),('ordinary_joint_Markov_samples',ordinary)]:
  mean=draws.mean(0);marginal=np.asarray([np.cov(draws[:,t].T,ddof=1) for t in range(3)]);mean_se=np.sqrt(np.asarray([np.diag(p) for p in true_covs])/16384);cov_se=np.asarray([np.sqrt((np.outer(np.diag(p),np.diag(p))+p*p)/(16384-1)) for p in true_covs]);assert np.all(np.abs(mean-true_means)<5*mean_se+1e-5) and np.all(np.abs(marginal-true_covs)<5*cov_se+1e-5),(name,mean,marginal)
  centered=draws-mean;cross=centered[:,0].T@centered[:,2]/(len(draws)-1);target=np.zeros((2,2)) if name.startswith('native') else joint[0,2];cross_se=np.sqrt((np.outer(np.diag(true_covs[0]),np.diag(true_covs[2]))+target*target)/(16384-1));assert np.all(np.abs(cross-target)<5*cross_se+1e-5),(name,cross,target);true_cross=joint[0,2];se_joint=np.sqrt((np.outer(np.diag(true_covs[0]),np.diag(true_covs[2]))+true_cross*true_cross)/(16384-1));mismatch_z=float(np.max(np.abs(cross-true_cross)/se_joint));contrast=np.var(draws[:,2]-draws[:,0],axis=0,ddof=1);true_contrast=np.diag(true_covs[0]+true_covs[2]-joint[0,2]-joint[2,0]);sampling.append({'kind':name,'marginal_mean_max_abs':float(np.max(np.abs(mean-true_means))),'marginal_covariance_max_abs':float(np.max(np.abs(marginal-true_covs))),'cross_time_covariance_0_2':cross.tolist(),'exact_joint_cross_time_covariance_0_2':true_cross.tolist(),'max_cross_discrepancy_in_exact_joint_MC_SE':mismatch_z,'temporal_difference_variance':contrast.tolist(),'exact_joint_temporal_difference_variance':true_contrast.tolist()})
 assert sampling[0]['max_cross_discrepancy_in_exact_joint_MC_SE']>10
 raw=O/'linear_sampling.npz';np.savez_compressed(raw,native_z=native,ordinary_joint_z=ordinary,exact_mean=np.asarray(true_means),exact_covariance_blocks=joint,observations=y,mask=mask,past_times=past,future_times=future)
 # Distinct nonlinear API/backprop qualification, actual author components.
 torch.set_default_dtype(torch.float32);torch.manual_seed(202645102);aux=AuxInferenceModel(nn.Identity(),GaussianOutput(nn.Identity(),dist_dim=2,use_tied_cov=True,use_trainable_cov=True),aux_dim=2,concat_mask=False);emission=GaussianOutput(nn.Identity(),dist_dim=2,use_tied_cov=True,use_trainable_cov=True,use_independent=False);drift=MLP(in_dim=6,h_dim=64,out_dim=6,n_hidden_layers=1,nonlinearity=nn.Softplus);model=NCDSSMNL(aux,emission,aux_dim=2,z_dim=6,y_dim=2,u_dim=0,f=drift,H=torch.eye(2,6),integration_step_size=.05,integration_method='rk4');times=torch.tensor([0.,.11,.23,.37,.49,.63,.79,.92]);phase=torch.arange(4,dtype=torch.float32)[:,None];fixture=torch.stack((torch.sin(times[None]+phase),torch.cos(times[None]+phase)),-1);fixture_mask=torch.ones(4,8);fixture_mask[:,3]=0;optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);losses=[]
 for step in range(10):
  optimizer.zero_grad();result=model(fixture,fixture_mask,times,num_samples=2);loss=-result['elbo'].mean()/8;assert torch.isfinite(loss);loss.backward();gradient_values=[p.grad for p in model.parameters() if p.grad is not None];assert gradient_values and all(torch.isfinite(g).all() for g in gradient_values);torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();losses.append(float(loss))
 state['discarded_updates']=10;dump(O/'status.json',state);calls=[]
 def observe(module,args,kwargs,output):calls.append({'input_batch':int(args[0].shape[0]),'requested_aux_samples':kwargs.get('num_samples'),'returned_aux_shape':list(output[0].shape)})
 handle=model.aux_inference_net.register_forward_hook(observe,with_kwargs=True);future_times=torch.tensor([1.1,1.31,1.57]);model.eval();torch.manual_seed(202645103);forecast=model.forecast(fixture[:1],fixture_mask[:1],times,future_times,num_samples=16);handle.remove();assert calls==[{'input_batch':1,'requested_aux_samples':1,'returned_aux_shape':[1,1,8,2]}];assert forecast['forecast'].shape==(16,1,3,2) and all(torch.isfinite(v).all() for v in forecast.values());checkpoint=O/'discarded_preflight.pt';torch.save(model.state_dict(),checkpoint);replay_state=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(replay_state);torch.manual_seed(202645103);replay=model.forecast(fixture[:1],fixture_mask[:1],times,future_times,num_samples=16);replay_error=float((forecast['forecast']-replay['forecast']).abs().max());assert replay_error==0,replay_error
 calls2=[]
 def observe2(module,args,kwargs,output):calls2.append({'input_batch':int(args[0].shape[0]),'requested_aux_samples':kwargs.get('num_samples'),'returned_aux_shape':list(output[0].shape)})
 handle=model.aux_inference_net.register_forward_hook(observe2,with_kwargs=True);torch.manual_seed(202645104);batch_control=model.forecast(fixture[:1].repeat(16,1,1),fixture_mask[:1].repeat(16,1),times,future_times,num_samples=1);handle.remove();assert calls2==[{'input_batch':16,'requested_aux_samples':1,'returned_aux_shape':[1,16,8,2]}] and batch_control['forecast'].shape==(1,16,3,2)
 assert not torch.cuda.is_initialized();result={'status':'PASS_WITH_EXPLICIT_SAMPLING_CONTRACTS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':manifest['revision'],'source_files_verified':len(manifest['files']),'linear_inference_max_abs':filter_errors,'independent_gradient_abs':gradient_error,'linear_sampling_draws_per_contract':16384,'sampling':sampling,'nonlinear_discarded_updates':10,'nonlinear_parameters':sum(p.numel() for p in model.parameters()),'nonlinear_preflight_losses':losses,'checkpoint_replay_max_abs':replay_error,'native_aux_trace':calls,'ordinary_independent_outer_batch_trace':calls2,'source_unchanged':True,'GPU_initialized':False,'sampling_arrays_sha256':sha(raw),'discarded_checkpoint_sha256':sha(checkpoint),'source_sha256':sha(pathlib.Path(__file__)),'plan_sha256':sha(R/'NATIVE_BASELINE_PLAN.md'),'comparative_effects':0,'not_a_new_model_claim':True};dump(O/'qualification.json',result);state.update(status='completed',completed_utc=result['utc']);dump(O/'status.json',state);print(json.dumps({k:v for k,v in result.items() if k not in ['sampling','nonlinear_preflight_losses']}),flush=True)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());dump(O/'status.json',state);dump(O/'qualification.json',state);raise
