"""Discarded numerical qualifications and device timing, no EVAL access."""
from common46 import *
from ncdssm.models import BaseLTI
from scipy.linalg import expm,solve_continuous_lyapunov
import datetime,os
torch.set_num_threads(1)

def main():
 out=R/'outputs/qualification_v1';assert not out.exists();out.mkdir(parents=True)
 manifest=read(SRC/'SOURCE_TREE.json')
 for item in manifest['files']:assert sha(SRC/item['path'])==item['sha256']
 torch.set_default_dtype(torch.float64);seed(20264690);F=np.array([[-.3,.2],[-.1,-.4]]);Q=np.diag([.07,.04]);m=np.array([.15,-.12]);P=np.diag([.3,.4]);times=np.array([.5,1.,2.]);base=BaseLTI(z_dim=2,y_dim=2,u_dim=0,F=torch.tensor(F),Q=torch.tensor(np.diag(Q)),integration_method='matrix_exp');n=32768
 values=state_paths(base,torch.tensor(m)[None].repeat(n,1),torch.tensor(np.linalg.cholesky(P))[None].repeat(n,1,1),0.,torch.tensor(times),torch.Generator().manual_seed(20264691)).numpy();stationary=solve_continuous_lyapunov(F,-Q);means=np.stack([expm(F*t)@m for t in times]);covs=np.stack([stationary+expm(F*t)@(P-stationary)@expm(F.T*t) for t in times]);flat=values.reshape(n,6);exact=np.empty((6,6))
 for i in range(3):
  for j in range(3):exact[2*i:2*i+2,2*j:2*j+2]=expm(F*(times[i]-times[j]))@covs[j] if i>=j else covs[i]@expm(F.T*(times[j]-times[i]))
 actual=np.cov(flat.T);se=np.sqrt((np.outer(np.diag(exact),np.diag(exact))+exact**2)/(n-1));max_z=float(np.max(np.abs(actual-exact)/se));assert max_z<5.5,max_z;assert np.max(np.abs(flat.mean(0)-means.ravel())/np.sqrt(np.diag(exact)/n))<5.5
 np.savez_compressed(out/'linear_joint.npz',draws=values,exact_mean=means,exact_covariance=exact)
 torch.set_default_dtype(torch.float32);seed(20264692);aux=AuxInferenceModel(nn.Identity(),GaussianOutput(nn.Identity(),dist_dim=2,use_tied_cov=True,use_trainable_cov=True),aux_dim=2);emission=GaussianOutput(nn.Identity(),dist_dim=2,use_tied_cov=True,use_trainable_cov=True,use_independent=False);model=NCDSSMNL(aux,emission,aux_dim=2,z_dim=6,y_dim=2,u_dim=0,f=MLP(in_dim=6,h_dim=64,out_dim=6,n_hidden_layers=1,nonlinearity=nn.Softplus),H=torch.eye(2,6),integration_step_size=.05,integration_method='rk4');model.load_state_dict(torch.load(R.parent/'cycle45/outputs/native_qualification_v1/discarded_preflight.pt',weights_only=True,map_location='cpu'))
 # Same Brownian increments aggregated into coarser grids, constant diagonal diffusion.
 seed(20264693);z0=torch.randn(256,6);dt=.00625;increments=torch.randn(160,256,6)*math.sqrt(dt);paths={}
 with torch.no_grad():
  for factor in [1,2,4,8]:
   z=z0.clone();states=[]
   for j in range(0,160,factor):z=z+dt*factor*model.base_ssm.dynamics.f(z)+model.base_ssm.dynamics.GQGt(z).diagonal(dim1=-2,dim2=-1).sqrt()*increments[j:j+factor].sum(0);states.append(z.clone())
   paths[factor]=z.numpy()
 errs=[float(np.sqrt(np.mean((paths[f]-paths[1])**2))) for f in [8,4,2]];assert errs[0]>errs[1]>errs[2]>0,errs;np.savez_compressed(out/'nonlinear_coupled.npz',**{f'h{f}':a for f,a in paths.items()})
 history=torch.stack((torch.sin(torch.arange(8)*.1),torch.cos(torch.arange(8)*.1)),-1)[None];rng_before=torch.random.get_rng_state().clone();weight_before=checksum(model);draws,states=sample_joint(model,history,torch.ones(1,8),torch.arange(8)*.1,torch.arange(8,12)*.1,128,20264694,return_states=True);assert torch.isfinite(draws).all() and torch.isfinite(states).all() and draws.shape==(1,128,4,2);assert torch.equal(rng_before,torch.random.get_rng_state()) and weight_before==checksum(model)
 np.savez_compressed(out/'nonlinear_joint.npz',draws=draws.numpy(),states=states.numpy())
 # Verify ordinary contextual model executes original source filter/ELBO, anchors included.
 data=np.load(R/'data/train.npz');anchors=np.load(R/'data/anchors.npz')['X'];timings=[]
 for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
  for obs,dyn in [(0,0),(1,1)]:
   seed(20264695);w=World(obs,dyn).to(device);opt=torch.optim.Adam(w.parameters(),lr=1e-3);losses=[];dur=[]
   for i in range(7):
    y=torch.as_tensor(anchors[:16] if i%2 else data['v0_s1'][:16],device=device);v=None if i%2 else 0;system=i%4;w.train();opt.zero_grad()
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();loss=w.loss(y,v,system);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in w.parameters() if p.grad is not None);opt.step()
    if device=='cuda':torch.cuda.synchronize()
    dur.append(time.perf_counter()-start);losses.append(float(loss.detach().cpu()))
   start=time.perf_counter();future=w.sample(data['v0_s1'][:8,:HIST],0,1,32,20264696);es=energy(future,data['v0_s1'][:8,HIST:]);assert torch.isfinite(es).all();forecast_time=time.perf_counter()-start;timings.append({'device':device,'observation_shared':obs,'dynamics_compositional':dyn,'parameters':sum(p.numel() for p in w.parameters()),'discarded_updates':7,'median_temporal_step_seconds':float(np.median(dur[2::2])),'median_anchor_step_seconds':float(np.median(dur[1::2])),'eight_origin_32_draw_seconds':forecast_time,'losses':losses})
 for item in manifest['files']:assert sha(SRC/item['path'])==item['sha256']
 device=min(['cpu']+(['cuda'] if torch.cuda.is_available() else []),key=lambda d:np.mean([.8*t['median_temporal_step_seconds']+.2*t['median_anchor_step_seconds'] for t in timings if t['device']==d]))
 result={'status':'QUALIFIED_ORDINARY_ADAPTER_AND_CONTEXTUAL_BASELINE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'linear_joint_max_MC_SE':max_z,'nonlinear_coupled_strong_RMS_h_05_025_0125_vs_00625':errs,'nonlinear_scope':'author default constant diagonal diffusion; EM, not native marginal-moment forecast law','RNG_restored':True,'source_unchanged':True,'formal_fits':0,'timings':timings,'selected_training_device':device,'source_sha256':sha(__file__),'common_sha256':sha(R/'scripts/common46.py'),'adapter_sha256':sha(R/'scripts/joint_adapter46.py')};dump(out/'qualification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
