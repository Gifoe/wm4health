"""Independent dense Gaussian audit of retained arrays and actual source origins."""
import pathlib,sys,os,json,hashlib,datetime,inspect
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.linalg import expm,solve_continuous_lyapunov
S=R/'source_archive/ncdssm';sys.path.insert(0,str(S/'src'))
from ncdssm.models import BaseLTI,NCDSSMNL
import torch
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
out=R/'outputs/verification_v1.json';assert not out.exists();q=read(R/'outputs/native_qualification_v1/qualification.json');assert q['status']=='PASS_WITH_EXPLICIT_SAMPLING_CONTRACTS';manifest=read(S/'SOURCE_TREE.json');hashes={r['path']:r['sha256'] for r in manifest['files']};origins=[]
for name,module in sorted(sys.modules.items()):
 if name.startswith('ncdssm') and getattr(module,'__file__',None):
  p=pathlib.Path(module.__file__).resolve();assert (S/'src').resolve() in p.parents,(name,str(p));rel=p.relative_to(S).as_posix();assert sha(p)==hashes[rel];origins.append({'module':name,'path':rel,'sha256':sha(p)})
assert len(origins)>=8 and not torch.cuda.is_initialized()
raw=R/'outputs/native_qualification_v1/linear_sampling.npz';assert sha(raw)==q['sampling_arrays_sha256'];z=np.load(raw);F=np.array([[-.3,.2],[-.1,-.4]]);Q=np.diag([.07,.04]);H=np.array([[1.,.3],[.2,1.]]);Rnoise=np.diag([.05,.08]);m0=np.array([.15,-.12]);P0=np.diag([.3,.4]);stationary=solve_continuous_lyapunov(F,-Q);past=z['past_times'];future=z['future_times'];observed=past[z['mask'][0].astype(bool)];y=z['observations'][0,z['mask'][0].astype(bool)].reshape(-1)
def state_mean(t):return expm(F*t)@m0
def state_cov(t):
 A=expm(F*t);return A@P0@A.T+stationary-A@stationary@A.T
def cross(t,s):return expm(F*(t-s))@state_cov(s) if t>=s else state_cov(t)@expm(F.T*(s-t))
obs_mean=np.concatenate([H@state_mean(t) for t in observed]);target_mean=np.concatenate([state_mean(t) for t in future]);obs_cov=np.block([[H@cross(t,s)@H.T+(Rnoise if i==j else 0.) for j,s in enumerate(observed)] for i,t in enumerate(observed)]);target_cov=np.block([[cross(t,s) for s in future] for t in future]);target_obs=np.block([[cross(t,s)@H.T for s in observed] for t in future]);gain=np.linalg.solve(obs_cov,target_obs.T).T;mean=(target_mean+gain@(y-obs_mean)).reshape(3,2);cov=target_cov-gain@target_obs.T;blocks=cov.reshape(3,2,3,2).transpose(0,2,1,3);mean_error=float(np.max(np.abs(mean-z['exact_mean'])));cov_error=float(np.max(np.abs(blocks-z['exact_covariance_blocks'])));assert mean_error<1e-10 and cov_error<1e-10,(mean_error,cov_error)
checks=[]
for field,reported in [('native_z',q['sampling'][0]),('ordinary_joint_z',q['sampling'][1])]:
 values=z[field];assert values.shape==(16384,3,2) and np.isfinite(values).all();centered=values-values.mean(0);actual=centered[:,0].T@centered[:,2]/(len(values)-1);contrast=np.var(values[:,2]-values[:,0],axis=0,ddof=1);assert np.max(np.abs(actual-reported['cross_time_covariance_0_2']))<1e-14 and np.max(np.abs(contrast-reported['temporal_difference_variance']))<1e-14;checks.append({'field':field,'reported_statistics_max_abs':max(float(np.max(np.abs(actual-reported['cross_time_covariance_0_2']))),float(np.max(np.abs(contrast-reported['temporal_difference_variance']))))})
result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'loaded_native_source_modules':origins,'independent_dense_conditioning_mean_max_abs':mean_error,'independent_dense_conditioning_covariance_max_abs':cov_error,'raw_statistics_checks':checks,'qualification_sha256':sha(R/'outputs/native_qualification_v1/qualification.json'),'raw_sampling_sha256':sha(raw),'checkpoint_sha256':sha(R/'outputs/native_qualification_v1/discarded_preflight.pt'),'source_sha256':sha(pathlib.Path(__file__)),'no_training_updates':True,'GPU_initialized':False,'claim_scope':'source origins plus independent dense Gaussian conditioning and retained-sample arithmetic; not nonlinear model-quality evidence'};out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='loaded_native_source_modules'}))
