"""Pinned native algorithm, source settings, train-only convergence and independent evaluation."""
import native_control_runtime
import csv,hashlib,json,pathlib,platform,time,traceback
import numpy as np
import scipy
from scipy.integrate import odeint
from scipy.linalg import eig,expm
import numba
from actuated_Koopman_EM_4 import ActuatedKoopmanEM

ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/native_em'
DT=.01;PREFIXES=[25,100,250];HORIZON=251;FIT_SEEDS=[20262711,20262712,20262713]
PARAMETERS=['V_matrices','C','h','Sig_v','Sig_w','mu_0','Sig_0']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(path,x):path.write_text(json.dumps(x,indent=2),encoding='utf-8')
def progress(**x):print(json.dumps(x),flush=True)

def generators():
 v0=np.zeros((5,5));v1=np.zeros((5,5));v0[1,1]=-1;v0[2,2]=-5;v0[2,4]=5;v0[3,3]=-2;v0[4,4]=-3
 v1[1,0]=1;v1[3,1]=2;v1[4,3]=3
 return v0,v1

def make_data(seed,n,dwell):
 rng=np.random.RandomState(seed);x=np.zeros((n,501,2));y=np.zeros((n,501,1));u=np.zeros((n,500,2));u[:,:,0]=1
 for j in range(n):
  state=rng.randn(2);x[j,0]=state;y[j,0,0]=state[1]+.1*rng.randn()
  for begin in range(0,500,dwell):
   count=min(dwell,500-begin);control=np.sqrt(5)*rng.randn()
   solution=odeint(lambda z,t:(-z[0]+control,5*(z[0]**3-z[1])),state,DT*np.arange(count+1))
   u[j,begin:begin+count,1]=control
   for local in range(count):
    state=solution[local+1];x[j,begin+local+1]=state;y[j,begin+local+1,0]=state[1]+.1*rng.randn()
 return dict(x=x,y=y,u=u,seed=np.array(seed),dwell=np.array(dwell))

def exact_physics(initial,controls):
 v0,v1=generators();n,h=controls.shape;result=np.empty((n,h))
 for j in range(n):
  a,b=initial[j];state=np.array([1,a,b,a*a,a**3]);previous=None;operator=None
  for t,control in enumerate(controls[j]):
   if previous is None or control!=previous:operator=expm(DT*(v0+control*v1));previous=control
   state=operator@state;result[j,t]=state[2]
 return result

def reflect_eigs(a):
 values,left,right=eig(a,left=True,right=True);w=np.diag(np.conj(right).T@left);left=left/w
 mask=np.abs(values)>1;values[mask]=1/values[mask]
 return np.real_if_close((right*values)@np.conj(left).T,tol=1000).astype(np.float64)

def initialize(seed,y,u):
 rng=np.random.RandomState(seed);a0=reflect_eigs(rng.randn(4,4)/2);a1=reflect_eigs(rng.randn(5,5)/np.sqrt(5))
 v=np.zeros((2,5,5));v[0,1:,1:]=.5*(a0-np.eye(4))/DT;v[1,1:,:]=.5*a1[1:,:]/(DT*np.max(np.abs(u[:,:,1])))
 c=np.array([[1.,0,0,0]])
 return ActuatedKoopmanEM(y,DT*u,v,c,np.zeros(1),.1*np.eye(4),.1*np.eye(1),np.zeros(4),np.eye(4),compute_log_likelihood=True)

def save_weights(model,path):
 data={name:np.asarray(getattr(model,name)) for name in PARAMETERS}
 data.update(iteration=np.array(model.EM_iter),likelihood=np.array(model.L))
 for name,value in data.items():assert np.isfinite(value).all(),name
 for name in ['Sig_v','Sig_w','Sig_0']:assert np.linalg.eigvalsh(data[name]).min()>-1e-10,(name,data[name])
 assert np.max(np.abs(data['V_matrices'][:,0]))==0
 np.savez_compressed(path,**data)
 return dict(path=path.name,sha256=digest(path),iteration=model.EM_iter,likelihood=float(model.L))

def fit(seed,y,u):
 started=time.monotonic();folder=OUT/f'seed{seed}';folder.mkdir();trace=[];checkpoints=[]
 model=initialize(seed,y,u);checkpoints.append(save_weights(model,folder/'weights_init.npz'))
 best=None;best_likelihood=-np.inf;previous=None;delta=np.inf;stopping='iteration_cap'
 for iteration in range(1,1001):
  check=iteration==1 or iteration%10==0
  model.run_EM_step(explicit_time_step=True,optimize_IC=True,optimize_observation_map=False,optimize_process_noise=True,compute_log_likelihood=check)
  if check:
   likelihood=float(model.L);assert np.isfinite(likelihood)
   if previous is not None:delta=(likelihood-previous[1])/(iteration-previous[0])
   previous=(iteration,likelihood)
   record=dict(seed=seed,iteration=iteration,likelihood=likelihood,average_ll_gain=float(delta) if np.isfinite(delta) else None,elapsed_s=time.monotonic()-started)
   trace.append(record);checkpoint=save_weights(model,folder/f'weights_iter{iteration}.npz');checkpoints.append(checkpoint)
   if likelihood>best_likelihood:best=checkpoint;best_likelihood=likelihood
   write(folder/'trace.json',trace);progress(phase='fit',**record)
   if iteration>=100 and delta<=.01:stopping='train_likelihood_tolerance';break
 final=dict(seed=seed,stopping=stopping,total_iterations=iteration,final_average_ll_gain=float(delta),best=best,
    checkpoint100=next(c for c in checkpoints if c['iteration']==100),runtime_s=time.monotonic()-started,checkpoints=checkpoints)
 write(folder/'fit.json',final);return final

def load_model(path):
 # Native no-data constructor is used only as a parameter container;its saved default is never fitted/scored.
 values=np.load(path);model=ActuatedKoopmanEM()
 for name in PARAMETERS:setattr(model,name,values[name].copy())
 model.z_dim=5;model.y_dim=1;model.u_dim=2;model.EM_iter=int(values['iteration']);model.L=float(values['likelihood'])
 return model

def fit_arx(data):
 y=data['y'][:,:250,0];u=data['u'][:,:249,1]
 xs=[];ys=[];times=[]
 for t in range(9,249):
  xs.append(np.concatenate((y[:,t-np.arange(10)],u[:,t-np.arange(10)]),axis=1));ys.append(y[:,t+1]);times.extend([t]*len(y))
 x=np.concatenate(xs);target=np.concatenate(ys);times=np.array(times);early=times<199
 def solve(x,z,lam):
  mean=x.mean(0);scale=x.std(0);scale=np.maximum(scale,1e-12);design=np.column_stack((np.ones(len(x)),(x-mean)/scale))
  penalty=np.eye(design.shape[1])*lam;penalty[0,0]=0
  coefficient=np.linalg.solve(design.T@design/len(x)+penalty,design.T@z/len(x))
  return dict(mean=mean,scale=scale,coefficient=coefficient)
 choices=[]
 for lam in [1e-4,.01,1,100]:
  m=solve(x[early],target[early],lam);design=np.column_stack((np.ones((~early).sum()),(x[~early]-m['mean'])/m['scale']))
  loss=float(np.mean((design@m['coefficient']-target[~early])**2));choices.append(dict(lam=lam,validation_mse=loss))
 lam=min(choices,key=lambda c:c['validation_mse'])['lam'];model=solve(x,target,lam)
 np.savez_compressed(OUT/'arx.npz',**model,lam=np.array(lam));write(OUT/'arx_selection.json',dict(choices=choices,selected_lambda=lam,selection='Training-prefix validation targets200..249 only'))
 return model

def arx_predict(model,data,prefix):
 y=data['y'][:,:,0];u=data['u'][:,:,1];current=y[:,:prefix].copy();forecasts=[]
 for t in range(prefix-1,prefix+HORIZON-1):
  x=np.concatenate((current[:,t-np.arange(10)],u[:,t-np.arange(10)]),axis=1)
  design=np.column_stack((np.ones(len(y)),(x-model['mean'])/model['scale']));next_y=design@model['coefficient']
  forecasts.append(next_y);current=np.column_stack((current,next_y))
 return np.stack(forecasts,axis=1)

def scores(mean,var,data,prefix):
 target=data['y'][:,prefix:prefix+HORIZON,0];clean=data['x'][:,prefix:prefix+HORIZON,1]
 error=mean-target;ce=mean-clean
 result=dict(mse_observed=float(np.mean(error**2)),mse_clean=float(np.mean(ce**2)),relative_l2_clean=float(np.linalg.norm(ce)/np.linalg.norm(clean)),finite_fraction=float(np.isfinite(mean).mean()))
 if var is not None:
  assert np.isfinite(var).all() and np.min(var)>0
  result.update(marginal_nll=float(np.mean(.5*(np.log(2*np.pi*var)+error**2/var))),coverage95=float(np.mean(np.abs(error)<=1.959963984540054*np.sqrt(var))),mean_predictive_std=float(np.mean(np.sqrt(var))))
 return result

def main():
 assert not OUT.exists(),'Refuse overwriting scientific output'
 OUT.mkdir(parents=True);start=time.monotonic();data_files=[];datasets={}
 for name,seed,n,dwell in [('train_continuation',20262701,50,50),('independent_dwell50',20262702,64,50),('independent_dwell5',20262703,64,5),('independent_dwell100',20262704,64,100)]:
  data=make_data(seed,n,dwell);datasets[name]=data;path=OUT/f'data_{name}.npz';np.savez_compressed(path,**data)
  data_files.append(dict(name=name,path=path.name,sha256=digest(path),n_trajectories=n,dwell=dwell,seed=seed));progress(phase='data',name=name,n=n,dwell=dwell)
 train=datasets['train_continuation'];y=np.ascontiguousarray(train['y'][:,:250]);u=np.ascontiguousarray(train['u'][:,:249])
 write(OUT/'fit_input_receipt.json',dict(y_shape=list(y.shape),u_shape=list(u.shape),y_sha256=array_digest(y),u_unscaled_sha256=array_digest(u),maximum_observation_index=249,maximum_input_index=248,hidden_state_used_for_fit=False,data=data_files,protocol_sha256=digest(ROOT/'PROTOCOL.md')))
 fits=[];failures=[]
 for seed in FIT_SEEDS:
  try:fits.append(fit(seed,y,u))
  except Exception as e:
   failure=dict(seed=seed,error=repr(e),traceback=traceback.format_exc());failures.append(failure);write(OUT/f'failure_seed{seed}.json',failure);progress(phase='fit_failed',seed=seed,error=repr(e))
 if failures:
  write(OUT/'fit_failure_summary.json',dict(fits=fits,failures=failures));raise RuntimeError('Declared native fit failed;retain results and diagnose before scoring')
 write(OUT/'fits_completed_before_evaluation.json',dict(fits=fits,elapsed_s=time.monotonic()-start))
 arx=fit_arx(train);records=[];predictions=[];data_gates=[]
 for name,data in datasets.items():
  exact=exact_physics(data['x'][:,0],data['u'][:,:,1]);actual=data['x'][:,1:,1];difference=np.abs(exact-actual)
  assert np.allclose(exact,actual,atol=1e-4,rtol=5e-6)
  data_gates.append(dict(dataset=name,max_physical_output_abs_error=float(difference.max()),rms_physical_output_error=float(np.sqrt(np.mean(difference**2)))))
  prefixes=[250] if name=='train_continuation' else PREFIXES
  for prefix in prefixes:
   base=[('persistence',np.repeat(data['y'][:,prefix-1,0,None],HORIZON,axis=1),None),('arx',arx_predict(arx,data,prefix),None)]
   oracle=exact_physics(data['x'][:,prefix-1],data['u'][:,prefix-1:prefix-1+HORIZON,1]);base.append(('oracle_true_state_physics',oracle,np.full_like(oracle,.01)))
   for method,mean,var in base:
    path=OUT/f'prediction_{name}_p{prefix}_{method}.npz';parts=dict(mean=mean,prefix=np.array(prefix))
    if var is not None:parts['variance']=var
    np.savez_compressed(path,**parts);predictions.append(dict(path=path.name,sha256=digest(path)))
    records.append(dict(dataset=name,prefix=prefix,method=method,seed=-1,checkpoint='none',**scores(mean,var,data,prefix)))
 for fit_result in fits:
  seed=fit_result['seed']
  for checkpoint,weight in [('iteration100',fit_result['checkpoint100']),('trainbest_final',fit_result['best'])]:
   model=load_model(OUT/f'seed{seed}'/weight['path'])
   for name,data in datasets.items():
    prefixes=[250] if name=='train_continuation' else PREFIXES
    for prefix in prefixes:
     history=np.ascontiguousarray(data['y'][:,:prefix]);past_inputs=np.ascontiguousarray(DT*data['u'][:,:prefix-1])
     means,covs,_=model.infer_latent_states(history,past_inputs,explicit_time_step=True)
     for convention,start_time,index in [('filtered_prefix',prefix-1,-1),('native_smoothed_initial',0,0)]:
      plan=np.ascontiguousarray(DT*data['u'][:,start_time:prefix-1+HORIZON])
      pred,predcov,_,_=model.predict_dynamics(means[:,index].copy(),covs[:,index].copy(),plan,explicit_time_step=True)
      offset=prefix-1-start_time;mean=pred[:,offset:,0];variance=predcov[:,offset:,0,0];assert mean.shape==(len(history),HORIZON)
      path=OUT/f'prediction_{name}_p{prefix}_seed{seed}_{checkpoint}_{convention}.npz'
      np.savez_compressed(path,mean=mean,variance=variance,latent_initial_mean=means[:,index],latent_initial_covariance=covs[:,index],prefix=np.array(prefix),start_time=np.array(start_time),weights_relative=np.array(f'seed{seed}/'+weight['path']))
      predictions.append(dict(path=path.name,sha256=digest(path)))
      records.append(dict(dataset=name,prefix=prefix,method=convention,seed=seed,checkpoint=checkpoint,weight_iteration=weight['iteration'],**scores(mean,variance,data,prefix)))
    progress(phase='evaluate',seed=seed,checkpoint=checkpoint,dataset=name,rows=len(records))
 write(OUT/'data_gate.json',data_gates)
 fieldnames=list(dict.fromkeys(k for row in records for k in row))
 with (OUT/'rows.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(records)
 result=dict(status='completed',scope='Native algorithm/source settings positive control plus independent causal-prefix evaluation;no new method',
   n_rows=len(records),fit_seeds=FIT_SEEDS,fits=fits,data=data_files,predictions=predictions,runtime_s=time.monotonic()-start,
   environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,numba_threads=numba.get_num_threads(),device='CPU'),
   source_revision='ef830a6d4a6cff699492d54230fc2ad1729cab20',source_manifest_sha256=digest(ROOT.parent/'cycle26/outputs/koopman_source_manifest.json'),
   script_sha256=digest(pathlib.Path(__file__)),protocol_sha256=digest(ROOT/'PROTOCOL.md'),accepted_final_directions=0)
 write(OUT/'results.json',result);progress(phase='completed',rows=len(records),runtime_s=result['runtime_s'])

if __name__=='__main__':main()
