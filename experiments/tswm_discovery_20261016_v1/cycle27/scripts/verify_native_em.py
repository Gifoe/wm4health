"""Independent NumPy physical dynamics, Kalman/RTS, ARX and score verification. No native import."""
import runtime_profile
import csv,hashlib,json,pathlib,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];SOURCE=ROOT/'outputs/native_em';OUT=ROOT/'outputs/native_em_evaluation_v2'
DT=.01;H=251
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def physical(x,u,dt=DT):
 a=u;b=x[:,0]-u;e=np.exp(-dt);e5=np.exp(-5*dt)
 next1=a+b*e
 next2=x[:,1]*e5
 for k,comb in enumerate([1,3,3,1]):next2+=5*comb*a**(3-k)*b**k*(np.exp(-k*dt)-e5)/(5-k)
 return np.column_stack((next1,next2))

def kalman_rts(parameters,y,u):
 n,p=y.shape;dim=4;c=parameters['C'][0];h=parameters['h'][0];r=parameters['Sig_w'][0,0];q=parameters['Sig_v']
 filtered=np.zeros((p,n,dim));fcov=np.zeros((p,n,dim,dim));prior=np.zeros_like(filtered);pcov=np.zeros_like(fcov);transitions=[]
 mean=np.repeat(parameters['mu_0'][None],n,axis=0);cov=np.repeat(parameters['Sig_0'][None],n,axis=0)
 for t in range(p):
  prior[t]=mean;pcov[t]=cov
  innovation=y[:,t]-h-mean@c;cp=cov@c;s=cp@c+r;assert np.min(s)>0
  gain=cp/s[:,None];mean=mean+gain*innovation[:,None]
  cov=cov-gain[:,:,None]*cp[:,None,:]
  filtered[t]=mean;fcov[t]=cov
  if t<p-1:
   full=np.eye(5)[None]+DT*(parameters['V_matrices'][0][None]+u[:,t,None,None]*parameters['V_matrices'][1][None])
   a=full[:,1:,1:];b=full[:,1:,0];transitions.append(a)
   mean=(a@mean[:,:,None])[:,:,0]+b;cov=a@cov@a.transpose(0,2,1)+q
 smoothed=filtered.copy();scov=fcov.copy()
 for t in range(p-2,-1,-1):
  numerator=fcov[t]@transitions[t].transpose(0,2,1)
  gain=np.linalg.solve(pcov[t+1].transpose(0,2,1),numerator.transpose(0,2,1)).transpose(0,2,1)
  smoothed[t]=filtered[t]+(gain@(smoothed[t+1]-prior[t+1])[:,:,None])[:,:,0]
  scov[t]=fcov[t]+gain@(scov[t+1]-pcov[t+1])@gain.transpose(0,2,1)
 return smoothed.transpose(1,0,2),scov.transpose(1,0,2,3)

def forecast(parameters,mean,cov,control):
 means=[];variances=[];c=parameters['C'][0];h=parameters['h'][0];r=parameters['Sig_w'][0,0]
 for t in range(control.shape[1]):
  full=np.eye(5)[None]+DT*(parameters['V_matrices'][0][None]+control[:,t,None,None]*parameters['V_matrices'][1][None])
  a=full[:,1:,1:];b=full[:,1:,0]
  mean=(a@mean[:,:,None])[:,:,0]+b;cov=a@cov@a.transpose(0,2,1)+parameters['Sig_v']
  means.append(mean@c+h);variances.append((cov@c)@c+r)
 return np.stack(means,axis=1),np.stack(variances,axis=1)

def main():
 started=time.monotonic();result=json.loads((OUT/'results.json').read_text());assert result['status']=='completed'
 receipt=json.loads((SOURCE/'fit_input_receipt.json').read_text());assert sha(ROOT/'PROTOCOL.md')==result['protocol_sha256']==receipt['protocol_sha256']
 assert sha(SOURCE/'fits_completed_before_evaluation.json')==result['training_receipt_sha256']
 maximum=dict(physical_abs=0.,native_mean_abs=0.,native_cov_abs=0.,forecast_mean_abs=0.,forecast_variance_abs=0.,score_abs=0.,score_scaled=0.,arx_parameters_abs=0.)
 counts=dict(data_hashes=0,weight_hashes=0,prediction_hashes=0,physical_states=0,posterior_histories=0,forecast_files=0,score_values=0)
 datasets={}
 for row in receipt['data']:
  path=SOURCE/row['path'];assert sha(path)==row['sha256'];counts['data_hashes']+=1;d=dict(np.load(path));datasets[row['name']]=d
  x=d['x'];y=d['y'][:,:,0];u=d['u'][:,:,1];rng=np.random.RandomState(row['seed']);dwell=row['dwell']
  assert np.array_equal(d['u'][:,:,0],np.ones_like(u))
  for j in range(len(x)):
   initial=rng.randn(2);assert np.array_equal(initial,x[j,0]);assert y[j,0]==x[j,0,1]+.1*rng.randn()
   for t in range(500):
    if t%dwell==0:control=np.sqrt(5)*rng.randn()
    assert control==u[j,t];assert y[j,t+1]==x[j,t+1,1]+.1*rng.randn()
  current=x[:,0].copy()
  for t in range(500):
   current=physical(current,u[:,t]);diff=float(np.max(np.abs(current-x[:,t+1])));maximum['physical_abs']=max(maximum['physical_abs'],diff)
   assert np.allclose(current,x[:,t+1],atol=1e-4,rtol=5e-6);counts['physical_states']+=int(current.size)
 train=datasets['train_continuation'];assert arrsha(train['y'][:,:250])==receipt['y_sha256'];assert arrsha(train['u'][:,:249])==receipt['u_unscaled_sha256']
 parameters={}
 for fit in result['fits']:
  for row in fit['checkpoints']:
   path=SOURCE/f"seed{fit['seed']}"/row['path'];assert sha(path)==row['sha256'];counts['weight_hashes']+=1
  for label,w in [('iteration100',fit['checkpoint100']),('trainbest_final',fit['best'])]:
   parameters[(fit['seed'],label)]=dict(np.load(SOURCE/f"seed{fit['seed']}"/w['path']))
 for row in result['predictions']:assert sha(OUT/row['path'])==row['sha256'];counts['prediction_hashes']+=1
 # ARX via augmented least squares,rather than the producer's normal equations.
 train_y=train['y'][:,:250,0];train_u=train['u'][:,:249,1]
 design=[];target=[];time_index=[]
 for j in range(len(train_y)):
  for t in range(9,249):
   design.append([train_y[j,t-l] for l in range(10)]+[train_u[j,t-l] for l in range(10)])
   target.append(train_y[j,t+1]);time_index.append(t)
 x=np.array(design);target=np.array(target);early=np.array(time_index)<199
 def solve(features,response,lam):
  mean=features.mean(0);scale=features.std(0);scale=np.maximum(scale,1e-12)
  matrix=np.column_stack((np.ones(len(features)),(features-mean)/scale));reg=np.eye(21)*np.sqrt(lam*len(features));reg[0,0]=0
  coef=np.linalg.lstsq(np.vstack((matrix,reg)),np.r_[response,np.zeros(21)],rcond=None)[0]
  return mean,scale,coef
 choices=[]
 for lam in [1e-4,.01,1,100]:
  mean,scale,coef=solve(x[early],target[early],lam)
  pred=np.column_stack((np.ones((~early).sum()),(x[~early]-mean)/scale))@coef;choices.append((float(np.mean((pred-target[~early])**2)),lam))
 selected=min(choices)[1];arx=dict(np.load(OUT/'arx.npz'));assert float(arx['lam'])==selected
 mean,scale,coef=solve(x,target,selected)
 for key,value in [('mean',mean),('scale',scale),('coefficient',coef)]:
  difference=float(np.max(np.abs(arx[key]-value)));maximum['arx_parameters_abs']=max(maximum['arx_parameters_abs'],difference);assert np.allclose(arx[key],value,atol=1e-8,rtol=1e-8)
 with (OUT/'rows.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 assert len(rows)==150==result['n_rows']
 posterior={}
 for row in rows:
  name=row['dataset'];data=datasets[name];p=int(row['prefix']);seed=int(row['seed']);method=row['method'];checkpoint=row['checkpoint'];n=len(data['y'])
  if seed<0:path=OUT/f'prediction_{name}_p{p}_{method}.npz'
  else:path=OUT/f'prediction_{name}_p{p}_seed{seed}_{checkpoint}_{method}.npz'
  saved=dict(np.load(path));observed=data['y'][:,p:p+H,0];clean=data['x'][:,p:p+H,1];pred=saved['mean']
  if seed>=0:
   key=(seed,checkpoint,name,p);params=parameters[(seed,checkpoint)]
   if key not in posterior:
    posterior[key]=kalman_rts(params,data['y'][:,:p,0],data['u'][:,:p-1,1]);counts['posterior_histories']+=1
   mu,cov=posterior[key];index=-1 if method=='filtered_prefix' else 0;start=p-1 if index==-1 else 0
   for key2,value,category in [('latent_initial_mean',mu[:,index],'native_mean_abs'),('latent_initial_covariance',cov[:,index],'native_cov_abs')]:
    diff=float(np.max(np.abs(saved[key2]-value)));maximum[category]=max(maximum[category],diff)
    assert np.allclose(saved[key2],value,atol=1e-8,rtol=1e-7),(path.name,key2,diff)
   replay,replayvar=forecast(params,mu[:,index].copy(),cov[:,index].copy(),data['u'][:,start:p-1+H,1]);offset=p-1-start
   for value,original,category in [(replay[:,offset:],pred,'forecast_mean_abs'),(replayvar[:,offset:],saved['variance'],'forecast_variance_abs')]:
    diff=float(np.max(np.abs(value-original)));maximum[category]=max(maximum[category],diff)
    assert np.allclose(value,original,atol=2e-7,rtol=1e-7),(path.name,category,diff)
  elif method=='oracle_true_state_physics':
   state=data['x'][:,p-1].copy();output=[]
   for t in range(H):state=physical(state,data['u'][:,p-1+t,1]);output.append(state[:,1].copy())
   assert np.allclose(pred,np.stack(output,axis=1),atol=1e-10,rtol=1e-10)
  elif method=='persistence':assert np.array_equal(pred,np.repeat(data['y'][:,p-1,0,None],H,axis=1))
  elif method=='arx':
   history=[data['y'][:,t,0].copy() for t in range(p)];output=[]
   for t in range(p-1,p+H-1):
    raw=np.column_stack([history[t-l] for l in range(10)]+[data['u'][:,t-l,1] for l in range(10)])
    z=np.column_stack((np.ones(n),(raw-arx['mean'])/arx['scale']))@arx['coefficient'];history.append(z);output.append(z)
   assert np.allclose(pred,np.stack(output,axis=1),atol=1e-8,rtol=1e-8)
  else:raise ValueError(method)
  error=(pred-observed).ravel();ce=(pred-clean).ravel()
  values=dict(mse_observed=float(error@error/error.size),mse_clean=float(ce@ce/ce.size),relative_l2_clean=float(np.sqrt((ce@ce)/np.sum(clean**2))),finite_fraction=float(np.isfinite(pred).sum()/pred.size))
  if 'variance' in saved:
   var=saved['variance'];assert np.min(var)>0
   values.update(marginal_nll=float(np.sum(np.log(var)+np.log(2*np.pi)+(pred-observed)**2/var)/(2*var.size)),coverage95=float(np.count_nonzero((pred-observed)**2<=1.959963984540054**2*var)/var.size),mean_predictive_std=float(np.sqrt(var).sum()/var.size))
  for key,value in values.items():
   actual=float(row[key]);diff=abs(value-actual);maximum['score_abs']=max(maximum['score_abs'],diff);maximum['score_scaled']=max(maximum['score_scaled'],diff/(1+abs(actual)))
   assert np.isclose(value,actual,atol=1e-9,rtol=1e-12),(path.name,key,value,actual);counts['score_values']+=1
  counts['forecast_files']+=1
 result=dict(status='passed',maxima=maximum,counts=counts,runtime_s=time.monotonic()-started,results_sha256=sha(OUT/'results.json'),verifier_sha256=sha(pathlib.Path(__file__)),
    scope='All datasets,declared checkpoints,posterior initial moments,forecast arrays,ARX selection and scores;no native-module import',
    tolerances=dict(posterior_atol=1e-8,posterior_rtol=1e-7,forecast_atol=2e-7,forecast_rtol=1e-7,score_atol=1e-9,score_rtol=1e-12))
 (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
