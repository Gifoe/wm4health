"""Independent distribution metrics, model selection and validation replay."""
from models65 import *
from checkpoint_io65_v1 import load_checkpoint
from scipy.stats import wasserstein_distance
import scipy.linalg as sla,time
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
e=read(OUT/'EVALUATION65_V1.json');f=read(OUT/'FITS65_V1.json');o=read(OUT/'ORACLE65_V1.json');assert e['generated_laws']==240;source_check();assert sha(OUT/'FITS65_V1.json')==e['fit_receipt_sha256'] and sha(OUT/'ORACLE65_V1.json')==e['oracle_receipt_sha256'];assert sha(o['artifact'])==o['artifact_sha256'];raw=np.load(o['artifact']);D=raw['directions'];device=f['device'];errors={};counts={};start=time.perf_counter();data={}
def check(label,x,y,tol=3e-6):
 err=float(np.max(np.abs(np.asarray(x)-np.asarray(y))/(1+np.abs(y))));errors[label]=max(errors.get(label,0.),err);assert err<=tol,(label,err)
for corpus in f['corpora']:
 assert sha(corpus['path'])==corpus['sha256'];z=np.load(corpus['path']);data[corpus['key']]={k:z[k] for k in z.files};assert z['train'].shape==(16384,2,8) and z['validation'].shape==(4096,2,8) and z['observational_test'].shape==(8192,2,8)
 for arr in data[corpus['key']].values():assert np.isfinite(arr).all()
counts['raw_corpus_hashes']=len(data);counts['checkpoint_hashes']=0;counts['selected_checkpoints']=0;counts['native_validation_views']=0;counts['MDN_validation_checkpoints']=0;counts['trained_CPU_GPU_output_checks']=0
for fit in f['fits']:
 corpus=data[fit['key']];selected=fit['selected_checkpoint'];selection='mean' if fit['model'].startswith('CSDI') else 'validation_NLL';chosen=min(fit['checkpoints'],key=lambda z:z['validation']['mean'] if selection=='mean' else z['validation_NLL']);assert selected['sha256']==chosen['sha256'];counts['selected_checkpoints']+=1
 for cp in fit['checkpoints']:
  assert sha(cp['path'])==cp['sha256'];counts['checkpoint_hashes']+=1;z=load_checkpoint(cp['path'],map_location=device)
  if fit['model'].startswith('CSDI'):
   mean=corpus['train'].mean((0,2),keepdims=True);std=corpus['train'].std((0,2),keepdims=True);check('train_only_mean',z['normalization_mean'],mean);check('train_only_std',z['normalization_std'],std);model=csdi(device);model.load_state_dict(z['state_dict']);model.eval();x=torch.tensor((corpus['validation'][:256]-mean)/std,device=device);values={}
   with torch.no_grad(),torch.random.fork_rng(devices=[0] if device=='cuda' else []):
    for kind in ['block','prefix']:
     torch.manual_seed(650003);m=torch.zeros_like(x)
     if kind=='block':m[:,1]=1
     else:
      cuts=np.random.default_rng(650004).integers(0,H,size=len(x))
      for j,t in enumerate(cuts):m[j,0,:t]=1;m[j,1,:t+1]=1
     ss=side(model,x,m);losses=[]
     for t in range(50):
      noise=torch.randn_like(x);alpha=model.alpha_torch[torch.full((len(x),),t,dtype=torch.long,device=device)];noisy=alpha.sqrt()*x+(1-alpha).sqrt()*noise;inp=torch.stack([m*x,(1-m)*noisy],1);pred=model.diffmodel(inp,ss,torch.full((len(x),),t,dtype=torch.long,device=device));losses.append(((noise-pred).square()*(1-m)).sum()/(1-m).sum())
     values[kind]=float(torch.stack(losses).mean());check('native_validation_DSM_'+kind,values[kind],cp['validation'][kind],2e-5);counts['native_validation_views']+=1
   check('validation_selection_mean',sum(values.values())/2,cp['validation']['mean'],2e-5)
   if cp['sha256']==selected['sha256']:
    cpu=csdi('cpu');cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()});cpu.eval();xx=torch.linspace(-1,1,8*2*H).reshape(8,2,H);mm=torch.zeros_like(xx);mm[:,1]=1;tt=torch.arange(8)%50
    with torch.no_grad():a=cpu.diffmodel(torch.stack([mm*xx,(1-mm)*xx],1),side(cpu,xx,mm),tt);b=model.diffmodel(torch.stack([mm*xx,(1-mm)*xx],1).to(device),side(model,xx.to(device),mm.to(device)),tt.to(device)).cpu()
    check('trained_CPU_GPU_output',a.numpy(),b.numpy(),2e-4);counts['trained_CPU_GPU_output_checks']+=1;del cpu
  else:
   model=MDN().to(device);model.load_state_dict(z['state_dict']);model.eval();vx,vy=transitions(corpus['validation'])
   with torch.no_grad():val=float(model.nll(torch.tensor(vx,device=device),torch.tensor(vy,device=device)))
   check('MDN_validation_NLL',val,cp['validation_NLL'],2e-5);counts['MDN_validation_checkpoints']+=1
  del model,z
counts['sample_law_hashes']=0;counts['target_metric_views']=0;counts['projected_W1_values']=0
for index,row in enumerate(e['laws']):
 assert sha(row['artifact'])==row['artifact_sha256'];x=np.load(row['artifact'])['x'].astype('float64');assert x.shape==(512,H) and np.isfinite(x).all();counts['sample_law_hashes']+=1;proj=x@D.T
 for target,record in row['scores'].items():
  ref=raw[row['mechanism']+'_'+row['query']+'_'+target];rp=ref@D.T;wd=np.array([wasserstein_distance(proj[:,j],rp[:,j]) for j in range(len(D))]);check('exact_empirical_projected_W1',wd.mean(),record['finite_sliced_W1'],1e-10);check('normalized_projected_W1',wd.mean()/rp.std(0).mean(),record['normalized_finite_sliced_W1'],1e-10);check('terminal_marginal_W1',wasserstein_distance(x[:,-1],ref[:,-1]),record['terminal_marginal_W1'],1e-10);check('path_mean_MSE',np.square(x.mean(0)-ref.mean(0)).mean(),record['path_mean_MSE']);check('variance_trace_ratio',np.var(x,axis=0).sum()/np.var(ref,axis=0).sum(),record['variance_trace_ratio']);tail=float(np.count_nonzero(np.abs(x[:,-1])>2)/len(x));rtail=float(np.count_nonzero(np.abs(ref[:,-1])>2)/len(ref));check('terminal_tail',tail,record['terminal_tail_probability'],1e-12);check('reference_tail',rtail,record['reference_tail_probability'],1e-12);check('terminal_tail_abs_error',abs(tail-rtail),record['terminal_tail_abs_error'],1e-12);counts['target_metric_views']+=1;counts['projected_W1_values']+=len(D)
 if row['method']=='causal_linear_AR':
  corpus=data[row['mechanism']+'_s'+str(row['training_seed'])];XX,yy=transitions(corpus['train']);A=np.c_[XX,np.ones(len(XX))];coef=sla.lstsq(A,yy,lapack_driver='gelsy')[0];check('AR_QR_coefficients',coef,row['ar_coefficients']);check('AR_QR_residual_sd',np.sqrt(np.mean((yy-A@coef)**2)),row['noise_sd'])
 if index%20==0:state('replay_state65_v1.json',dict(status='INDEPENDENT_SAVED_LAW_REPLAY',completed=index+1,total=len(e['laws'])));print(json.dumps({'replayed_laws':index+1}),flush=True)
assert counts==dict(raw_corpus_hashes=6,checkpoint_hashes=36,selected_checkpoints=18,native_validation_views=48,MDN_validation_checkpoints=12,trained_CPU_GPU_output_checks=12,sample_law_hashes=240,target_metric_views=480,projected_W1_values=65280),counts
dump(OUT/'INDEPENDENT_REPLAY65_V1.json',dict(utc=utc(),status='PASS',counts=counts,max_scaled_errors=errors,runtime_seconds=time.perf_counter()-start,evaluation_sha256=sha(OUT/'EVALUATION65_V1.json'),scope='Native learned-checkpoint validation/selection replay and CPU/GPU output checks; independent SciPy CDF integration for every projected empirical W1 and every raw-law metric. Not an independent reproduction of original CSDI benchmarks or novelty proof.'));state('replay_state65_v1.json',dict(status='PASS',counts=counts));print(json.dumps(dict(status='PASS',counts=counts,max_scaled_error=max(errors.values()))))
