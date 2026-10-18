from native87_v1 import *
from physics87_v1 import *
from ordinary87_v1 import identify,predict as poly_predict
import time,gc
class NativeField:
 def __init__(self,model,context):
  self.model=model;b,t,n,_=context.shape;tt=np.broadcast_to((np.arange(n)*.05)[None,None,:,None],(b,t,n,1)).copy().astype(np.float32)
  with torch.no_grad():self.enc,self.mask,self.concept=model.trajectory_encoding(torch.as_tensor(pad(context),device='cuda'),torch.as_tensor(tt,device='cuda'),torch.ones((b,t,n,1),dtype=torch.bool,device='cuda'))
 def __call__(self,x):
  single=x.ndim==2
  if single:x=x[:,None,:]
  q=torch.nn.functional.pad(x,(0,1));q=self.model.spatial_norm.normalization_map(q,self.concept._states_norm_stats)
  with torch.no_grad():r=self.model.function_decoding(q,self.mask,self.enc,self.concept);y=self.model.get_prediction_for_eval(r)[...,:2]
  return y[:,0] if single else y
def native_rollout(fn,initial,steps=200,dt=.05,substeps=4):
 x=torch.as_tensor(initial,device='cuda',dtype=torch.float32);active=torch.ones(len(x),dtype=torch.bool,device='cuda');parts=[x.clone()];h=dt/substeps
 with torch.no_grad():
  for _ in range(steps):
   for __ in range(substeps):
    k1=fn(x);k2=fn(x+h*k1/2);k3=fn(x+h*k2/2);k4=fn(x+h*k3);new=x+h*(k1+2*k2+2*k3+k4)/6
    active=active&torch.isfinite(new).all(-1)&(abs(new)<1000).all(-1);x=torch.where(active[:,None],new,torch.zeros_like(new))
   parts.append(torch.where(active[:,None],x,torch.full_like(x,float('nan'))))
 return torch.stack(parts,dim=1).cpu().numpy()
def ordinary_rollout(coef,degree,initial,steps=200,dt=.05,substeps=4):
 x=initial.astype(np.float64).copy();active=np.ones(len(x),bool);parts=[x.copy()];h=dt/substeps
 with np.errstate(over='ignore',invalid='ignore'):
  for _ in range(steps):
   for __ in range(substeps):
    k1=poly_predict(coef,x,degree);k2=poly_predict(coef,x+h*k1/2,degree);k3=poly_predict(coef,x+h*k2/2,degree);k4=poly_predict(coef,x+h*k3,degree);new=x+h*(k1+2*k2+2*k3+k4)/6
    active=active&np.isfinite(new).all(-1)&(abs(new)<1000).all(-1);x=np.where(active[:,None],new,0.)
   parts.append(np.where(active[:,None],x,np.nan))
 return np.stack(parts,axis=1)
def metrics(pred,truth,h):
 n=round(h/.05)+1;p=pred[:,1:n];y=truth[:,1:n];ok=np.isfinite(p).all(axis=(1,2));mse=np.full(len(p),np.nan);nr=np.full(len(p),np.nan)
 mse[ok]=np.mean((p[ok]-y[ok])**2,axis=(1,2));nr[ok]=np.sqrt(mse[ok])/np.maximum(np.sqrt(np.mean(y[ok]**2,axis=(1,2))),.1)
 def avg(x):return float(np.mean(x)) if len(x) else None
 return dict(horizon=h,n=len(p),complete_count=int(ok.sum()),failed_count=int((~ok).sum()),population_mse=avg(mse) if ok.all() else None,population_mean_nrmse=avg(nr) if ok.all() else None,population_median_nrmse=float(np.median(nr)) if ok.all() else None,finite_only_mse=avg(mse[ok]),finite_only_mean_nrmse=avg(nr[ok]),failure_penalized_mean_nrmse=float(np.mean(np.where(ok,nr,1000.))))
def cohorts():
 rows=[];folder=RAW/'evaluation_v1';folder.mkdir(exist_ok=False)
 for fi,family in enumerate(FAMILIES):
  rng=np.random.default_rng(np.random.SeedSequence([2026875,fi]));p=parameters(family,1024,rng);ics=rng.normal(0,.8,(1024,9,2));full=trajectories(family,p,ics);accept=(abs(full)<2.5).all(axis=(1,2,3));ids=np.flatnonzero(accept)[:128];assert len(ids)==128
  pairs=ics[ids,:2].copy();perm=rng.permutation(128);newic=rng.normal(0,.8,(1024,1,2));noise=rng.normal(0,.01,(128,2,33,2)).astype(np.float32)
  pop=folder/f'{family}_pool_v1.npz';save(pop,parameters=p,initial=ics,accepted=accept,retained_ids=ids,context_pair_permutation=perm,new_initial=newic);rows.append(dict(kind='pool',family=family,array_path=str(pop),array_sha256=sha(pop),accepted_count=int(accept.sum()),retained_vs_unselected_overlap=int(np.isin(ids,np.arange(128)).sum())))
  for name in ['retained','crossed','unselected']:
   pick=np.arange(128) if name=='unselected' else ids
   init=ics[pick,:2] if name!='crossed' else pairs[perm]
   clean=trajectories(family,p[pick],init,steps=32);observed=clean+noise
   truth=trajectories(family,p[pick],newic[pick])[:,0];loc_context=clean.reshape(128,-1,2);loc_future=truth[:,np.linspace(0,200,17,dtype=int)]
   path=folder/f'{family}_{name}_v1.npz';save(path,mechanism_ids=pick,parameters=p[pick],context_initial=init,context_clean=clean,context=observed,noise=noise,query_initial=newic[pick,0],truth=truth,context_locations=loc_context,context_field=field(family,p[pick],loc_context),future_locations=loc_future,future_field=field(family,p[pick],loc_future))
   rows.append(dict(kind='cohort',family=family,cohort=name,array_path=str(path),array_sha256=sha(path)))
 write(O/'EVAL_MANIFEST87_V1.json',dict(utc=utc(),seed=2026875,rows=rows,selection_sha256=sha(O/'SELECTION87_V1.json'),ordinary_selection_sha256=sha(O/'ORDINARY_SELECTION87_V1.json'),protocol_sha256=sha(C/'PROTOCOL87_V1.md'),details_sha256=sha(C/'EVALUATION_DETAILS87_V1.md')))
 return folder
def main():
 started=time.perf_counter();selection=read(O/'SELECTION87_V1.json');assert len(selection['fits'])==9 and not selection['eval_opened'];ordinary=read(O/'ORDINARY_SELECTION87_V1.json');assert not ordinary['eval_opened']
 for fit in selection['fits']:assert sha(fit['selected']['checkpoint_path'])==fit['selected']['checkpoint_sha256']
 # Opening receipt is written before fresh held-out generation.
 write(O/'EVAL_OPENING87_V1.json',dict(utc=utc(),status='ALL_NINE_NATIVE_SELECTIONS_AND_ORDINARY_SETTING_SEALED',selection_sha256=sha(O/'SELECTION87_V1.json'),ordinary_sha256=sha(O/'ORDINARY_SELECTION87_V1.json')))
 folder=cohorts();combined={}
 for cohort in ['retained','crossed','unselected']:
  items=[]
  for fam in FAMILIES:
   with np.load(folder/f'{fam}_{cohort}_v1.npz') as z:items.append({k:z[k] for k in z.files})
  combined[cohort]={k:np.concatenate([d[k] for d in items]) for k in items[0]}
 models=[dict(name='native_frozen',seed=None,arm='frozen',checkpoint_path=None)]+[dict(name=f"native_seed{f['seed']}_{f['arm']}",seed=f['seed'],arm=f['arm'],checkpoint_path=f['selected']['checkpoint_path']) for f in selection['fits']]+[dict(name='ordinary_weak_polynomial',seed=None,arm='ordinary'),dict(name='persistence',seed=None,arm='persistence')]
 output=folder/'predictions';output.mkdir();rows=[];checks=[]
 for item in models:
  name=item['name'];native=name.startswith('native');m=None
  if native:
   m=load_native();
   if item['checkpoint_path']:
    package=torch.load(item['checkpoint_path'],map_location='cpu',weights_only=False);m.load_state_dict(package['state_dict'],strict=True);del package
   m.eval()
   for parameter in m.parameters():parameter.requires_grad_(False)
  for cohort,data in combined.items():
   stage=time.perf_counter();loc1=data['context_locations'];loc2=data['future_locations']
   if native:
    fn=NativeField(m,data['context']);direct=predict_field(m,data['context'],loc1[:,:3]);cached=fn(torch.as_tensor(loc1[:,:3],device='cuda',dtype=torch.float32)).cpu().numpy();diff=float(abs(direct-cached).max());assert np.allclose(direct,cached,atol=2e-4,rtol=2e-5),(name,cohort,diff);checks.append(dict(model=name,cohort=cohort,cached_direct_max_abs=diff))
    f1=fn(torch.as_tensor(loc1,device='cuda',dtype=torch.float32)).cpu().numpy();f2=fn(torch.as_tensor(loc2,device='cuda',dtype=torch.float32)).cpu().numpy();prediction=native_rollout(fn,data['query_initial']);del fn
   elif item['arm']=='ordinary':
    deg=ordinary['selected']['degree'];coef=identify(data['context'],deg,ordinary['selected']['ridge']);f1=poly_predict(coef,loc1,deg);f2=poly_predict(coef,loc2,deg);prediction=ordinary_rollout(coef,deg,data['query_initial'])
   else:f1=np.zeros_like(loc1);f2=np.zeros_like(loc2);prediction=np.broadcast_to(data['query_initial'][:,None,:],data['truth'].shape).copy()
   path=output/f'{name}_{cohort}_v1.npz';save(path,prediction=prediction,context_field_prediction=f1,future_field_prediction=f2)
   for fi,fam in enumerate(FAMILIES):
    sl=slice(fi*128,(fi+1)*128);err1=f1[sl]-data['context_field'][sl];err2=f2[sl]-data['future_field'][sl]
    row=dict(model=name,seed=item['seed'],arm=item['arm'],family=fam,cohort=cohort,context_field_mse=float(np.mean(err1*err1)),future_field_mse=float(np.mean(err2*err2)),field_finite=bool(np.isfinite(f1[sl]).all() and np.isfinite(f2[sl]).all()),horizons=[metrics(prediction[sl],data['truth'][sl],h) for h in [1,5,10]],array_path=str(path),array_sha256=sha(path));rows.append(row)
   log(dict(stage='closed_eval',model=name,cohort=cohort,closed_rows=len(rows),seconds=time.perf_counter()-stage,h10=[r['horizons'][-1] for r in rows[-2:]]))
  if m is not None:del m;gc.collect();torch.cuda.empty_cache()
 assert len(rows)==72;write(O/'EVALUATION_RESULTS87_V1.json',dict(utc=utc(),status='TWELVE_MODELS_THREE_COHORTS_TWO_FAMILIES_CLOSED_UNVERIFIED',rows=rows,cached_native_checks=checks,selection_sha256=sha(O/'SELECTION87_V1.json'),evaluation_manifest_sha256=sha(O/'EVAL_MANIFEST87_V1.json'),ordinary_selection=ordinary['selected'],elapsed_seconds=time.perf_counter()-started,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED'))
if __name__=='__main__':main()
