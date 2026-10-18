from models65 import *
import time
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
f=read(OUT/'FITS65_V1.json');o=read(OUT/'ORACLE65_V1.json');assert f['status']=='ALL_18_MODELS_FROZEN' and o['status']=='EXACT_ORACLES_QUALIFIED';source_check();assert sha(o['artifact'])==o['artifact_sha256'];device=f['device'];pp=paths();truth=np.load(o['artifact']);D=truth['directions'];refs={};rows=[];start=time.perf_counter();P=512
for mech in MECHANISMS:
 for pk in ['zero','sine']:
  for target in ['do','obs']:
   x=truth[mech+'_'+pk+'_'+target];v=x@D.T;refs[(mech,pk,target)]=dict(x=x,sorted=np.sort(v,axis=0),scale=np.std(v,axis=0).mean(),tail=float(np.mean(np.abs(x[:,-1])>2)),mean=x.mean(0),vartrace=np.var(x,axis=0).sum())
def plan(pk):return np.zeros(H) if pk=='zero' else .4*np.sin(2*np.pi*(np.arange(H)+1)/8)
def metrics(x,mech,pk,target):
 ref=refs[(mech,pk,target)];xs=np.sort(x@D.T,axis=0);assert len(ref['x'])%len(x)==0;w=np.abs(np.repeat(xs,len(ref['x'])//len(x),axis=0)-ref['sorted']).mean(0);tail=float(np.mean(np.abs(x[:,-1])>2));return dict(finite_sliced_W1=float(w.mean()),normalized_finite_sliced_W1=float(w.mean()/ref['scale']),terminal_marginal_W1=float(w[-1]),path_mean_MSE=float(np.mean((x.mean(0)-ref['mean'])**2)),variance_trace_ratio=float(np.var(x,axis=0).sum()/ref['vartrace']),terminal_tail_probability=tail,reference_tail_probability=ref['tail'],terminal_tail_abs_error=abs(tail-ref['tail']))
def key(mech,train_seed,method,pk,mc):return f'{mech}_train{train_seed}_{method}_{pk}_MC{mc}'
def cached(kk):
 p=pp['artifact_root']/'law_receipts'/(kk+'.json')
 if not p.exists():return False
 r=read(p);assert r['fits_sha256']==sha(OUT/'FITS65_V1.json') and r['oracle_sha256']==sha(OUT/'ORACLE65_V1.json');assert sha(r['artifact'])==r['artifact_sha256'];rows.append(r);return True
def retain(kk,x,metadata):
 assert x.shape==(P,H) and np.isfinite(x).all();p=pp['artifact_root']/'laws'/(kk+'.npz');p.parent.mkdir(exist_ok=True,parents=True);assert not p.exists(),'Orphaned array; inspect before resuming';np.savez_compressed(p,x=x);r=dict(**metadata,law_key=kk,trajectories=P,artifact=str(p),artifact_sha256=sha(p),fits_sha256=sha(OUT/'FITS65_V1.json'),oracle_sha256=sha(OUT/'ORACLE65_V1.json'),scores={t:metrics(x,metadata['mechanism'],metadata['query'],t) for t in ['do','obs']});dump(pp['artifact_root']/'law_receipts'/(kk+'.json'),r);rows.append(r);state('evaluation_state65_v1.json',dict(status='SCORING_FROZEN_LAWS',completed_laws=len(rows),latest=kk));print(json.dumps(dict(law=kk,do_discrepancy=r['scores']['do']['normalized_finite_sliced_W1'],obs_discrepancy=r['scores']['obs']['normalized_finite_sliced_W1'])),flush=True)
def synchronize():
 if device=='cuda':torch.cuda.synchronize()
for fit in f['fits']:
 mech,seed=fit['mechanism'],fit['seed'];sel=fit['selected_checkpoint'];assert sha(sel['path'])==sel['sha256'];z=torch.load(sel['path'],map_location=device,weights_only=False)
 if fit['model'].startswith('CSDI'):
  model=csdi(device);model.load_state_dict(z['state_dict']);model.eval();mean=np.asarray(z['normalization_mean'],dtype='float32');std=np.asarray(z['normalization_std'],dtype='float32')
 else:model=MDN().to(device);model.load_state_dict(z['state_dict']);model.eval()
 for pk in ['zero','sine']:
  u=plan(pk)
  for mc in SEEDS:
   for operation in (['block','prefix'] if fit['model'].startswith('CSDI') else ['causal_rollout']):
    method=fit['model']+'_'+operation;kk=key(mech,seed,method,pk,mc)
    if cached(kk):continue
    torch.manual_seed(int(np.random.SeedSequence([65100,mc,seed,MECHANISMS.index(mech)]).generate_state(1)[0]));synchronize();t0=time.perf_counter()
    with torch.inference_mode():
     if fit['model'].startswith('CSDI'):
      raw=np.zeros((P,2,H),dtype='float32');m=torch.zeros((P,2,H),device=device);calls=50 if operation=='block' else H*50
      if operation=='block':
       raw[:,1]=u;m[:,1]=1;obs=torch.tensor((raw-mean)/std,device=device);ss=side(model,obs,m);a=model.impute(obs,m,ss,1)[:,0,0].cpu().numpy();x=a*std[0,0,0]+mean[0,0,0]
      else:
       x=np.empty((P,H),dtype='float32')
       for t in range(H):
        raw[:,1,:t+1]=u[:t+1];m[:,1,:t+1]=1
        if t:raw[:,0,:t]=x[:,:t];m[:,0,:t]=1
        obs=torch.tensor((raw-mean)/std,device=device);ss=side(model,obs,m);x[:,t]=model.impute(obs,m,ss,1)[:,0,0,t].cpu().numpy()*std[0,0,0]+mean[0,0,0]
     else:
      last=torch.full((P,),.25,device=device);out=[]
      for t in range(H):last=model.sample(torch.stack([last,torch.full_like(last,float(u[t]))],1));out.append(last.cpu().numpy())
      x=np.stack(out,1);calls=H
    synchronize();retain(kk,x,dict(method=method,model=fit['model'],operation=operation,mechanism=mech,training_seed=seed,query=pk,MC_seed=mc,checkpoint_sha256=sel['sha256'],selected_step=sel['step'],native_denoiser_calls_per_particle_batch=calls if fit['model'].startswith('CSDI') else 0,transition_network_calls_per_particle_batch=calls if fit['model']=='causal_MDN' else 0,runtime_seconds=time.perf_counter()-t0))
 del model,z
for corpus in f['corpora']:
 assert sha(corpus['path'])==corpus['sha256'];z=np.load(corpus['path']);xx,yy=transitions(z['train']);A=np.c_[xx,np.ones(len(xx))];coef=np.linalg.lstsq(A,yy,rcond=None)[0];sd=float(np.sqrt(np.mean((yy-A@coef)**2)));mech=corpus['mechanism'];seed=corpus['seed']
 for pk in ['zero','sine']:
  for mc in SEEDS:
   method='causal_linear_AR';kk=key(mech,seed,method,pk,mc)
   if cached(kk):continue
   t0=time.perf_counter();rng=np.random.default_rng(np.random.SeedSequence([65120,seed,mc,MECHANISMS.index(mech)]));u=plan(pk);last=np.full(P,.25);x=np.empty((P,H))
   for t in range(H):last=coef[0]*last+coef[1]*u[t]+coef[2]+sd*rng.normal(size=P);x[:,t]=last
   retain(kk,x,dict(method=method,model=method,operation='causal_rollout',mechanism=mech,training_seed=seed,query=pk,MC_seed=mc,ar_coefficients=coef.tolist(),noise_sd=sd,runtime_seconds=time.perf_counter()-t0,scope='Train-only existing linear AR; misspecified on nonlinear mechanism'))
for mech in MECHANISMS:
 for pk in ['zero','sine']:
  for mc in SEEDS:
   for target in ['do','obs']:
    method='privileged_iid_'+target;kk=key(mech,0,method,pk,mc)
    if cached(kk):continue
    x=truth[f'{mech}_{pk}_iid_{target}_s{mc}'];retain(kk,x,dict(method=method,model=method,operation='statistical_reference',mechanism=mech,training_seed=None,query=pk,MC_seed=mc,scope='Exact iid floor, not deployable inference method',runtime_seconds=None))
assert len(rows)==240,len(rows)
dump(OUT/'EVALUATION65_V1.json',dict(utc=utc(),status='ALL_240_FROZEN_LAWS_SCORED',laws=rows,generated_laws=len(rows),scored_target_views=2*len(rows),directions=136,reference_records_per_law=8192,trajectories_per_generated_law=P,elapsed_seconds=time.perf_counter()-start,fit_receipt_sha256=sha(OUT/'FITS65_V1.json'),oracle_receipt_sha256=sha(OUT/'ORACLE65_V1.json'),evaluation_spec_sha256=sha(ROOT/'EVALUATION_SPEC65_V1.md'),accepted_final_directions=0));state('evaluation_state65_v1.json',dict(status='ALL_240_FROZEN_LAWS_SCORED',generated_laws=len(rows),scored_target_views=2*len(rows)))
