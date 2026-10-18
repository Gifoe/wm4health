"""Sealed-model proper future density and remaining-life audit, without retuning."""
from native61_v2 import *
import numpy as np
from reference61_v1 import filter_all,matrix,survival,marginal_means

PREFIXES=[32,64,96];HORIZONS=[1,4,8,16,32]

def main():
 torch.set_num_threads(4);sources();seal=read(OUT/'MODEL_SEAL61_V1.json');data=read(OUT/'DATA61_V1.json')
 assert seal['status']=='ALL_TWELVE_EXISTING_MODEL_ATTEMPTS_TERMINAL_SEALED' and len(seal['models'])+len(seal['failed'])==12
 for row in seal['models']:
  assert sha(OUT/'runs'/row['tag']/'result.json')==row['result_sha256'];assert sha(row['selected']['checkpoint'])==row['selected']['checkpoint_sha256']
 dump(OUT/'TEST_OPEN61_V1.json',{'utc':utc(),'status':'FIRST_C61_TEST_SCORE_AFTER_ALL_FIXED_MODEL_SELECTION','seal_sha256':sha(OUT/'MODEL_SEAL61_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL61_V1.md'),'script_sha256':sha(__file__),'training_reads_audit_truth':False,'test_based_selection':False,'scoring_dtype':'float64 native CPU, exact cast of frozen Float32 weights, qualified native paths','generation_read_all_split_arrays_before_fits':True})
 folder=paths()['artifact_root']/'scores_v1';folder.mkdir(exist_ok=False);cache={};oracle_rows=[]
 for family in ['peaked','bimodal']:
  r=data['families'][family]['test'];a=data['audit_truth'][family]['test'];assert sha(r['file'])==r['sha256'] and sha(a['file'])==a['sha256']
  with np.load(r['file']) as f:X=f['X'].astype('float64')
  with np.load(a['file']) as f:law={k:f[k].copy() for k in ['initial','duration','transition','mean','sd']};ep=f['episode'].copy()
  transition=np.repeat(law['transition'][:,None,:],32,axis=1);post,ll,P,S=filter_all(X,law['initial'],law['duration'],transition,law['mean'],law['sd']**2)
  nll=np.stack([-(ll[:,p+47]-ll[:,p-1])/48 for p in PREFIXES],1);surv=np.stack([survival(post[:,p-1],S,HORIZONS) for p in PREFIXES],1);ends=np.stack([np.stack([ep[:,p-1+h]!=ep[:,p-1] for h in HORIZONS],1) for p in PREFIXES],1);pred=np.stack([marginal_means(post[:,p-1],P,law['mean'],48) for p in PREFIXES],1);actual=np.stack([X[:,p:p+48] for p in PREFIXES],1);mse=((pred-actual)**2).mean((2,3));brier=(1-surv-ends)**2;dest=folder/(family+'_oracle.npz')
  with dest.open('xb') as f:np.savez(f,conditional_nll=nll,survival_probability=surv,end_event=ends,marginal_mean=pred,observation_mse=mse,brier=brier)
  oracle_rows.append({'family':family,'conditional_nll':float(nll.mean()),'brier':float(brier.mean()),'observation_mse':float(mse.mean()),'arrays':str(dest),'arrays_sha256':sha(dest),'role':'Known-law reference, not a learned or deployable baseline'})
  cache[family]={'X':X,'oracle_nll':nll,'oracle_survival':surv,'end_event':ends,'actual':actual}
 records=[]
 for row in seal['models']:
  rec=read(OUT/'runs'/row['tag']/'result.json');d=cache[rec['family']];x=torch.tensor(d['X'],dtype=torch.float64);m=model(rec['arm'],rec['seed'],dtype=torch.float64);ck=torch.load(rec['selected']['checkpoint'],weights_only=True,map_location='cpu');m.load_state_dict(ck['model'],strict=True);m.eval();pi,dur,A=parameters(m);loc,cov=m.dist.emission();mu=loc[0,0].detach().numpy();variance=cov[0,0].diagonal(dim1=-2,dim2=-1).detach().numpy();pd=dur.detach().exp().numpy();trans=A.detach().exp().numpy();initial=pi.detach().exp().numpy();post,ind_ll,P,S=filter_all(d['X'],initial,pd,trans,mu,variance);native_nll=[];native_surv=[];maxq=0.
  with torch.no_grad():
   for j,p in enumerate(PREFIXES):
    # Only the prefix reaches the predictive posterior. Future observations enter a density score separately.
    alpha=m.forward(m._build_sequence_set(x[:,:p]));last=alpha[:,-1];prefix_ll=torch.logsumexp(last.flatten(1),-1);q=(last-prefix_ll[:,None,None]).exp();maxq=max(maxq,float(np.max(np.abs(q.numpy()-post[:,p-1]))));forecast=active_episode_survival_forecast(HSMMFilterState(q.clamp_min(1e-300).log()),dur[None].expand(len(x),-1,-1),HORIZONS).survival_probability.numpy()
    if rec['arm']=='hmm':
     # A same-state HMM transition is persistence, whereas a D=1 HSMM episode ends at every boundary.
     diag=np.array([trans[z,0,z] for z in range(3)]);forecast=np.stack([np.sum(q.numpy()[:,:,0]*diag[None]**h,1) for h in HORIZONS],1)
    native_surv.append(forecast);native_nll.append((-(m.log_likelihood(x[:,:p+48])-prefix_ll)/48).numpy())
  nll=np.stack(native_nll,1);surv=np.stack(native_surv,1);ind_nll=np.stack([-(ind_ll[:,p+47]-ind_ll[:,p-1])/48 for p in PREFIXES],1)
  if rec['arm']=='hmm':
   independent_surv=np.stack([np.stack([np.sum(post[:,p-1,:,0]*np.diagonal(trans[:,0,:])[None]**h,1) for h in HORIZONS],1) for p in PREFIXES],1)
  else:independent_surv=np.stack([survival(post[:,p-1],S,HORIZONS) for p in PREFIXES],1)
  maxnll=float(np.max(np.abs(nll-ind_nll)));maxsurv=float(np.max(np.abs(surv-independent_surv)));assert maxq<1e-9 and maxnll<1e-9 and maxsurv<1e-9,(row['tag'],maxq,maxnll,maxsurv)
  pred=np.stack([marginal_means(post[:,p-1],P,mu,48) for p in PREFIXES],1);mse=((pred-d['actual'])**2).mean((2,3));brier=(1-surv-d['end_event'])**2;prob_error=(surv-d['oracle_survival'])**2;dest=folder/(row['tag']+'.npz')
  with dest.open('xb') as f:np.savez(f,conditional_nll=nll,survival_probability=surv,oracle_survival_probability=d['oracle_survival'],end_event=d['end_event'],observation_mse=mse,marginal_mean=pred,brier=brier,probability_squared_error=prob_error,initial=initial,duration=pd,transition=trans,emission_mean=mu,emission_variance=variance)
  r={'tag':row['tag'],'family':rec['family'],'arm':rec['arm'],'seed':rec['seed'],'selected_step':rec['selected']['step'],'parameters':rec['parameters'],'trainable_parameters':rec['trainable_parameters'],'updates':rec['updates_completed'],'training_seconds':rec['training_seconds'],'conditional_nll':float(nll.mean()),'conditional_nll_above_oracle':float((nll-d['oracle_nll']).mean()),'brier':float(brier.mean()),'survival_probability_mse_vs_oracle':float(prob_error.mean()),'observation_mse':float(mse.mean()),'max_native_vs_independent_posterior_error':maxq,'max_native_vs_independent_conditional_nll_error':maxnll,'max_native_vs_independent_survival_error':maxsurv,'arrays':str(dest),'arrays_sha256':sha(dest),'checkpoint_sha256':rec['selected']['checkpoint_sha256'],'nll_selection':'Validation only, prior to test opening','finite':bool(np.isfinite(nll).all() and np.isfinite(surv).all())}
  assert r['finite'];records.append(r);print(json.dumps({k:r[k] for k in ['tag','conditional_nll','conditional_nll_above_oracle','brier','survival_probability_mse_vs_oracle']}),flush=True)
 dump(OUT/'EVALUATION61_V1.json',{'utc':utc(),'status':'SEALED_EXISTING_HSMM_AND_HMM_SCORED_WITH_INDEPENDENT_REPLAY','records':records,'oracle':oracle_rows,'failed':seal['failed'],'prefixes':PREFIXES,'survival_horizons':HORIZONS,'physical_test_records_per_family':128,'real_domains':0,'seal_sha256':sha(OUT/'MODEL_SEAL61_V1.json'),'test_open_sha256':sha(OUT/'TEST_OPEN61_V1.json'),'script_sha256':sha(__file__),'scope':'Two controlled finite-duration Gaussian-emission families. Unknown law parameters learned without latent labels, but known state count/support and fresh-start convention. No new method, arbitrary-real-data capability or accepted research direction.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0})

if __name__=='__main__':main()
