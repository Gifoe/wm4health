"""Finite-library context sensitivity with all registered pairs and fixed state estimates."""
import pathlib,sys,json,hashlib,datetime,shutil,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'context_resampling_audit_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\context_resampling_audit_v1');ART.mkdir(exist_ok=False)
OLD=ART.parent/'context_transplant_audit_v1/arrays.npz'
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
PATHS=['factual','naive_cf','target_prefix','frozen_fit_cf']
KEYS=['factual_mse','naive_cf_mse','target_prefix_mse','frozen_fit_cf_mse','naive_response_target_only','naive_response_both_contexts','frozen_fit_response_target_only']
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def event(stage,**kw):
 row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
try:
 verified=read(ROOT/'context_transplant_verification_v1/VERIFICATION_V1.json');assert verified['decision']=='REJECT_CANDIDATE_NATIVE_CONTEXT_TRANSPLANT_SUFFICES'
 previous=read(ROOT/'context_transplant_audit_v1/RESULTS_V1.json');plan=read(ROOT/'context_transplant_audit_v1/PLAN_V1.json');assert sha_file(OLD)==previous['payload_sha256'];old=np.load(OLD)
 torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(1000)
 sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 from dataloader import MyDataset
 test=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(test['path'])==test['file_sha256'];assert shutil.disk_usage(ART).free>16*1024**3
 ds=MyDataset(test['path'],12,42,'cuda');model=iLaD('cuda').cuda();ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';assert sha_file(ckpt)==plan['checkpoint_sha256'];model.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));model.eval()
 for parameter in model.parameters():parameter.requires_grad_(False)
 before=digest(model);assert before==previous['model_state_after'];rows=plan['rows'];allowed=[];legacy=[]
 for pair in range(20):
  rr=rows[pair*8:(pair+1)*8];assert all(x['pair']==pair for x in rr);roots={x['initial_state_index'] for x in rr};s=rr[0]['source_environment'];t=rr[0]['target_environment'];lib=[];original=[]
  for env in [s,t]:
   indexes=np.array([i for i in np.where(ds.labels==env)[0] if i//75 not in roots]);assert len(indexes)==32
   lib.append(indexes.tolist());original.append(plan['contexts'][str(env)]);assert set(original[-1]).issubset(indexes.tolist())
  allowed.append(lib);legacy.append(original)
 sets=[legacy,allowed];rng=np.random.default_rng(202610233)
 for draw in range(64):sets.append([[sorted(rng.choice(pool,11,replace=False).tolist()) for pool in pair] for pair in allowed])
 labels=['legacy11','extra_information32']+[f'fixed11_draw{i:02d}' for i in range(64)]
 dump('PLAN_V1.json',dict(protocol_sha256=sha_file(ROOT/'CONTEXT_RESAMPLING_PROTOCOL_V1.md'),checkpoint_sha256=plan['checkpoint_sha256'],data_sha256=test['file_sha256'],old_payload_sha256=previous['payload_sha256'],rows=rows,context_sets=sets,group_labels=labels,seed=202610233,new_weight_training_fits=0,new_state_estimation_updates=0))
 with torch.no_grad():h=torch.cat([model.encoder.embed(ds.images[i:i+32,:40]) for i in range(0,3000,32)])
 feature_errors=[];code_errors=[]
 with torch.no_grad():
  for indexes in [legacy[0][0],allowed[0][0]]:
   ix=torch.tensor(indexes,device='cuda');direct=model.encoder.embed(ds.images[ix,:40]);feature_errors.append(float((direct-h[ix]).abs().max()));code_errors.append(float((model.encoder.fc_mu(direct.mean(0,keepdim=True))-model.encoder.fc_mu(h[ix].mean(0,keepdim=True))).abs().max()))
 assert max(feature_errors)<=1e-4 and max(code_errors)<=1e-5,(feature_errors,code_errors)
 cs=[];ct=[];lv=[]
 with torch.no_grad():
  for group in sets:
   codes=[];logvars=[]
   for pair in group:
    pc=[];pl=[]
    for indexes in pair:
     mean=h[torch.tensor(indexes,device='cuda')].mean(0,keepdim=True);pc.append(model.encoder.fc_mu(mean)[0]);pl.append(model.encoder.fc_logvar(mean)[0])
    codes.append(torch.stack(pc));logvars.append(torch.stack(pl))
   c=torch.stack(codes);cs.append(c[:,0].repeat_interleave(8,0));ct.append(c[:,1].repeat_interleave(8,0));lv.append(torch.stack(logvars))
 cs=torch.stack(cs);ct=torch.stack(ct);lv=torch.stack(lv)
 legacy_code_error=max(float((cs[0]-torch.from_numpy(old['source_c']).cuda()).abs().max()),float((ct[0]-torch.from_numpy(old['target_c']).cuda()).abs().max()));assert legacy_code_error<=1e-5,legacy_code_error
 zs=torch.from_numpy(old['source_z0']).cuda();zt=torch.from_numpy(old['target_prefix_z0']).cuda();zf=torch.from_numpy(old['fitted_z0']).cuda();grid=torch.linspace(0.,3.9,40,device='cuda')
 si=old['source_indices'];ti=old['target_indices'];truth_s=ds.images[torch.from_numpy(si).cuda(),:40].cpu().double().numpy();truth_t=ds.images[torch.from_numpy(ti).cuda(),:40].cpu().double().numpy();effect=truth_t[:,3:]-truth_s[:,3:]
 fixed_fact=old['factual_prediction'].astype(np.float64);fixed_fit_fact=old['shooting_factual_prediction'].astype(np.float64)
 stores={name:np.lib.format.open_memmap(ART/(name+'.npy'),mode='w+',dtype=np.float32,shape=(66,160,40,32,32)) for name in PATHS};scores=np.empty((66,160,len(KEYS)),dtype=np.float64);legacy_pixel_errors={}
 def rollout(z,c):return model.decode(model.dynamics(z,c,compute_logp=False,t=grid))
 def mse(p,t):return np.square(p.astype(np.float64)-t).sum((2,3)).mean(1)
 for group in range(66):
  with torch.no_grad():preds=[rollout(zs,cs[group]),rollout(zs,ct[group]),rollout(zt,ct[group]),rollout(zf,ct[group])]
  arrays=[p.cpu().numpy() for p in preds];assert all(np.isfinite(p).all() for p in arrays)
  if group==0:
   for name,array,key in zip(PATHS,arrays,['factual_prediction','naive_transplant_prediction','target_prefix_diagnostic_prediction','shooting_transplant_prediction']):
    legacy_pixel_errors[name]=float(np.max(np.abs(array-old[key])));assert legacy_pixel_errors[name]<=1e-4,(name,legacy_pixel_errors[name])
   dump('CACHE_AND_REPLAY_QUALIFICATION_V1.json',dict(feature_errors=feature_errors,code_errors=code_errors,legacy_code_max_error=legacy_code_error,legacy_pixel_max_errors=legacy_pixel_errors,feature_tolerance=1e-4,code_tolerance=1e-5,pixel_tolerance=1e-4))
  for name,array in zip(PATHS,arrays):stores[name][group]=array;stores[name].flush()
  fact,cf,target,fit=arrays
  scores[group,:,0]=mse(fact[:,3:],truth_s[:,3:]);scores[group,:,1]=mse(cf[:,3:],truth_t[:,3:]);scores[group,:,2]=mse(target[:,3:],truth_t[:,3:]);scores[group,:,3]=mse(fit[:,3:],truth_t[:,3:])
  scores[group,:,4]=mse(cf[:,3:].astype(np.float64)-fixed_fact[:,3:],effect);scores[group,:,5]=mse(cf[:,3:].astype(np.float64)-fact[:,3:].astype(np.float64),effect);scores[group,:,6]=mse(fit[:,3:].astype(np.float64)-fixed_fit_fact[:,3:],effect)
  if group%8==0 or group==65:event('group_complete',group=group,label=labels[group],naive_cf_mse=float(scores[group,:,1].mean()),target_prefix_mse=float(scores[group,:,2].mean()))
 assert digest(model)==before
 zero=np.square(effect).sum((2,3)).mean(1);metadata=ART/'metadata.npz'
 with metadata.open('xb') as f:np.savez_compressed(f,metrics=scores,source_codes=cs.cpu().numpy(),target_codes=ct.cpu().numpy(),native_context_logvars=lv.cpu().numpy(),cached_embeddings=h.cpu().numpy(),mu_head_weight=model.encoder.fc_mu.weight.cpu().numpy(),mu_head_bias=model.encoder.fc_mu.bias.cpu().numpy(),logvar_head_weight=model.encoder.fc_logvar.weight.cpu().numpy(),logvar_head_bias=model.encoder.fc_logvar.bias.cpu().numpy(),source_indices=si,target_indices=ti,zero_effect_error=zero)
 summaries=[]
 for pair in range(20):
  pm=scores[:,pair*8:(pair+1)*8].mean(1);den=float(zero[pair*8:(pair+1)*8].mean());ratio=pm[:,1]/pm[:,2];fit_ratio=pm[:,3]/pm[:,2];response=pm[:,4]/den;paired_response=pm[:,5]/den;fit_response=pm[:,6]/den
  def describe(v):return dict(legacy=float(v[0]),extra32=float(v[1]),fixed11_mean=float(v[2:].mean()),fixed11_median=float(np.median(v[2:])),fixed11_q10=float(np.quantile(v[2:],.1)),fixed11_q90=float(np.quantile(v[2:],.9)),legacy_empirical_percentile=float(np.mean(v[2:]<=v[0])))
  summaries.append(dict(pair=pair,source_environment=rows[pair*8]['source_environment'],target_environment=rows[pair*8]['target_environment'],zero_effect_error=den,naive_ratio=describe(ratio),frozen_fit_ratio=describe(fit_ratio),naive_target_only_response=describe(response),naive_both_contexts_response=describe(paired_response),frozen_fit_target_only_response=describe(fit_response),fixed11_naive_exceedance_fraction=float(np.mean((ratio[2:]>1.25)|(response[2:]>.25))),fixed11_frozen_fit_exceedance_fraction=float(np.mean((fit_ratio[2:]>1.25)|(fit_response[2:]>.25))),extra32_frozen_fit_within_descriptive_thresholds=bool(fit_ratio[1]<=1.25 and fit_response[1]<=.25)))
 payloads={p.name:dict(bytes=p.stat().st_size,sha256=sha_file(p)) for p in ART.iterdir() if p.is_file()}
 result=dict(status='ALL_REGISTERED_PAIRS_CONTEXT_RESAMPLING_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),groups=66,fixed_context_draws=64,pairs=20,root_rows=160,paths=PATHS,metric_names=KEYS,summaries=summaries,original_residual_pair_ids=[5,6,8],payloads=payloads,model_state_before=before,model_state_after=digest(model),new_weight_training_fits=0,new_state_estimation_updates=0,reused_frozen_states_from_prior60_updates=True,independent_verification_pending=True,novel_direction_accepted=False,local_payload_sync=False)
 dump('RESULTS_V1.json',result);event('resampling_complete',groups=66,pairs=20,payload_bytes=sum(x['bytes'] for x in payloads.values()))
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
