"""Ordinary joint latent fitting on observed contexts; world-model weights stay frozen."""
import pathlib,sys,json,hashlib,datetime,time,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'joint_context_refinement_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\joint_context_refinement_v1');ART.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
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
 assert read(ROOT/'context_resampling_verification_v1/VERIFICATION_V1.json')['groups_verified']==66
 previous=read(ROOT/'context_transplant_audit_v1/RESULTS_V1.json');plan=read(ROOT/'context_transplant_audit_v1/PLAN_V1.json');oldpath=ART.parent/'context_transplant_audit_v1/arrays.npz';assert sha_file(oldpath)==previous['payload_sha256'];old=np.load(oldpath)
 torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(1000)
 sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 from dataloader import MyDataset
 test=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(test['path'])==test['file_sha256'];ds=MyDataset(test['path'],12,42,'cuda')
 model=iLaD('cuda').cuda();ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';assert sha_file(ckpt)==plan['checkpoint_sha256'];model.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));model.eval()
 for p in model.parameters():p.requires_grad_(False)
 before=digest(model);assert before==previous['model_state_after'];environments=sorted(map(int,plan['contexts']));e=len(environments);assert e==38;emap={env:i for i,env in enumerate(environments)}
 context_indices=np.array([plan['contexts'][str(env)] for env in environments]);assert context_indices.shape==(e,11)
 query_indices=set(int(x) for name in ['source_indices','target_indices'] for x in old[name]);assert not query_indices.intersection(context_indices.reshape(-1).tolist())
 context=ds.images[torch.from_numpy(context_indices).cuda(),:40].contiguous().reshape(e*11,40,32,32)
 with torch.no_grad():
  c0=[];lv0=[]
  for indexes in context_indices:
   h=model.encoder.embed(ds.images[torch.from_numpy(indexes).cuda(),:40]).mean(0,keepdim=True);c0.append(model.encoder.fc_mu(h)[0]);lv0.append(model.encoder.fc_logvar(h)[0])
  c0=torch.stack(c0);lv0=torch.stack(lv0);z0,_=model.vae.encode(context[:,:3].contiguous());z0=z0.reshape(e,11,2)
 rows=plan['rows'];senv=torch.tensor([emap[x['source_environment']] for x in rows],device='cuda');tenv=torch.tensor([emap[x['target_environment']] for x in rows],device='cuda')
 cerror=max(float((c0[senv]-torch.from_numpy(old['source_c']).cuda()).abs().max()),float((c0[tenv]-torch.from_numpy(old['target_c']).cuda()).abs().max()));assert cerror<1e-5,cerror
 dump('PLAN_V1.json',dict(protocol_sha256=sha_file(ROOT/'JOINT_CONTEXT_REFINEMENT_PROTOCOL_V1.md'),checkpoint_sha256=plan['checkpoint_sha256'],data_sha256=test['file_sha256'],old_payload_sha256=previous['payload_sha256'],source_revision=plan['source_revision'],rows=rows,environment_ids=environments,context_indices=context_indices.tolist(),initial_context_code_max_error=cerror,new_network_weight_training_fits=0,context_update_budget_per_control=100,query_prefix_update_budget=60))
 grid=torch.linspace(0.,3.9,40,device='cuda')
 def rollout(z,c,t=grid):return model.decode(model.dynamics(z,c,compute_logp=False,t=t))
 def context_losses(z,c):return (rollout(z.reshape(-1,2),c.repeat_interleave(11,0))-context).square().sum((2,3)).mean(1).reshape(e,11).mean(1)
 with torch.no_grad():initial_context_prediction=rollout(z0.reshape(-1,2),c0.repeat_interleave(11,0));initial_loss=(initial_context_prediction-context).square().sum((2,3)).mean(1).reshape(e,11).mean(1)
 def fit_context(joint):
  start=time.monotonic();state=torch.nn.Parameter(z0.clone());code=torch.nn.Parameter(c0.clone()) if joint else c0
  parameters=[state,code] if joint else [state];optimizer=torch.optim.Adam(parameters,lr=.01);best_z=z0.clone();best_c=c0.clone();best_loss=initial_loss.clone();best_step=torch.zeros(e,device='cuda',dtype=torch.int64);trace=[initial_loss.cpu().numpy()]
  for step in range(1,101):
   optimizer.zero_grad(set_to_none=True);loss=context_losses(state,code);loss.mean().backward();torch.nn.utils.clip_grad_norm_(parameters,10.);optimizer.step()
   with torch.no_grad():
    current=context_losses(state,code);assert torch.isfinite(current).all();improved=current<best_loss;best_z[improved]=state.detach()[improved];best_c[improved]=code.detach()[improved];best_loss[improved]=current[improved];best_step[improved]=step;trace.append(current.cpu().numpy())
   if step%25==0:event('context_latent_fit',joint=joint,step=step,current_context_mse=float(current.mean()),selected_context_mse=float(best_loss.mean()))
  with torch.no_grad():prediction=rollout(best_z.reshape(-1,2),best_c.repeat_interleave(11,0))
  return dict(z=best_z,c=best_c,loss=best_loss,step=best_step,trace=np.stack(trace),prediction=prediction,seconds=time.monotonic()-start)
 roots_only=fit_context(False);joint=fit_context(True);assert torch.equal(roots_only['c'],c0)
 si=old['source_indices'];ti=old['target_indices'];xs=ds.images[torch.from_numpy(si).cuda(),:40];xt=ds.images[torch.from_numpy(ti).cuda(),:40];history=xs[:,:3].contiguous();zs=torch.from_numpy(old['source_z0']).cuda();zt=torch.from_numpy(old['target_prefix_z0']).cuda();cs=joint['c'][senv];ct=joint['c'][tenv]
 with torch.no_grad():
  refined_factual=rollout(zs,cs);refined_naive=rollout(zs,ct);refined_target=rollout(zt,ct);prefix_prediction=rollout(zs,cs,grid[:3]);best_loss=(prefix_prediction-history).square().sum((2,3)).mean(1);best=zs.clone();best_step=torch.zeros(160,device='cuda',dtype=torch.int64)
 state=torch.nn.Parameter(zs.clone());optimizer=torch.optim.Adam([state],lr=.01);prefix_trace=[best_loss.cpu().numpy()]
 for step in range(1,61):
  optimizer.zero_grad(set_to_none=True);loss=(rollout(state,cs,grid[:3])-history).square().sum((2,3)).mean(1);loss.mean().backward();torch.nn.utils.clip_grad_norm_([state],10.);optimizer.step()
  with torch.no_grad():
   current=(rollout(state,cs,grid[:3])-history).square().sum((2,3)).mean(1);assert torch.isfinite(current).all();improved=current<best_loss;best[improved]=state.detach()[improved];best_loss[improved]=current[improved];best_step[improved]=step;prefix_trace.append(current.cpu().numpy())
 with torch.no_grad():refined_fit=rollout(best,ct);refined_fit_factual=rollout(best,cs);selected_query_prefix=rollout(best,cs,grid[:3])
 assert digest(model)==before
 arrays=dict(context_indices=context_indices,environment_ids=np.array(environments),context_initial_states=z0.cpu().numpy(),initial_codes=c0.cpu().numpy(),initial_code_logvars=lv0.cpu().numpy(),initial_context_prediction=initial_context_prediction.cpu().numpy(),roots_only_context_prediction=roots_only['prediction'].cpu().numpy(),joint_context_prediction=joint['prediction'].cpu().numpy(),roots_only_states=roots_only['z'].cpu().numpy(),joint_states=joint['z'].cpu().numpy(),refined_codes=joint['c'].cpu().numpy(),roots_only_trace=roots_only['trace'],joint_trace=joint['trace'],roots_only_best_step=roots_only['step'].cpu().numpy(),joint_best_step=joint['step'].cpu().numpy(),roots_only_best_loss=roots_only['loss'].cpu().numpy(),joint_best_loss=joint['loss'].cpu().numpy(),refined_query_states=best.cpu().numpy(),query_prefix_trace=np.stack(prefix_trace),query_prefix_best_step=best_step.cpu().numpy(),query_prefix_best_loss=best_loss.cpu().numpy(),selected_query_prefix_prediction=selected_query_prefix.cpu().numpy(),refined_factual_prediction=refined_factual.cpu().numpy(),refined_naive_prediction=refined_naive.cpu().numpy(),refined_target_prefix_prediction=refined_target.cpu().numpy(),refined_fit_prediction=refined_fit.cpu().numpy(),refined_fit_factual_prediction=refined_fit_factual.cpu().numpy())
 payload=ART/'arrays.npz'
 with payload.open('xb') as f:np.savez_compressed(f,**arrays)
 true_s=xs.cpu().double().numpy();true_t=xt.cpu().double().numpy();effect=true_t[:,3:]-true_s[:,3:]
 def mse(p,t):return np.square(p.astype(np.float64)-t).sum((2,3)).mean(1)
 scores={}
 paths=[('old_naive',old['naive_transplant_prediction'],old['factual_prediction']),('old_fit',old['shooting_transplant_prediction'],old['shooting_factual_prediction']),('refined_naive',arrays['refined_naive_prediction'],arrays['refined_factual_prediction']),('refined_fit',arrays['refined_fit_prediction'],arrays['refined_fit_factual_prediction'])]
 for name,cf,factual in paths:
  scores[name+'_cf']=mse(cf[:,3:],true_t[:,3:]);scores[name+'_factual']=mse(factual[:,3:],true_s[:,3:]);scores[name+'_response']=mse(cf[:,3:].astype(np.float64)-factual[:,3:].astype(np.float64),effect)
 scores['old_target_prefix']=mse(old['target_prefix_diagnostic_prediction'][:,3:],true_t[:,3:]);scores['refined_target_prefix']=mse(arrays['refined_target_prefix_prediction'][:,3:],true_t[:,3:]);scores['zero_effect']=np.square(effect).sum((2,3)).mean(1)
 pair_rows=[]
 for pair in range(20):
  means={k:float(v[pair*8:(pair+1)*8].mean()) for k,v in scores.items()};new_ratio=means['refined_fit_cf']/means['refined_target_prefix'];new_response=means['refined_fit_response']/means['zero_effect'];pair_rows.append(dict(pair=pair,source_environment=rows[pair*8]['source_environment'],target_environment=rows[pair*8]['target_environment'],metrics=means,refined_fit_ratio=new_ratio,refined_fit_response_ratio=new_response,within_descriptive_thresholds=bool(new_ratio<=1.25 and new_response<=.25)))
 drift=(joint['c']-c0).detach();prior_std=torch.exp(.5*lv0)
 result=dict(status='OBSERVED_CONTEXT_JOINT_LATENT_REFINEMENT_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),aggregate={k:float(v.mean()) for k,v in scores.items()},pairs=pair_rows,context_mse_initial=float(initial_loss.mean()),context_mse_roots_only=float(roots_only['loss'].mean()),context_mse_joint=float(joint['loss'].mean()),context_control_seconds=roots_only['seconds'],joint_context_seconds=joint['seconds'],code_displacement_max=float(drift.abs().max()),code_displacement_in_native_std_max=float((drift.abs()/prior_std).max()),context_optimizer_updates=200,query_state_optimizer_updates=60,new_network_weight_training_fits=0,model_state_before=before,model_state_after=digest(model),payload_sha256=sha_file(payload),independent_verification_pending=True,novel_direction_accepted=False,local_payload_sync=False)
 dump('RESULTS_V1.json',result);event('joint_refinement_complete',aggregate=result['aggregate'],residual_pairs=[x for x in pair_rows if x['pair'] in [5,6,8]],context_mse_joint=result['context_mse_joint'])
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
