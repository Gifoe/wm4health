"""A frozen-weight structural-twin test with an ordinary prefix-only state fit."""
import pathlib,sys,json,hashlib,datetime,time,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'context_transplant_audit_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\context_transplant_audit_v1');ART.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
try:
 verified=read(ROOT/'pendulum_native_closure_v2/VERIFICATION_V1.json');assert verified['all_factual_competence_gates'] and verified['all_refined_reference_metrics_within_one_percent']
 torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(1000)
 sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 from dataloader import MyDataset
 data_receipt=read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json');test=next(x for x in data_receipt['sets'] if x['split']=='test');assert sha_file(test['path'])==test['file_sha256']
 ds=MyDataset(test['path'],12,42,'cuda');assert len(ds)==3000 and np.array_equal(ds.labels,np.tile(np.arange(75),40))
 model=iLaD('cuda').cuda();ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';model.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));model.eval()
 for p in model.parameters():p.requires_grad_(False)
 before=digest(model);rng=np.random.default_rng(202610232);sources=rng.permutation(75)[:20];rows=[];context={}
 for pair,s in enumerate(sources):
  s=int(s);t=(s+37)%75;excluded=set((ds.spt_idx[s]//75).tolist()+(ds.spt_idx[t]//75).tolist());allowed=np.array([i for i in range(40) if i not in excluded]);roots=rng.permutation(allowed)[:8];assert len(roots)==8
  for env in [s,t]:context[env]=ds.spt_idx[env][:11].tolist()
  for initial in roots:
   si=int(initial)*75+s;ti=int(initial)*75+t;assert torch.equal(ds.states[si,0],ds.states[ti,0]);assert si not in context[s] and ti not in context[t]
   rows.append(dict(pair=pair,source_environment=s,target_environment=t,initial_state_index=int(initial),source_index=si,target_index=ti))
 assert len(rows)==160
 dump('PLAN_V1.json',dict(protocol_sha256=sha_file(ROOT/'CONTEXT_TRANSPLANT_AUDIT_PROTOCOL_V1.md'),checkpoint_sha256=sha_file(ckpt),data_sha256=test['file_sha256'],rows=rows,contexts=context,source_revision=data_receipt['source_revision'],mean_point_estimates=True,weight_training_fits=0))
 si=torch.tensor([x['source_index'] for x in rows],device='cuda');ti=torch.tensor([x['target_index'] for x in rows],device='cuda');xs=ds.images[si,:40];xt=ds.images[ti,:40];history=xs[:,:3].contiguous()
 c={};logvars={}
 with torch.no_grad():
  for env,indexes in context.items():
   h=model.encoder.embed(ds.images[torch.tensor(indexes,device='cuda'),:40]).mean(0,keepdim=True);c[env]=model.encoder.fc_mu(h)[0];logvars[env]=model.encoder.fc_logvar(h)[0]
  cs=torch.stack([c[x['source_environment']] for x in rows]);ct=torch.stack([c[x['target_environment']] for x in rows]);zs,_=model.vae.encode(history);zt,_=model.vae.encode(xt[:,:3].contiguous())
 grid=torch.linspace(0.,3.9,40,device='cuda');prefix_grid=grid[:3]
 def rollout(z,code,times):
  latent=model.dynamics(z,code,compute_logp=False,t=times);return model.decode(latent),latent
 with torch.no_grad():
  factual,z_factual=rollout(zs,cs,grid);naive,z_naive=rollout(zs,ct,grid);privileged,z_privileged=rollout(zt,ct,grid)
  original_prefix,_=rollout(zs,cs,prefix_grid);best_loss=(original_prefix-history).square().sum((2,3)).mean(1);best=zs.clone();best_step=torch.zeros(len(rows),device='cuda',dtype=torch.int64)
 state=torch.nn.Parameter(zs.clone());optimizer=torch.optim.Adam([state],lr=.01);trace=[best_loss.cpu().numpy()];aggregate=[]
 for step in range(1,61):
  optimizer.zero_grad(set_to_none=True);reconstruction,_=rollout(state,cs,prefix_grid);loss=(reconstruction-history).square().sum((2,3)).mean(1);loss.mean().backward();torch.nn.utils.clip_grad_norm_([state],10.);optimizer.step()
  with torch.no_grad():
   reconstruction,_=rollout(state,cs,prefix_grid);current=(reconstruction-history).square().sum((2,3)).mean(1);assert torch.isfinite(current).all();improved=current<best_loss;best[improved]=state.detach()[improved];best_loss[improved]=current[improved];best_step[improved]=step;trace.append(current.cpu().numpy());aggregate.append(dict(step=step,current_prefix_mse=float(current.mean()),selected_prefix_mse=float(best_loss.mean())))
  if step%20==0:print(json.dumps(dict(stage='prefix_only_state_fit',**aggregate[-1])),flush=True)
 with torch.no_grad():
  shooting,z_shooting=rollout(best,ct,grid);shooting_factual,z_shooting_factual=rollout(best,cs,grid);selected_prefix,_=rollout(best,cs,prefix_grid);identity,identity_z=rollout(zs,cs,grid)
 assert torch.equal(identity_z,z_factual);assert digest(model)==before
 arrays=dict(source_indices=si.cpu().numpy(),target_indices=ti.cpu().numpy(),source_c=cs.cpu().numpy(),target_c=ct.cpu().numpy(),source_z0=zs.cpu().numpy(),target_prefix_z0=zt.cpu().numpy(),fitted_z0=best.cpu().numpy(),prefix_loss_trace=np.stack(trace),selected_prefix_loss=best_loss.cpu().numpy(),selected_prefix_step=best_step.cpu().numpy(),selected_prefix_prediction=selected_prefix.cpu().numpy(),factual_prediction=factual.cpu().numpy(),naive_transplant_prediction=naive.cpu().numpy(),target_prefix_diagnostic_prediction=privileged.cpu().numpy(),shooting_transplant_prediction=shooting.cpu().numpy(),shooting_factual_prediction=shooting_factual.cpu().numpy())
 payload=ART/'arrays.npz'
 with payload.open('xb') as f:np.savez_compressed(f,**arrays)
 true_s=xs.cpu().double().numpy();true_t=xt.cpu().double().numpy();true_effect=true_t[:,3:]-true_s[:,3:]
 def mse(a,b):return np.square(a.astype(np.float64)-b.astype(np.float64)).sum((2,3)).mean(1)
 scores={key:mse(arrays[key],true_s if key in ['factual_prediction','shooting_factual_prediction'] else true_t) for key in []}
 for key in ['factual_prediction','shooting_factual_prediction']:scores[key]=mse(arrays[key][:,3:],true_s[:,3:])
 for key in ['naive_transplant_prediction','target_prefix_diagnostic_prediction','shooting_transplant_prediction']:scores[key]=mse(arrays[key][:,3:],true_t[:,3:])
 scores['source_persistence']=mse(np.repeat(true_s[:,2:3],37,axis=1),true_s[:,3:]);scores['target_persistence']=mse(np.repeat(true_t[:,2:3],37,axis=1),true_t[:,3:])
 scores['zero_effect_error']=np.square(true_effect).sum((2,3)).mean(1)
 for name,cf,base in [('naive','naive_transplant_prediction','factual_prediction'),('shooting','shooting_transplant_prediction','shooting_factual_prediction')]:scores[name+'_response_error']=np.square((arrays[cf][:,3:].astype(np.float64)-arrays[base][:,3:].astype(np.float64))-true_effect).sum((2,3)).mean(1)
 means={k:float(v.mean()) for k,v in scores.items()};naive_ratio=means['naive_transplant_prediction']/means['target_prefix_diagnostic_prediction'];shoot_ratio=means['shooting_transplant_prediction']/means['target_prefix_diagnostic_prediction'];naive_response=means['naive_response_error']/means['zero_effect_error'];shoot_response=means['shooting_response_error']/means['zero_effect_error']
 gate=means['factual_prediction']<.9*means['source_persistence'] and means['target_prefix_diagnostic_prediction']<.9*means['target_persistence']
 excess=means['naive_transplant_prediction']-means['target_prefix_diagnostic_prediction'];reduction=1-(means['shooting_transplant_prediction']-means['target_prefix_diagnostic_prediction'])/excess if excess>0 else None
 if not gate:decision='AUDIT_POINT_VARIANTS_NOT_FACTUALLY_QUALIFIED'
 elif naive_ratio<=1.25 and naive_response<=.25:decision='REJECT_CANDIDATE_NATIVE_CONTEXT_TRANSPLANT_SUFFICES'
 elif (shoot_ratio<=1.25 and shoot_response<=.25) or (reduction is not None and reduction>=.75):decision='ORDINARY_STATE_INFERENCE_EXPLAINS_TRANSPLANT_DEFICIT'
 else:decision='CANDIDATE_REQUIRES_ATTRIBUTION_AND_PRIOR_ART_NOT_ACCEPTED_NOVELTY'
 perrow=[dict(**row,**{k:float(v[i]) for k,v in scores.items()}) for i,row in enumerate(rows)]
 result=dict(status='FROZEN_CHECKPOINT_CONTEXT_TRANSPLANT_AUDIT_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision=decision,aggregate=means,naive_to_target_prefix_mse_ratio=naive_ratio,shooting_to_target_prefix_mse_ratio=shoot_ratio,naive_normalized_response_error=naive_response,shooting_normalized_response_error=shoot_response,ordinary_state_fit_excess_mse_reduction=reduction,point_variant_factual_gate=gate,pair_count=20,root_rows=160,distinct_environments=len(context),rows=perrow,optimization_trace=aggregate,state_estimation_optimizer_updates=60,weight_training_fits=0,model_state_before=before,model_state_after=digest(model),payload_sha256=sha_file(payload),independent_verification_pending=True,novel_direction_accepted=False,local_payload_sync=False)
 dump('RESULTS_V1.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','optimization_trace']}),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
