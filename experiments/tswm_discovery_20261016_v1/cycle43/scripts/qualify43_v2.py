"""Train discarded preflight, qualify actual native coarse/continuation APIs."""
from harness43_v2 import *
import datetime,time,traceback,copy
O=R/'outputs/native_v2';assert not O.exists();O.mkdir(parents=True)
record={'status':'running','scope':'source-native backbone/collator/scheduler and AST-extracted author methods; not full Lightning or headline reproduction','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(O/'qualification.json',record)
try:
 for item in json.loads((S/'CACHE_RECEIPTS.json').read_text())['files']:assert sha(S/item['path'])==item['sha256']
 seed(202643);ctx,hashes=make_context();examples=toy_examples();config=collate_config();opt=torch.optim.AdamW(ctx.model.parameters(),lr=3e-4)
 losses=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 for step in range(50):
  ctx.model.train();batch=collate_fn_spf(examples,config,ctx.noise_scheduler,ctx.datamodule_config.output_seq_len,is_training=True);batch=to_device(batch,ctx.device)
  opt.zero_grad(set_to_none=True);loss=ctx.training_step(batch,step)['loss'];assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 cond_grads={name:(None if p.grad is None else float(p.grad.abs().max())) for name,p in ctx.model.named_parameters() if name.startswith('cond_pos_embed.')}
 ctx.model.eval();xs,meta=prediction_inputs(examples[:2],ctx.device);shapes={};failures=[];standard={};native_valid={}
 with torch.no_grad():
  for ts,length in [('decadal',1),('yearly',1),('monthly',16)]:
   seed(20264301);standard[ts]=ctx.predict(xs,metadata=meta,timescale=ts,latent_size=[1,2,4]);assert standard[ts].shape==(2,length,1,8,8);assert torch.isfinite(standard[ts]).all();shapes[ts]=list(standard[ts].shape)
   try:
    g=torch.Generator(device=ctx.device).manual_seed(20264301);native_valid[ts]=ctx.predict_modular(xs,metadata=meta,timescale=ts,rng=g,latent_size=[1,2,4])[0]
   except Exception as exc:failures.append({'api':'predict_modular','timescale':ts,'error':repr(exc)})
  try:
   g=torch.Generator(device=ctx.device).manual_seed(20264302);ctx.predict_modular(xs,metadata=meta,timescale='monthly',stop_stage=1,return_internal=True,rng=g,latent_size=[1,2,4])
  except Exception as exc:failures.append({'api':'predict_modular_stop_stage1','error':repr(exc)})
 dump(O/'original_api_failures.json',{'failures':failures,'unchanged_source_sha256':sha(SOURCE)})
 repaired,_=make_context(repair=True);repaired.model=ctx.model;checks=[];prefix=[];raw={}
 with torch.no_grad():
  for ts in ['decadal','yearly','monthly']:
   g=torch.Generator(device=ctx.device).manual_seed(20264301);a=repaired.predict_modular(xs,metadata=meta,timescale=ts,rng=g,latent_size=[1,2,4])[0]
   h=torch.Generator(device=ctx.device).manual_seed(20264301);b=repaired.predict_modular(xs,metadata=meta,timescale=ts,rng=h,latent_size=[1,2,4])[0]
   replay=float((a-b).abs().max());assert replay==0
   existing=float((a-native_valid[ts]).abs().max()) if ts in native_valid else None
   if existing is not None:assert existing==0
   standard_error=float((a-standard[ts]).abs().max())
   checks.append({'timescale':ts,'replay_max_abs':replay,'original_valid_modular_max_abs':existing,'native_standard_vs_seeded_max_abs':standard_error});raw['sample_'+ts]=a.cpu().numpy()
   g=torch.Generator(device=ctx.device).manual_seed(20264303);p,_=repaired.predict_modular(xs,metadata=meta,timescale=ts,stop_stage=1,return_internal=True,rng=g,latent_size=[1,2,4]);prefix.append(p)
  prefix_error=max(float((p-prefix[0]).abs().max()) for p in prefix);assert prefix_error==0
  g=torch.Generator(device=ctx.device).manual_seed(20264304);full,_=repaired.predict_modular(xs,metadata=meta,timescale='monthly',rng=g,latent_size=[1,2,4])
  g=torch.Generator(device=ctx.device).manual_seed(20264304);p,_=repaired.predict_modular(xs,metadata=meta,timescale='monthly',stop_stage=1,return_internal=True,rng=g,latent_size=[1,2,4]);resumed,_=repaired.predict_modular(xs,metadata=meta,timescale='monthly',start_stage=1,init_latents=p,rng=g,latent_size=[1,2,4]);resume_error=float((full-resumed).abs().max());assert resume_error==0
 np.savez_compressed(O/'qualification_samples.npz',**raw,shared_prefix=prefix[0].cpu().numpy(),full=full.cpu().numpy(),resumed=resumed.cpu().numpy())
 torch.save({'state_dict':ctx.model.state_dict(),'discarded_preflight_only':True,'source_commit':'a7624ae25b47b474d5a31ca22b3ac6e6907795dd'},O/'preflight.pt')
 record.update(status='PASS_WITH_NARROW_METADATA_REPAIR',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parameters=ctx.n_params,discarded_updates=50,first_loss=losses[0],last_loss=losses[-1],runtime_seconds=elapsed,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,condition_projection_gradients=cond_grads,shapes=shapes,original_api_failures=failures,repair='indices=None initialized at predict_modular entry; sampling computations unchanged',checks=checks,shared_first_stage_max_abs=prefix_error,split_resume_max_abs=resume_error,source_method_ast_sha256=hashes,source_wrapper_sha256=sha(SOURCE),source_backbone_sha256=sha(S/'emulator/models/spatiotemporal_pyramid_flow/model.py'),plan_sha256=sha(R/'NATIVE_QUALIFICATION_PLAN.md'),samples_sha256=sha(O/'qualification_samples.npz'),checkpoint_sha256=sha(O/'preflight.pt'),comparative_effects_measured=False,old_real_test_scored=False)
 dump(O/'qualification.json',record);print(json.dumps(record),flush=True)
except Exception:
 record.update(status='FAIL',traceback=traceback.format_exc());dump(O/'qualification.json',record);raise
