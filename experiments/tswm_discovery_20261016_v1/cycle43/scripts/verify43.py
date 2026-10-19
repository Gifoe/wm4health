"""Independent schedule algebra, retained samples, and checkpoint API replay."""
from harness43_v2 import *
import datetime
O=R/'outputs/native_v3';q=json.loads((O/'qualification.json').read_text());assert q['status']=='PASS_WITH_NARROW_METADATA_REPAIR'
assert sha(O/'preflight.pt')==q['checkpoint_sha256'] and sha(O/'qualification_samples.npz')==q['samples_sha256']
for x in json.loads((R/'runtime_source/spf/ADAPTER_MANIFEST.json').read_text())['files']:
 assert sha(R/'runtime_source/spf'/x['path'])==x['sha256']
 if not x['import_only_change']:assert x['sha256']==x['original_sha256']
# NumPy-only reconstruction from declared ns and stage boundaries, not native code.
global_sigmas=np.arange(1000,0,-1,dtype=np.float32)/np.float32(1000);global_t=global_sigmas*np.float32(1000)
edges=[0,330,660,1000];ns=[[4,4],[4,4],[4,64]];max_schedule=0.;stage0_end=[]
for level in range(3):
 distances=[]
 for stage in range(3):
  start=float(global_sigmas[edges[stage]]);end=float(global_sigmas[edges[stage+1]]) if stage<2 else 0.
  if stage:start=start*np.sqrt(ns[level][stage-1])/(1-start+start*np.sqrt(ns[level][stage-1]))
  distances.append(start-end)
 for stage in range(3):
  begin=int(sum(distances[:stage])/sum(distances)*1000);end=min(int(sum(distances[:stage+1])/sum(distances)*1000),999)
  reconstructed=np.linspace(global_t[begin],global_t[end],1001)[:-1]
  native=np.asarray(q['training_schedule_by_temporal_level'][str(level)][str(stage)])
  assert len(native)==1000;max_schedule=max(max_schedule,float(np.max(np.abs(native-reconstructed))))
  if stage==0:stage0_end.append(float(native[-1]))
assert max_schedule<1e-9,max_schedule
raw=np.load(O/'qualification_samples.npz');resume=float(np.max(np.abs(raw['full']-raw['resumed'])));assert resume==0
ctx,_=make_context(repair=True);checkpoint=torch.load(O/'preflight.pt',map_location='cpu',weights_only=False);ctx.model.load_state_dict(checkpoint['state_dict']);ctx.model.eval();xs,metadata=prediction_inputs(toy_examples()[:2],ctx.device)
prefix=[]
with torch.no_grad():
 for ts in ['decadal','yearly','monthly']:
  g=torch.Generator(device=ctx.device).manual_seed(20264303);p,_=ctx.predict_modular(xs,metadata=metadata,timescale=ts,stop_stage=1,return_internal=True,rng=g,latent_size=[1,2,4]);prefix.append(p.cpu().numpy())
 g=torch.Generator(device=ctx.device).manual_seed(20264304);full,_=ctx.predict_modular(xs,metadata=metadata,timescale='monthly',rng=g,latent_size=[1,2,4])
 g=torch.Generator(device=ctx.device).manual_seed(20264304);p,_=ctx.predict_modular(xs,metadata=metadata,timescale='monthly',stop_stage=1,return_internal=True,rng=g,latent_size=[1,2,4]);resumed,_=ctx.predict_modular(xs,metadata=metadata,timescale='monthly',start_stage=1,init_latents=p,rng=g,latent_size=[1,2,4])
delta=max(float(np.max(np.abs(p-prefix[0]))) for p in prefix);replay=float(np.max(np.abs(full.cpu().numpy()-raw['full'])));within_resume=float((full-resumed).abs().max())
assert abs(delta-q['shared_first_stage_max_abs'])<2e-7 and replay<2e-6 and within_resume==0,(delta,replay,within_resume)
np.savez_compressed(O/'independent_api_replay.npz',prefix_decadal=prefix[0],prefix_yearly=prefix[1],prefix_monthly=prefix[2],full=full.cpu().numpy(),resumed=resumed.cpu().numpy())
v={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independent_scheduler_values':9000,'max_schedule_abs':max_schedule,'stage0_training_timestep_end_by_level':stage0_end,'native_retained_full_resume_max_abs':resume,'checkpoint_replay_max_abs':replay,'independent_within_level_resume_max_abs':within_resume,'cross_level_prefix_max_abs':delta,'cross_level_prefix_is_identical':delta==0,'cross_process_replay_tolerance':2e-6,'qualification_sha256':sha(O/'qualification.json'),'checkpoint_sha256':q['checkpoint_sha256'],'replay_raw_sha256':sha(O/'independent_api_replay.npz'),'no_fresh_training':True,'comparative_effects_measured':False,'old_real_test_scored':False,'verifier_sha256':sha(pathlib.Path(__file__))}
dump(R/'outputs/verification_v1.json',v);print(json.dumps(v),flush=True)
