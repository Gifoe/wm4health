from models65 import *
import subprocess,sys,time,faulthandler
faulthandler.enable(all_threads=True);pp=paths();old=read(OUT/'fit_registry65_v1.json');assert len(old['fits'])==12;assert read(OUT/'train_exit65_v1.json')['exit_code']==3221225477;assert not (OUT/'FITS65_V1.json').exists();assert not (OUT/'EVALUATION65_V1.json').exists();dump(OUT/'RECOVERY_SOURCE65_V1.json',dict(utc=utc(),original_registry=old,original_registry_sha256=sha(OUT/'fit_registry65_v1.json'),original_exit=read(OUT/'train_exit65_v1.json'),invalid_checkpoint='nonlinear_s2022_csdi_random_step8000.pt, zero bytes',root_cause='Unknown; Windows event reports python310.dll offset0x195f83',stale_dependency_processes_verified_missing=True));fits=list(old['fits']);corpora=[];attempts=[];start=time.perf_counter()
for mech in MECHANISMS:
 for seed in SEEDS:
  key=f'{mech}_s{seed}';p=pp['data_root']/(key+'.npz');generated={name:generate(N,mech,seed,idx) for name,N,idx in [('train',16384,0),('validation',4096,1),('observational_test',8192,2)]}
  if p.exists():
   z=np.load(p);assert all(np.array_equal(z[n],v) for n,v in generated.items())
  else:np.savez_compressed(p,**generated)
  corpora.append(dict(key=key,mechanism=mech,seed=seed,path=str(p),sha256=sha(p),training_records=16384,validation_records=4096,reserved_observational_test_records=8192))
for fit in fits:
 assert sha(next(x['path'] for x in corpora if x['key']==fit['key']))==fit['data_sha256']
 for cp in fit['checkpoints']:assert sha(cp['path'])==cp['sha256']
def run(script,args,log):
 with log.open('xb') as out:
  child=subprocess.Popen([sys.executable,'-B','-X','faulthandler','-u',str(ROOT/'scripts'/script),*args],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT);state('recovery_state65_v1.json',dict(status='CHILD_LIVE',script=script,pid=child.pid,args=args,closed_models=len(fits)));code=child.wait()
 return code
for corpus in corpora:
 for kind in ['CSDI_native_random','CSDI_native_coverage','causal_MDN']:
  key=corpus['key']
  if any(r['key']==key and r['model']==kind for r in fits):continue
  resume=None;success=False
  for attempt in [1,2]:
   code=run('recover_cell65_v1.py',[key,kind,str(attempt)]+([str(resume)] if resume else []),OUT/f'recover_{key}_{kind}_attempt{attempt}_stdout.log');attempts.append(dict(key=key,model=kind,attempt=attempt,exit_code=code,resumed_from=str(resume) if resume else None))
   if code==0:success=True;break
   candidates=sorted((pp['checkpoint_root']/'recovery_v1'/key/kind).glob('attempt*/step*.pt'),key=lambda p:int(p.stem[4:]),reverse=True)
   for candidate in candidates:
    try:
     zz=torch.load(candidate,map_location='cpu',weights_only=False);assert zz['step']<8000 and 'rng_cpu' in zz;resume=candidate;break
    except Exception:continue
  assert success,('Recovery training failed twice; inspect exact child logs',key,kind)
  for vtry in [1,2]:
   code=run('recover_validate65_v1.py',[key,kind],OUT/f'validate_{key}_{kind}_attempt{vtry}_stdout.log');attempts.append(dict(key=key,model=kind,stage='validation',attempt=vtry,exit_code=code))
   if code==0:break
  assert code==0,('Native validation failed twice; frozen checkpoints retained',key,kind)
  fit=read(OUT/f'RECOVERED_FIT_{key}_{kind}_V1.json');fits.append(fit);state('recovery_registry65_v1.json',dict(status='PARTIAL',fits=fits,corpora=corpora,attempts=attempts))
assert len(fits)==18
dump(OUT/'FITS65_V1.json',dict(utc=utc(),status='ALL_18_MODELS_FROZEN',fits=fits,corpora=corpora,unique_training_corpora=6,models=18,optimizer_updates=144000,aborted_original_optimizer_updates_lower_bound=8000,original_successful_models_reused=12,recovered_models=6,recovery_attempts=attempts,device=read(OUT/'QUALIFICATION65_V1.json')['device'],config=config(),elapsed_seconds=sum(x['elapsed_seconds'] for x in fits),elapsed_scope='Sum of recorded per-fit runtimes; recovery resume segments, if any, listed separately. Not an equal-compute claim.',recovery_wall_seconds=time.perf_counter()-start,protocol_sha256=sha(ROOT/'PROTOCOL65_V1.md'),actual_recovery_script='recover_campaign65_v1.py',accepted_final_directions=0));state('recovery_state65_v1.json',dict(status='ALL_18_MODELS_FROZEN',models=18,optimizer_updates=144000))
for script,log,receipt in [('evaluate65_v2.py','evaluate65_v2_stdout.log','evaluation_exit65_v2.json'),('replay65_v1.py','replay65_v1_stdout.log','replay_exit65_v1.json'),('summarize65_v1.py','summarize65_v1_stdout.log','summary_exit65_v1.json')]:
 code=run(script,[],OUT/log);dump(OUT/receipt,dict(utc=utc(),exit_code=code,log_sha256=sha(OUT/log)));assert code==0,('Post-training child failed',script,code)
state('recovery_state65_v1.json',dict(status='TRAIN_EVALUATE_REPLAY_SUMMARY_TERMINAL_SUCCESS',models=18));print(json.dumps(dict(status='TRAIN_EVALUATE_REPLAY_SUMMARY_TERMINAL_SUCCESS',models=18)),flush=True)
