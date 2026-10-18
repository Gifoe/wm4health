from models65 import *
from checkpoint_io65_v1 import load_checkpoint
import subprocess,sys,time
pp=paths();old=read(OUT/'recovery_registry65_v1.json');assert len(old['fits'])==16;assert read(OUT/'CHECKPOINT_IO_QUALIFICATION65_V1.json')['status']=='PASS';assert not (OUT/'FITS65_V1.json').exists();fits=list(old['fits']);corpora=old['corpora'];attempts=list(old['attempts']);start=time.perf_counter()
dump(OUT/'NPZ_RECOVERY_SOURCE65_V1.json',dict(utc=utc(),previous_registry_sha256=sha(OUT/'recovery_registry65_v1.json'),previous_models=16,previous_registry=old,checkpoint_io_qualification_sha256=sha(OUT/'CHECKPOINT_IO_QUALIFICATION65_V1.json'),failed_coverage_logs={name:sha(OUT/name) for name in ['recover_nonlinear_s2023_CSDI_native_coverage_attempt1_stdout.log','recover_nonlinear_s2023_CSDI_native_coverage_attempt2_stdout.log']},discarded_original_updates_lower_bound=8000,additional_coverage_duplicated_updates_lower_bound=4000))
def run(script,args,log):
 with log.open('xb') as out:
  child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'isolated_entry65_v1.py'),script,*args],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT);state('recovery_state65_v1.json',dict(status='NPZ_CHILD_LIVE',script=script,pid=child.pid,args=args,closed_models=len(fits)));code=child.wait()
 return code
for key,kind in [('nonlinear_s2023','CSDI_native_coverage'),('nonlinear_s2023','causal_MDN')]:
 resume=None
 if kind.startswith('CSDI'):
  candidates=sorted((pp['checkpoint_root']/'recovery_v1'/key/kind).glob('attempt*/step*.pt'),key=lambda p:int(p.stem[4:]),reverse=True);assert candidates;resume=candidates[0];zz=load_checkpoint(resume,map_location='cpu');assert zz['step']==4000 and zz['kind']==kind
 success=False
 for attempt in [1,2]:
  code=run('recover_cell65_v2.py',[key,kind,str(attempt)]+([str(resume)] if resume else []),OUT/f'npz_train_{key}_{kind}_attempt{attempt}_stdout.log');attempts.append(dict(key=key,model=kind,backend='NPZ_JSON',attempt=attempt,exit_code=code,resumed_from=str(resume) if resume else None))
  if code==0:success=True;break
  candidates=sorted((pp['checkpoint_root']/'recovery_npz_v1'/key/kind).glob('attempt*/step*.npz'),key=lambda p:int(p.stem[4:]),reverse=True)
  if candidates:resume=candidates[0]
 assert success,('NPZ training failed twice; stop this strategy and inspect',key,kind)
 for attempt in [1,2]:
  code=run('recover_validate65_v2.py',[key,kind],OUT/f'npz_validate_{key}_{kind}_attempt{attempt}_stdout.log');attempts.append(dict(key=key,model=kind,stage='validation',backend='NPZ_JSON',attempt=attempt,exit_code=code))
  if code==0:break
 assert code==0,('Validation failed twice; sealed weights preserved',key,kind)
 fits.append(read(OUT/f'RECOVERED_FIT_{key}_{kind}_V1.json'));state('npz_registry65_v1.json',dict(status='PARTIAL',fits=fits,corpora=corpora,attempts=attempts))
assert len(fits)==18
dump(OUT/'FITS65_V1.json',dict(utc=utc(),status='ALL_18_MODELS_FROZEN',fits=fits,corpora=corpora,unique_training_corpora=6,models=18,optimizer_updates=144000,aborted_original_optimizer_updates_lower_bound=8000,additional_coverage_duplicated_updates_lower_bound=4000,original_successful_models_reused=12,recovered_models=6,recovery_attempts=attempts,device=read(OUT/'QUALIFICATION_ISOLATED65_V1.json')['device'],config=config(),elapsed_seconds=sum(r['elapsed_seconds'] for r in fits),elapsed_scope='Recorded per-fit/last-resume-segment runtimes; not total compute. Failure/resume traces retained separately.',npz_finish_wall_seconds=time.perf_counter()-start,actual_recovery_script='finish_npz65_v1.py',prior_coordinator='recover_campaign65_v2.py',protocol_sha256=sha(ROOT/'PROTOCOL65_V1.md'),accepted_final_directions=0));state('recovery_state65_v1.json',dict(status='ALL_18_MODELS_FROZEN',models=18,optimizer_updates=144000))
for script,log,receipt in [('evaluate65_v3.py','evaluate65_v3_stdout.log','evaluation_exit65_v3.json'),('replay65_v2.py','replay65_v2_stdout.log','replay_exit65_v2.json'),('summarize65_v1.py','summarize65_v1_stdout.log','summary_exit65_v1.json')]:
 code=run(script,[],OUT/log);dump(OUT/receipt,dict(utc=utc(),exit_code=code,log_sha256=sha(OUT/log)));assert code==0,(script,code)
state('recovery_state65_v1.json',dict(status='TRAIN_EVALUATE_REPLAY_SUMMARY_TERMINAL_SUCCESS',models=18));print(json.dumps(dict(status='TRAIN_EVALUATE_REPLAY_SUMMARY_TERMINAL_SUCCESS',models=18)),flush=True)
