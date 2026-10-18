"""Resume only sealed-law evaluation after a terminal native breakpoint."""
from common65 import *
import subprocess,sys
assert read(OUT/'FITS65_V1.json')['models']==18
assert read(OUT/'evaluation_exit65_v3.json')['exit_code']==2147483651
pp=paths();receipts=list((pp['artifact_root']/'law_receipts').glob('*.json'))
arrays=set((pp['artifact_root']/'laws').glob('*.npz'));claimed=set()
for p in receipts:
 r=read(p);a=pathlib.Path(r['artifact']);assert sha(a)==r['artifact_sha256'];assert r['fits_sha256']==sha(OUT/'FITS65_V1.json');claimed.add(a)
assert arrays==claimed,('Orphan/missing law array',len(arrays),len(claimed))
dump(OUT/'EVALUATION_RESUME65_V1.json',dict(utc=utc(),status='HASH_VERIFIED_RESUME',verified_laws=len(receipts),original_exit_code=2147483651,original_log_sha256=sha(OUT/'evaluate65_v3_stdout.log'),algorithm_change=False,original_laws_preserved=True,max_evaluation_attempts=2))
def run(script,stem):
 log=OUT/(stem+'_stdout.log')
 with log.open('xb') as out:
  child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'isolated_entry65_v1.py'),script],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
  state('evaluation_resume_state65_v1.json',dict(status='CHILD_LIVE',script=script,pid=child.pid));code=child.wait()
 dump(OUT/(stem+'_exit.json'),dict(utc=utc(),script=script,exit_code=code,pid=child.pid,log_sha256=sha(log)))
 return code
for attempt in [1,2]:
 code=run('evaluate65_v3.py',f'evaluation_resume65_attempt{attempt}_v1')
 if code==0:break
assert code==0,('Native evaluation failed twice; stop this runtime strategy',code)
for script,stem in [('replay65_v2.py','replay_resume65_v1'),('summarize65_v1.py','summary_resume65_v1')]:
 assert run(script,stem)==0,('Replay/summary failed; preserve frozen results',script)
state('recovery_state65_v1.json',dict(status='TRAIN_EVALUATE_REPLAY_SUMMARY_TERMINAL_SUCCESS',models=18,actual_evaluation_coordinator='capture_evaluation_resume65_v1.py'))
state('evaluation_resume_state65_v1.json',dict(status='TERMINAL_SUCCESS',models=18))
