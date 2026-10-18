"""Three CPU workers, all twelve attempts, terminal sealing before scoring."""
from common61 import *
import subprocess,sys,time,traceback
statefile=OUT/'campaign_state61_v1.json';completed=[];failed=[];running={}
def state(status,**kw):
 p=statefile.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,'completed':completed,'failed':failed,'active':[{'tag':k,'pid':v[0].pid} for k,v in running.items()],**kw},indent=2));p.replace(statefile)
try:
 assert not statefile.exists()
 assert read(OUT/'preflight_state61_v2.json')['status']=='NATIVE_COMPONENT_QUALIFICATION_READY'
 assert read(OUT/'QUALIFICATION61_V2.json')['chosen_device']=='cpu'
 assert not (OUT/'DATA61_V1.json').exists()
 with (OUT/'prepare61_v1.stdout.log').open('xb') as o,(OUT/'prepare61_v1.stderr.log').open('xb') as e:
  p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/prepare61_v1.py')],stdout=o,stderr=e,cwd=ROOT);state('PREPARING_UNLABELED_DATA',data_child_pid=p.pid);assert p.wait()==0
 jobs=[(family,arm,seed) for family in ['peaked','bimodal'] for arm in ['hsmm','hmm'] for seed in [2021,2022,2023]]
 dump(OUT/'TRAINING_PLAN61_V1.json',{'utc':utc(),'jobs':jobs,'updates_per_fit':1500,'workers':3,'threads_per_worker':4,'device':'cpu','protocol_sha256':sha(ROOT/'PROTOCOL61_V1.md'),'qualification_sha256':sha(OUT/'QUALIFICATION61_V2.json'),'data_sha256':sha(OUT/'DATA61_V1.json'),'test_scored':False})
 queue=list(jobs)
 while queue or running:
  while queue and len(running)<3:
   family,arm,seed=queue.pop(0);tag=f'{family}_{arm}_seed{seed}';o=(OUT/(tag+'.stdout.log')).open('xb');e=(OUT/(tag+'.stderr.log')).open('xb');p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/train61_v1.py'),family,arm,str(seed)],stdout=o,stderr=e,cwd=ROOT);running[tag]=(p,o,e);state('RUNNING_ALL_PREDECLARED_ATTEMPTS')
  for tag,(p,o,e) in list(running.items()):
   code=p.poll()
   if code is None:continue
   o.close();e.close();rec=OUT/'runs'/tag/'result.json';assert rec.exists(),(tag,'Missing fit receipt',code);r=read(rec)
   if code==3:
    assert r['status']=='FAILED_NONFINITE_PRESERVED';failed.append({'tag':tag,'result_sha256':sha(rec),'status':r['status']})
   else:
    assert code==0 and r['status']=='COMPLETE_FIXED_BUDGET_NATIVE_DURATION_CONTROL' and r['updates_completed']==1500,(tag,code,r['status']);assert sha(r['selected']['checkpoint'])==r['selected']['checkpoint_sha256'];completed.append({'tag':tag,'result_sha256':sha(rec),'selected':r['selected']})
   del running[tag];state('RUNNING_ALL_PREDECLARED_ATTEMPTS')
  time.sleep(2)
 dump(OUT/'MODEL_SEAL61_V1.json',{'utc':utc(),'status':'ALL_TWELVE_EXISTING_MODEL_ATTEMPTS_TERMINAL_SEALED','models':completed,'failed':failed,'training_plan_sha256':sha(OUT/'TRAINING_PLAN61_V1.json'),'test_scored':False,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0})
 state('ALL_TWELVE_MODELS_SEALED_FOR_INDEPENDENT_SCORING')
except Exception:
 state('FAILED_IMPLEMENTATION_PRESERVED',traceback=traceback.format_exc());raise
