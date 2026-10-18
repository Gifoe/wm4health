from common62 import *
import subprocess,sys,os,traceback,shutil
old=read(OUT/'campaign_state62_v1.json');assert old['status']=='FAILED_INSPECT_EVIDENCE' and old['completed']==['Informer_2021','Informer_PRO_2021','Informer_DYN_2021','Informer_2022']
assert not (OUT/'campaign_state62_v2.json').exists()
progress=read(OUT/'fit_progress62_v1.json');assert progress['key']=='Informer_PRO_2022'
dump(OUT/'failed_attempt62_v1_progress.json',progress)
dump(OUT/'RUNTIME_FAILURE62_V1.json',{'utc':utc(),'failed_arm':'Informer_PRO','seed':2022,'exit_status':3221225477,'windows_exception':'0xc0000005','windows_reported_fault_module':'VCRUNTIME140.dll','root_cause':'unknown','last_completed_epoch':progress['history'][-1]['epoch'],'last_recorded_updates':progress['history'][-1]['training_forward_batches'],'exact_aborted_attempt_updates':'unavailable; last completed epoch lower bound only','test_opened':False,'amendment_sha256':sha(ROOT/'RUNTIME_AMENDMENT62_V1.md')})
done=old['completed'].copy()
try:
 for s in [2021,2022,2023]:
  for a in ['Informer','Informer_PRO','Informer_DYN']:
   key=a+'_'+str(s)
   if key in done:assert read(OUT/(key+'_fit62_v1.json'))['status']=='SEALED_FOR_TEST';continue
   state('campaign_state62_v2.json',{'status':'RUNNING','pid':os.getpid(),'active':key,'completed':done})
   with (OUT/(key+'_stdout62_v2.log')).open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(ROOT/'scripts/train62_v1.py'),a,str(s)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
   assert r.returncode==0,(key,r.returncode)
   assert read(OUT/(key+'_fit62_v1.json'))['status']=='SEALED_FOR_TEST';done.append(key)
 state('campaign_state62_v2.json',{'status':'ALL_NINE_NATIVE_FITS_SEALED','pid':os.getpid(),'completed':done,'test_opened':False,'aborted_native_attempts':1,'retry_same_seed_same_algorithm':True})
except Exception:
 state('campaign_state62_v2.json',{'status':'FAILED_INSPECT_EVIDENCE','pid':os.getpid(),'completed':done,'error':traceback.format_exc()});raise
