from common62 import *
import subprocess,sys,os,traceback
assert read(OUT/'campaign_state62_v2.json')['status']=='FAILED_INSPECT_EVIDENCE'
assert read(OUT/'CACHED_LOADER_QUALIFICATION62_V1.json')['status']=='PASS'
assert not (OUT/'campaign_state62_v3.json').exists()
progress=read(OUT/'fit_progress62_v1.json');assert progress['key']=='Informer_PRO_2022'
dump(OUT/'failed_attempt62_v2_progress.json',progress)
dump(OUT/'RUNTIME_FAILURE62_V2.json',{'utc':utc(),'failed_arm':'Informer_PRO','seed':2022,'exit_status':3221225477,'last_completed_epoch':progress['history'][-1]['epoch'],'last_recorded_updates':progress['history'][-1]['training_forward_batches'],'exact_aborted_attempt_updates':'unavailable; completed-epoch lower bound only','faulthandler_frame':'native baselines/utils.py line270; torch.tensor(seq_x.values,float32)','test_opened':False,'cached_loader_qualification_sha256':sha(OUT/'CACHED_LOADER_QUALIFICATION62_V1.json')})
done=read(OUT/'campaign_state62_v2.json')['completed'].copy()
try:
 for s in [2021,2022,2023]:
  for a in ['Informer','Informer_PRO','Informer_DYN']:
   key=a+'_'+str(s)
   if key in done:assert read(OUT/(key+'_fit62_v1.json'))['status']=='SEALED_FOR_TEST';continue
   state('campaign_state62_v3.json',{'status':'RUNNING','pid':os.getpid(),'active':key,'completed':done,'missing_fit_loader':'qualified_cached_tensor_v1'})
   with (OUT/(key+'_stdout62_v3.log')).open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(ROOT/'scripts/train62_v2.py'),a,str(s)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
   assert r.returncode==0,(key,r.returncode)
   assert read(OUT/(key+'_fit62_v1.json'))['status']=='SEALED_FOR_TEST';done.append(key)
 state('campaign_state62_v3.json',{'status':'ALL_NINE_NATIVE_FITS_SEALED','pid':os.getpid(),'completed':done,'test_opened':False,'aborted_native_attempts':2,'native_loader_fits':4,'qualified_cached_loader_fits':5})
except Exception:
 state('campaign_state62_v3.json',{'status':'FAILED_INSPECT_EVIDENCE','pid':os.getpid(),'completed':done,'error':traceback.format_exc()});raise
