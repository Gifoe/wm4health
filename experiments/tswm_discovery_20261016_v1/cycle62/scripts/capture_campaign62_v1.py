from common62 import *
import subprocess,sys,time,os,traceback
assert not (OUT/'campaign_state62_v1.json').exists()
state('campaign_state62_v1.json',{'status':'RUNNING','pid':os.getpid(),'completed':[]})
done=[]
try:
 for s in [2021,2022,2023]:
  for a in ['Informer','Informer_PRO','Informer_DYN']:
   key=a+'_'+str(s);state('campaign_state62_v1.json',{'status':'RUNNING','pid':os.getpid(),'active':key,'completed':done})
   with (OUT/(key+'_stdout62_v1.log')).open('x',encoding='utf-8') as f:
    r=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/train62_v1.py'),a,str(s)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   assert r.returncode==0,(key,r.returncode)
   receipt=read(OUT/(key+'_fit62_v1.json'));assert receipt['status']=='SEALED_FOR_TEST';done.append(key)
 state('campaign_state62_v1.json',{'status':'ALL_NINE_NATIVE_FITS_SEALED','pid':os.getpid(),'completed':done,'test_opened':False})
except Exception:
 state('campaign_state62_v1.json',{'status':'FAILED_INSPECT_EVIDENCE','pid':os.getpid(),'completed':done,'error':traceback.format_exc()});raise
