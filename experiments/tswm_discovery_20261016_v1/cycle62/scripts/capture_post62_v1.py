from common62 import *
import subprocess,sys,os,traceback,time
assert not (OUT/'post_state62_v1.json').exists()
state('post_state62_v1.json',{'status':'WAITING_FOR_NINE_FITS','pid':os.getpid()})
try:
 while True:
  c=read(OUT/'campaign_state62_v1.json')
  if c['status']=='ALL_NINE_NATIVE_FITS_SEALED':break
  assert c['status']=='RUNNING',c
  time.sleep(10)
 for name in ['score62_v1.py','replay62_v1.py','analyze62_v1.py','export62_v1.py']:
  state('post_state62_v1.json',{'status':'RUNNING','active':name,'pid':os.getpid()})
  with (OUT/(name.replace('.py','')+'_stdout.log')).open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  assert r.returncode==0,(name,r.returncode)
 state('post_state62_v1.json',{'status':'C62_CLOSED_PUBLICATION_READY','pid':os.getpid(),'accepted_final_directions':0})
except Exception:
 state('post_state62_v1.json',{'status':'FAILED_INSPECT_EVIDENCE','pid':os.getpid(),'error':traceback.format_exc()});raise
