from common62 import *
import subprocess,sys,os,traceback
assert not (OUT/'post_state62_v4.json').exists()
assert read(OUT/'MODEL_SEAL62_V2.json')['completed_count']==8
try:
 for name in ['score62_v4.py','replay62_v2.py','analyze62_v4.py','export62_v2.py']:
  state('post_state62_v4.json',{'status':'RUNNING','active':name,'pid':os.getpid(),'completed_fits':8,'planned_fits':9})
  with (OUT/(name.replace('.py','')+'_stdout.log')).open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(ROOT/'scripts'/name)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  assert r.returncode==0,(name,r.returncode)
 state('post_state62_v4.json',{'status':'C62_PARTIAL_REPRODUCTION_CLOSED_PUBLICATION_READY','pid':os.getpid(),'completed_fits':8,'planned_fits':9,'accepted_final_directions':0})
except Exception:
 state('post_state62_v4.json',{'status':'FAILED_INSPECT_EVIDENCE','pid':os.getpid(),'error':traceback.format_exc()});raise
