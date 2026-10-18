from common64 import *
import subprocess,sys,os,traceback
assert not (OUT/'campaign_state64_v1.json').exists()
try:
 for name in ['qualify64_v1.py','fit64_v2.py']:
  state('campaign_state64_v1.json',{'status':'RUNNING','active':name,'pid':os.getpid()})
  with (OUT/(name.replace('.py','')+'_stdout.log')).open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(ROOT/'scripts'/name)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  assert r.returncode==0,(name,r.returncode)
 state('campaign_state64_v1.json',{'status':'ORACLE_QUALIFIED_GAUSSIAN_FITS_SCORED','pid':os.getpid()})
except Exception:
 state('campaign_state64_v1.json',{'status':'FAILED_INSPECT_LOG','pid':os.getpid(),'error':traceback.format_exc()});raise
