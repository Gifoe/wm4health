from common63 import *
import subprocess,sys,os,traceback
assert not (OUT/'audit_state63_v1.json').exists()
dump(OUT/'audit_launch63_v1.json',{'utc':utc(),'pid':os.getpid(),'command':'audit63_v1.py','status':'LAUNCHED'})
with (OUT/'audit63_v1_stdout.log').open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(ROOT/'scripts/audit63_v1.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
dump(OUT/'audit_state63_v1.json',{'utc':utc(),'pid':os.getpid(),'exit_code':r.returncode,'status':'NATIVE_AUDIT_READY_FOR_REPLAY' if r.returncode==0 else 'FAILED_INSPECT_LOG'})
