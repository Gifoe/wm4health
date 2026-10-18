from common65 import *
import subprocess,sys,os
log=OUT/'train65_v2_stdout.log'
with log.open('xb') as f:
 child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/'train65_v2.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 state('train_launch65_v1.json',{'status':'CHILD_LIVE','pid':child.pid});code=child.wait()
dump(OUT/'train_exit65_v1.json',{'utc':utc(),'pid':child.pid,'exit_code':code,'status':'TRAINING_CLOSED' if code==0 else 'TRAINING_FAILED','log_sha256':sha(log)})
