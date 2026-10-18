from common64 import *
import subprocess,sys,os
assert not (OUT/'sampler_launch64_v1.json').exists()
dump(OUT/'sampler_launch64_v1.json',{'utc':utc(),'pid':os.getpid(),'status':'LAUNCHED'})
with (OUT/'samplers64_v1_stdout.log').open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(ROOT/'scripts/samplers64_v1.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
dump(OUT/'sampler_exit64_v1.json',{'utc':utc(),'pid':os.getpid(),'exit_code':r.returncode,'status':'SAMPLERS_CLOSED' if r.returncode==0 else 'FAILED_INSPECT_LOG'})
