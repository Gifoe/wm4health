from common62 import *
import subprocess,sys,os
state('loader_qualification_state62_v2.json',{'status':'RUNNING','pid':os.getpid()})
with (OUT/'qualify_cached_loader62_v1_stdout.log').open('x',encoding='utf-8') as f:
 r=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(ROOT/'scripts/qualify_cached_loader62_v1.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
state('loader_qualification_state62_v2.json',{'status':'PASS' if r.returncode==0 else 'FAILED','exit_code':r.returncode,'pid':os.getpid()})
