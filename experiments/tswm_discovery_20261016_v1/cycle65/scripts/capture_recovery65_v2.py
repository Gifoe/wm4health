from common65 import *
import subprocess,sys
with (OUT/'recover_campaign65_v2_stdout.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'isolated_entry65_v1.py'),'recover_campaign65_v2.py'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT);state('recovery_launch65_v2.json',dict(status='CHILD_LIVE',pid=child.pid));code=child.wait()
dump(OUT/'recovery_exit65_v2.json',dict(utc=utc(),exit_code=code,pid=child.pid,log_sha256=sha(OUT/'recover_campaign65_v2_stdout.log')))
