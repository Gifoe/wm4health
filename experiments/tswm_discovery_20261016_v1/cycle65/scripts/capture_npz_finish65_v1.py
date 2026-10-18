from common65 import *
import subprocess,sys
with (OUT/'finish_npz65_v1_stdout.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'isolated_entry65_v1.py'),'finish_npz65_v1.py'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT);state('npz_finish_launch65_v1.json',dict(status='CHILD_LIVE',pid=child.pid));code=child.wait()
dump(OUT/'npz_finish_exit65_v1.json',dict(utc=utc(),exit_code=code,pid=child.pid,log_sha256=sha(OUT/'finish_npz65_v1_stdout.log')))
