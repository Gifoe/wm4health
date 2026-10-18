"""Wait on the verified campaign owner, then score and close one sealed recipe."""
from common61 import *
import ctypes,time,sys,subprocess,traceback
statefile=OUT/'post_state61_v1.json'
def state(status,**kw):
 p=statefile.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2));p.replace(statefile)
def live(pid):
 fn=ctypes.windll.kernel32.OpenProcess;fn.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong];fn.restype=ctypes.c_void_p;h=fn(0x1000,False,pid)
 if not h:return False
 try:
  v=ctypes.c_ulong();f=ctypes.windll.kernel32.GetExitCodeProcess;f.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];assert f(h,ctypes.byref(v));return v.value==259
 finally:
  f=ctypes.windll.kernel32.CloseHandle;f.argtypes=[ctypes.c_void_p];f(h)
try:
 assert not statefile.exists() and not (OUT/'TEST_OPEN61_V1.json').exists()
 for name in ['score61_v1.py','analyze61_v1.py','export61_v1.py']:assert (ROOT/'scripts'/name).exists()
 owner=read(OUT/'campaign_state61_v1.json')['pid'];assert owner==27844
 while True:
  s=read(OUT/'campaign_state61_v1.json');assert s['pid']==owner
  if s['status']=='ALL_TWELVE_MODELS_SEALED_FOR_INDEPENDENT_SCORING':break
  assert s['status']=='RUNNING_ALL_PREDECLARED_ATTEMPTS' and live(owner),s
  state('WAITING_ON_VERIFIED_FIXED_CAMPAIGN_OWNER',owner_pid=owner);time.sleep(15)
 for name in ['score61_v1.py','analyze61_v1.py','export61_v1.py']:
  with (OUT/(name+'.stdout.log')).open('xb') as o,(OUT/(name+'.stderr.log')).open('xb') as e:
   child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],stdout=o,stderr=e,cwd=ROOT);state('RUNNING_SEALED_POSTPROCESSING',active_script=name,child_pid=child.pid);code=child.wait()
  assert code==0,(name,code)
 state('C61_CLOSED_RESULTS_AND_PUBLICATION_SNAPSHOT_READY')
except Exception:
 state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
