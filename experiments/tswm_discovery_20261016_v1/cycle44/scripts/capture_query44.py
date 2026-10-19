"""Wait on the existing trainer, then qualify and calibrate queries once."""
import pathlib,sys,os,json,datetime,subprocess,traceback,time,ctypes
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';statefile=O/'query_supervisor_v1.json';assert not statefile.exists()
state={'status':'waiting_for_training','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':[],'eval_opened':False}
def save():statefile.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 # Windows process handle remains attached to this process even if its PID is reused.
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32;kernel.CloseHandle.argtypes=[ctypes.c_void_p]
 handle=kernel.OpenProcess(0x00100000,False,35344)
 if handle:
  try:
   while kernel.WaitForSingleObject(handle,1000)==0x102:pass
  finally:kernel.CloseHandle(handle)
 training=json.loads((O/'fits_v1/training_summary.json').read_text());assert training['status']=='PASS' and training['fits']==16 and not training['eval_opened']
 for name in ['qualify_query44.py','pilot44.py']:
  state['status']='running';record={'script':name,'status':'running'};state['scripts'].append(record)
  with (O/(name.removesuffix('.py')+'.stdout.log')).open('wb') as out,(O/(name.removesuffix('.py')+'.stderr.log')).open('wb') as err:
   child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=out,stderr=err);record['pid']=child.pid;save();code=child.wait()
  record.update(status='completed' if code==0 else 'failed',exit_code=code);save();assert code==0,(name,code)
 state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
