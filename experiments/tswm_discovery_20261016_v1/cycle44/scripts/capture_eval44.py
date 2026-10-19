"""Sequential independent evaluation after the existing VAL supervisor exits."""
import pathlib,sys,os,json,datetime,subprocess,traceback,ctypes
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';statefile=O/'eval_supervisor_v1.json';assert not statefile.exists();state={'status':'waiting_for_pilot','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':[],'eval_opened':False}
def save():statefile.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32;kernel.CloseHandle.argtypes=[ctypes.c_void_p];handle=kernel.OpenProcess(0x00100000,False,42140)
 if handle:
  try:
   while kernel.WaitForSingleObject(handle,1000)==0x102:pass
  finally:kernel.CloseHandle(handle)
 pilot=json.loads((O/'pilot_v1/pilot_summary.json').read_text());assert pilot['status']=='PASS' and len(pilot['allocation_rows'])==240
 for name in ['quality44.py','verify_quality44.py','reference_eval44.py']:
  state.update(status='running',eval_opened=True);record={'script':name,'status':'running'};state['scripts'].append(record)
  with (O/(name.removesuffix('.py')+'.stdout.log')).open('wb') as out,(O/(name.removesuffix('.py')+'.stderr.log')).open('wb') as err:
   child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=out,stderr=err);record['pid']=child.pid;save();code=child.wait()
  record.update(status='completed' if code==0 else 'failed',exit_code=code);save();assert code==0,(name,code)
 state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
