"""Wait on existing held-out evaluator, then qualify and execute fixed queries."""
import pathlib,sys,os,json,datetime,subprocess,traceback,ctypes
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';statefile=O/'query_execution_supervisor_v1.json';assert not statefile.exists();state={'status':'waiting_for_quality_and_truth','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':[]}
def save():statefile.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32;kernel.CloseHandle.argtypes=[ctypes.c_void_p];handle=kernel.OpenProcess(0x00100000,False,35812)
 if handle:
  try:
   while kernel.WaitForSingleObject(handle,1000)==0x102:pass
  finally:kernel.CloseHandle(handle)
 quality=json.loads((O/'quality_verification_v1.json').read_text());truth=json.loads((O/'reference_eval_v1/reference_eval.json').read_text());assert quality['status']=='PASS' and truth['status']=='REFERENCE_READY'
 for name in ['qualify_execution44.py','query_execution44.py']:
  state['status']='running';record={'script':name,'status':'running'};state['scripts'].append(record)
  with (O/(name.removesuffix('.py')+'.stdout.log')).open('wb') as out,(O/(name.removesuffix('.py')+'.stderr.log')).open('wb') as err:
   child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=out,stderr=err);record['pid']=child.pid;save();code=child.wait()
  record.update(status='completed' if code==0 else 'failed',exit_code=code);save();assert code==0,(name,code)
 state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
