"""Finish independent C44 scores, then qualify the next native baseline once."""
import pathlib,sys,os,json,datetime,subprocess,traceback,ctypes
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(exist_ok=True);statefile=O/'supervisor_v1.json';assert not statefile.exists();state={'status':'waiting_for_C44_query_timings','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':[]}
def save():statefile.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32;kernel.CloseHandle.argtypes=[ctypes.c_void_p];handle=kernel.OpenProcess(0x00100000,False,38216)
 if handle:
  try:
   while kernel.WaitForSingleObject(handle,1000)==0x102:pass
  finally:kernel.CloseHandle(handle)
 query=R.parent/'cycle44';summary=json.loads((query/'outputs/query_execution_v1/execution_summary.json').read_text());assert summary['status']=='PASS' and summary['models']==16
 for script in [query/'scripts/analyze_query44.py',R/'scripts/qualify_native45.py']:
  state['status']='running';record={'script':script.relative_to(R.parent).as_posix(),'status':'running'};state['scripts'].append(record)
  with (O/(script.stem+'.stdout.log')).open('wb') as out,(O/(script.stem+'.stderr.log')).open('wb') as err:
   child=subprocess.Popen([sys.executable,'-B','-u',str(script)],cwd=script.parent.parent,stdout=out,stderr=err);record['pid']=child.pid;save();code=child.wait()
  record.update(status='completed' if code==0 else 'failed',exit_code=code);save();assert code==0,(str(script),code)
 state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
