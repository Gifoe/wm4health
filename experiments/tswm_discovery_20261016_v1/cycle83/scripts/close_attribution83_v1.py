"""Continue the existing C83 execution with oracle diagnostics after verification."""
import sys,pathlib,json,time,subprocess,datetime,ctypes,os
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs'
def dump(name,x):
 with (out/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 target=out/'VERIFICATION83_V1.json';launch=json.loads((out/'close83_v1_launch.json').read_text());kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;handle=kernel.OpenProcess(0x100000,False,launch['queue_pid']);assert handle or target.exists();dump('close_attribution83_v1_launch.json',dict(utc=now(),prior_queue_pid=launch['queue_pid'],pid=os.getpid()))
 try:
  while not target.exists():
   if handle and kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==0:time.sleep(2);assert target.exists(),'Verification queue terminated without PASS artifact';break
   time.sleep(20)
 finally:
  if handle:kernel.CloseHandle(ctypes.c_void_p(handle))
 assert json.loads(target.read_text())['status'].startswith('PASS_')
 log=out/'attribute83_v1_stdout.log'
 with log.open('xb') as f:code=subprocess.call([sys.executable,'-I','-B','-u',str(root/'scripts/attribute83_v1.py')],stdout=f,stderr=subprocess.STDOUT,cwd=root)
 dump('attribute83_v1_exit.json',dict(utc=now(),exit_code=code));assert code==0
 dump('close_attribution83_v1_exit.json',dict(utc=now(),status='ORACLE_ATTRIBUTION_READY_FOR_CHECKING'))
if __name__=='__main__':main()
