"""Queue verification/export on the actual numerical worker, not elapsed time."""
import sys,pathlib,json,time,subprocess,datetime,ctypes
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs'
def dump(name,x):
 with (out/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 exit_path=out/'run83_v1_exit.json';launch=json.loads((out/'run83_v1_launch.json').read_text());kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;handle=kernel.OpenProcess(0x100000,False,launch['pid']);assert handle or exit_path.exists();dump('close83_v1_launch.json',dict(utc=now(),worker_pid=launch['pid'],queue_pid=__import__('os').getpid()))
 try:
  while not exit_path.exists():
   if handle and kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==0:time.sleep(2);assert exit_path.exists(),'Worker terminated without captured exit';break
   time.sleep(20)
 finally:
  if handle:kernel.CloseHandle(ctypes.c_void_p(handle))
 assert json.loads(exit_path.read_text())['exit_code']==0
 for script in ['verify83_v1.py','export83_v1.py']:
  log=out/(script[:-3]+'_stdout.log')
  with log.open('xb') as f:code=subprocess.call([sys.executable,'-I','-B','-u',str(root/'scripts'/script)],stdout=f,stderr=subprocess.STDOUT,cwd=root)
  dump(script[:-3]+'_exit.json',dict(utc=now(),exit_code=code));assert code==0,(script,code)
 dump('close83_v1_exit.json',dict(utc=now(),status='VERIFIED_SERVER_SNAPSHOT_READY',local_payload_files_written=0))
if __name__=='__main__':main()
