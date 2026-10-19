"""Capture native exit code and logs; never duplicate an existing training run."""
import runtime_profile
import datetime,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def dump(p,o):p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def main():
 tag=os.environ.get('TSWM_CHILD_RUN','child_native_A');assert '/' not in tag and '\\' not in tag and tag not in ['.','..']
 out=ROOT/'outputs'/tag;out.mkdir(parents=True,exist_ok=True)
 if (out/'process.json').exists():raise RuntimeError('Already launched: inspect exact process/handle before any restart')
 with (out/'stdout.log').open('w',encoding='utf-8') as log:
  argv=[sys.executable,'-B','-u',str(ROOT/'scripts/run_child_native_A.py')]
  proc=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
  dump(out/'process.json',{'pid':proc.pid,'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  last=''
  while proc.poll() is None:
   time.sleep(5);lines=(out/'stdout.log').read_text(encoding='utf-8',errors='replace').splitlines();tail=lines[-1] if lines else 'Starting full native A model'
   if tail!=last:print(tail,flush=True);last=tail
  code=proc.wait()
 dump(out/'process_exit.json',{'exit_code':code,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print((out/'stdout.log').read_text(encoding='utf-8',errors='replace')[-2500:],flush=True)
 if code:raise SystemExit(code)
if __name__=='__main__':main()
