import datetime,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'outputs'/'cofe_etth1_affinity1';out.mkdir(parents=True,exist_ok=True)
 if (out/'plan.json').exists():raise RuntimeError('Existing retry; inspect process and results rather than duplicate')
 args=[sys.executable,'-B','-u',str(ROOT/'scripts'/'run_cofe_pinned.py')]
 with (out/'stdout.log').open('w',encoding='utf-8') as logfile:
  proc=subprocess.Popen(args,stdout=logfile,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  (out/'process.json').write_text(json.dumps({'pid':proc.pid,'argv':args,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2),encoding='utf-8')
  last=''
  while proc.poll() is None:
   time.sleep(5);lines=(out/'stdout.log').read_text(encoding='utf-8',errors='replace').splitlines();tail=lines[-1] if lines else 'Starting pinned COFE'
   if tail!=last:print(tail,flush=True);last=tail
  code=proc.wait()
  (out/'process_exit.json').write_text(json.dumps({'exit_code':code,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':proc.pid},indent=2),encoding='utf-8')
 if code:print((out/'stdout.log').read_text(encoding='utf-8',errors='replace')[-3000:],flush=True);raise SystemExit(1)
 print('Pinned COFE subprocess completed',flush=True)
if __name__=='__main__':main()
