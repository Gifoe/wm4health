"""Recover a terminal verifier-state mismatch; never rerun numerical experiments."""
import sys,pathlib,json,subprocess,datetime,os
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs'
def dump(name,obj):
 with (out/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 assert json.loads((out/'run83_v1_exit.json').read_text())['exit_code']==0
 assert json.loads((out/'verify83_v1_exit.json').read_text())['exit_code']==1
 diag=json.loads((out/'NATIVE_REPLAY_DIAGNOSTIC83_V1.json').read_text());assert all(z['exact_match'] for z in diag['rows'] if z['parameters_frozen']);dump('close83_v2_launch.json',dict(utc=utc(),pid=os.getpid(),reason='Align verifier with original frozen parameter flags; exact tolerance unchanged'))
 for script in ['verify83_v2.py','export83_v1.py','attribute83_v2.py']:
  log=out/(script[:-3]+'_stdout.log')
  with log.open('xb') as f:code=subprocess.call([sys.executable,'-I','-B','-u',str(root/'scripts'/script)],stdout=f,stderr=subprocess.STDOUT,cwd=root)
  dump(script[:-3]+'_exit.json',dict(utc=utc(),exit_code=code));assert code==0,(script,code)
 dump('close83_v2_exit.json',dict(utc=utc(),status='VERIFIED_SCREEN_AND_ORACLE_ATTRIBUTION_READY',numerical_experiments_restarted=False))
 dump('close_attribution83_v2_exit.json',dict(utc=utc(),status='ORACLE_ATTRIBUTION_READY_FOR_CHECKING',source_script='attribute83_v2.py'))
if __name__=='__main__':main()
