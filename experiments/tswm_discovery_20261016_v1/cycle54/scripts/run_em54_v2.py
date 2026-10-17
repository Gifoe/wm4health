"""Execute six corrected ordinary EM fits; preserve all prior failed artifacts."""
import datetime,hashlib,json,pathlib,subprocess,sys,traceback,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
sys.stdout=(OUT/'em_queue54_v2.stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(OUT/'em_queue54_v2.stderr.log').open('x',encoding='utf-8',buffering=1)
plan=json.loads((ROOT/'plan54_v1.json').read_text());done=[]
def state(status,**kwargs):
 (OUT/'em_queue54_v2.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pid':os.getpid(),'completed':done,'affinity_hex':'ff000000','amendment':'plan54_em_amendment_v2.json',**kwargs},indent=2),encoding='utf-8')
try:
 qual=json.loads((OUT/'QUALIFICATION_EM_FIX54_V1.json').read_text());assert qual['status']=='PASS_ORDINARY_EM_INITIAL_COVARIANCE_NORMALIZATION_FIX'
 for task in plan['tasks']:
  for seed in plan['training_seeds']:
   run=f"em_{task['name']}_seed{seed}";folder=OUT/'runs'/run;folder.mkdir(parents=True,exist_ok=True)
   assert not (folder/'result.json').exists(),'Unexpected previously complete EM fit; inspect'
   state('RUNNING',run=run)
   with (folder/'stdout_v2.log').open('xb') as stdout,(folder/'stderr_v2.log').open('xb') as stderr:
    proc=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/train_em54_affinity_v2.py'),'em',task['name'],str(seed)],stdout=stdout,stderr=stderr)
   assert proc.returncode==0,(run,proc.returncode)
   r=json.loads((folder/'result.json').read_text());assert r['status']=='COMPLETE_FIXED_BUDGET'
   done.append({'run':run,'iterations':r['iterations'],'checkpoint_sha256':r['checkpoint_sha256'],'runtime_seconds':r['runtime_seconds'],'baseline_amendment_sha256':r['baseline_amendment_sha256']})
 state('ALL_SIX_CORRECTED_EM_FITS_COMPLETE')
except Exception:
 state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
