"""Wait for immutable fit seal, then score and report without user intervention."""
import datetime,hashlib,json,pathlib,subprocess,sys,time,traceback,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
sys.stdout=(OUT/'postprocess54_v2.stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(OUT/'postprocess54_v2.stderr.log').open('x',encoding='utf-8',buffering=1)
def state(status,**kw):
 obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pid':os.getpid(),**kw};(OUT/'postprocess54_v2.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
try:
 state('WAITING_FOR_ALL_12_SEALED_FITS')
 while not (OUT/'SELECTION_SEAL54_V1.json').exists():
  if (OUT/'state54.json').exists():
   training=json.loads((OUT/'state54.json').read_text())
   if training.get('status')=='FAILED_PRESERVED':raise RuntimeError('Training failed; keep all artifacts and wait for diagnosis')
  time.sleep(30)
 qual=json.loads((OUT/'QUALIFICATION54_V2.json').read_text());assert qual['status']=='PASS_DUMMY_NATIVE_API_AND_INDEPENDENT_KALMAN_INTEGRATION'
 amendment=json.loads((ROOT/'plan54_em_amendment_v2.json').read_text())
 assert amendment['heldout_metrics_inspected_before_amendment'] is False
 for result in (OUT/'runs').glob('em_*/result.json'):
  assert json.loads(result.read_text())['baseline_amendment_sha256']==hashlib.sha256((ROOT/'plan54_em_amendment_v2.json').read_bytes()).hexdigest()
 for script in ['evaluate54_v2.py','analyze54_v2.py']:
  state('RUNNING',script=script)
  with (OUT/(script+'.stdout.log')).open('xb') as stdout,(OUT/(script+'.stderr.log')).open('xb') as stderr:
   subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/script)],stdout=stdout,stderr=stderr,check=True)
 state('COMPLETE_12_FITS_CAUSAL_SCORING_AND_REPORTS',accepted_directions=0)
except Exception:
 state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
