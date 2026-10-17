"""Wait for immutable fit seal, then score and report without user intervention."""
import datetime,hashlib,json,pathlib,subprocess,sys,time,traceback,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
sys.stdout=(OUT/'postprocess54_v1.stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(OUT/'postprocess54_v1.stderr.log').open('x',encoding='utf-8',buffering=1)
def state(status,**kw):
 obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pid':os.getpid(),**kw};(OUT/'postprocess54_v1.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
try:
 state('WAITING_FOR_ALL_12_SEALED_FITS')
 while not (OUT/'SELECTION_SEAL54_V1.json').exists():
  if (OUT/'state54.json').exists():
   training=json.loads((OUT/'state54.json').read_text())
   if training.get('status')=='FAILED_PRESERVED':raise RuntimeError('Training failed; keep all artifacts and wait for diagnosis')
  time.sleep(30)
 qual=json.loads((OUT/'QUALIFICATION54_V2.json').read_text());assert qual['status']=='PASS_DUMMY_NATIVE_API_AND_INDEPENDENT_KALMAN_INTEGRATION'
 for script in ['evaluate54_v2.py','analyze54.py']:
  state('RUNNING',script=script)
  with (OUT/(script+'.stdout.log')).open('xb') as stdout,(OUT/(script+'.stderr.log')).open('xb') as stderr:
   subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/script)],stdout=stdout,stderr=stderr,check=True)
 state('COMPLETE_12_FITS_CAUSAL_SCORING_AND_REPORTS',accepted_directions=0)
except Exception:
 state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
