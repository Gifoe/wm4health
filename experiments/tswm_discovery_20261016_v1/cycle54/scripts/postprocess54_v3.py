"""Wait for 12 primary plus 4 source-option models; freeze all readouts before scoring."""
import datetime,hashlib,json,pathlib,subprocess,sys,time,traceback,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';SUP=OUT/'em_option_learning_v1'
sys.stdout=(OUT/'postprocess54_v3.stdout.log').open('x',encoding='utf-8',buffering=1);sys.stderr=(OUT/'postprocess54_v3.stderr.log').open('x',encoding='utf-8',buffering=1)
def state(status,**kw):
 obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pid':os.getpid(),**kw};(OUT/'postprocess54_v3.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
def read_safe(p):
 try:return json.loads(p.read_text())
 except json.JSONDecodeError:return {}
try:
 state('WAITING_FOR_PRIMARY12_AND_SUPPLEMENTAL4_MODEL_SEALS')
 while not ((OUT/'SELECTION_SEAL54_V1.json').exists() and (SUP/'SELECTION_SEAL54_EM_TRUE_V1.json').exists()):
  for p in [OUT/'state54.json',SUP/'state.json']:
   if p.exists() and read_safe(p).get('status')=='FAILED_PRESERVED':raise RuntimeError('A training queue failed; retain artifacts and diagnose before evaluation')
  time.sleep(30)
 assert read_safe(OUT/'QUALIFICATION54_V2.json')['status']=='PASS_DUMMY_NATIVE_API_AND_INDEPENDENT_KALMAN_INTEGRATION'
 for result in (OUT/'runs').glob('em_*/result.json'):
  assert read_safe(result)['baseline_amendment_sha256']==hashlib.sha256((ROOT/'plan54_em_amendment_v2.json').read_bytes()).hexdigest()
 stages=[('evaluate_em_option54_v1.py',['prepare']),('evaluate54_v2.py',[]),('evaluate_em_option54_v1.py',['evaluate']),('analyze54_v3.py',[])]
 for script,args in stages:
  label=script+('_'+'_'.join(args) if args else '');state('RUNNING',script=script,arguments=args)
  with (OUT/(label+'.stdout.log')).open('xb') as stdout,(OUT/(label+'.stderr.log')).open('xb') as stderr:
   subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/script),*args],stdout=stdout,stderr=stderr,check=True)
 state('COMPLETE_PRIMARY12_AND_FOUR_MATCHED_EXISTING_EM_OPTION_CONTROLS',accepted_directions=0)
except Exception:state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
