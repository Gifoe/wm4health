"""Quiet downstream watcher; existing trainers and postprocessor stay unchanged."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
sys.stdout=(OUT/'watch_causal_risk54_v1.stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(OUT/'watch_causal_risk54_v1.stderr.log').open('x',encoding='utf-8',buffering=1)
def state(status,**kwargs):
    value={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':status,**kwargs};(OUT/'watch_causal_risk54_v1.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
def read_safe(path):
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {}
try:
    reg=read_safe(OUT/'CAUSAL_RISK_REGISTRATION54_V1.json');assert reg['heldout_opened_at_registration'] is False
    for p,value in reg['files_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value
    state('WAITING_FOR_EXISTING_ALL16_POSTPROCESS_COMPLETION_NO_EXTRA_TRAINING')
    while True:
        observed=read_safe(OUT/'postprocess54_v3.json')
        if observed.get('status')=='FAILED_PRESERVED':raise RuntimeError('Existing pipeline failed; preserve and diagnose without reading unsealed results')
        if observed.get('status')=='COMPLETE_PRIMARY12_AND_FOUR_MATCHED_EXISTING_EM_OPTION_CONTROLS':break
        time.sleep(30)
    state('READ_ONLY_RISK_INTERPRETATION_RUNNING')
    with (OUT/'analyze_causal_risk54_v1.stdout.log').open('xb') as stdout,(OUT/'analyze_causal_risk54_v1.stderr.log').open('xb') as stderr:
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/analyze_causal_risk54_v1.py')],stdout=stdout,stderr=stderr,check=True)
    state('COMPLETE_READ_ONLY_RISK_INTERPRETATION',additional_optimizer_updates=0,accepted_new_directions=0)
except Exception:state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
