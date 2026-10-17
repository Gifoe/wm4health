"""Run the registered Gaussian audit after existing all16 scoring/risk closes."""
import datetime,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
sys.stdout=(OUT/'watch_gaussian_beliefs54_v1.stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(OUT/'watch_gaussian_beliefs54_v1.stderr.log').open('x',encoding='utf-8',buffering=1)
def state(status,**kw):
    (OUT/'watch_gaussian_beliefs54_v1.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':status,**kw},indent=2),encoding='utf-8')
try:
    state('WAITING_FOR_EXISTING_ALL16_SCORING_AND_CAUSAL_RISK_NO_EXTRA_TRAINING')
    while True:
        p=OUT/'watch_causal_risk54_v1.json'
        try:current=json.loads(p.read_text()) if p.exists() else {}
        except json.JSONDecodeError:current={}
        if current.get('status')=='FAILED_PRESERVED':raise RuntimeError('Existing risk analysis failed; preserve and diagnose')
        p=OUT/'causal_risk_interpretation_v1/summary.json'
        try:ready=p.exists() and json.loads(p.read_text()).get('status')=='COMPLETE_READ_ONLY_CAUSAL_RISK_INTERPRETATION_OF_ALL16_FROZEN_FITS'
        except json.JSONDecodeError:ready=False
        if ready:break
        time.sleep(30)
    state('RUNNING_REGISTERED_READ_ONLY_GAUSSIAN_AUDIT')
    with (OUT/'audit_gaussian_beliefs54_v1.stdout.log').open('xb') as stdout,(OUT/'audit_gaussian_beliefs54_v1.stderr.log').open('xb') as stderr:
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/audit_gaussian_beliefs54_v1.py')],stdout=stdout,stderr=stderr,check=True)
    state('COMPLETE_ALL16_GAUSSIAN_BELIEF_AUDIT',optimizer_updates=0,accepted_final_directions=0)
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
