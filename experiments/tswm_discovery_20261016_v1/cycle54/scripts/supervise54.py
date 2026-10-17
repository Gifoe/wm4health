"""Sequential immutable 12-fit queue with a seal before any evaluation."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(parents=True,exist_ok=True)
plan=json.loads((ROOT/'plan54_v1.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(name,obj):
 obj['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
try:
 if not (OUT/'preflight54.json').exists():subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/prepare54.py')],check=True)
 completed=[]
 for method in ['rp','em']:
  for task in plan['tasks']:
   for seed in plan['training_seeds']:
    run=f"{method}_{task['name']}_seed{seed}";folder=OUT/'runs'/run;folder.mkdir(parents=True,exist_ok=True)
    result=folder/'result.json'
    if result.exists():
     r=json.loads(result.read_text());assert sha(r['checkpoint'])==r['checkpoint_sha256']
    else:
     write('state54.json',{'status':'RUNNING','run':run,'completed':completed,'pid':os.getpid()})
     with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
      proc=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/train54.py'),method,task['name'],str(seed)],stdout=stdout,stderr=stderr)
     assert proc.returncode==0,(run,proc.returncode)
    completed.append({'run':run,'result_sha256':sha(result),**{k:v for k,v in json.loads(result.read_text()).items() if k in ['checkpoint','checkpoint_sha256','iterations','method','runtime_seconds','parameter_count']}})
 seal=OUT/'SELECTION_SEAL54_V1.json';assert not seal.exists()
 write(seal.name,{'status':'ALL_12_TERMINAL_FITS_SEALED_BEFORE_STATE_READOUT_OR_HELDOUT_EVALUATION','plan_sha256':sha(ROOT/'plan54_v1.json'),'fits':completed,'states_used_in_training_or_selection':False})
 write('state54.json',{'status':'ALL_FITS_COMPLETE_SEALED','completed':completed,'selection_seal_sha256':sha(seal),'pid':os.getpid()})
 print('ALL_FITS_COMPLETE_SEALED',flush=True)
except Exception:
 write('state54.json',{'status':'FAILED_PRESERVED','traceback':traceback.format_exc(),'pid':os.getpid()});raise
