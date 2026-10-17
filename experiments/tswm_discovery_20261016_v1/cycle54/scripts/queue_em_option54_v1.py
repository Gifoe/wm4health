"""Four fixed source-option controls, kept separate from the original 12-fit queue."""
import pathlib,sys,json,subprocess,datetime,time,traceback,hashlib,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';folder=OUT/'em_option_learning_v1';folder.mkdir(exist_ok=True)
sys.stdout=(folder/'queue.stdout.log').open('x',encoding='utf-8',buffering=1);sys.stderr=(folder/'queue.stderr.log').open('x',encoding='utf-8',buffering=1)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
done=[]
def state(status,**extra):(folder/'state.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pid':os.getpid(),'completed':done,**extra},indent=2),encoding='utf-8')
try:
 state('WAITING_FOR_NATIVE_EM_SWITCH_QUALIFICATION')
 while not (OUT/'em_switch_audit_v2/summary.json').exists():
  q=OUT/'em_switch54_v2.json'
  if q.exists() and json.loads(q.read_text()).get('status')=='FAILED_PRESERVED':raise RuntimeError('EM switch qualification failed; inspect')
  time.sleep(10)
 for task in ['linear_small','linear_medium']:
  for seed in [0,1]:
   run=f'rp_emtrue_{task}_seed{seed}';rf=folder/'runs'/run;rf.mkdir(parents=True,exist_ok=True);assert not (rf/'result.json').exists()
   state('RUNNING',run=run)
   with (rf/'stdout.log').open('xb') as stdout,(rf/'stderr.log').open('xb') as stderr:subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/train_em_option54_v1.py'),task,str(seed)],stdout=stdout,stderr=stderr,check=True)
   r=json.loads((rf/'result.json').read_text());assert sha(r['checkpoint'])==r['checkpoint_sha256'];done.append(r)
 seal=folder/'SELECTION_SEAL54_EM_TRUE_V1.json';assert not seal.exists();seal.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ALL_FOUR_NATIVE_EM_TRUE_FITS_FROZEN','protocol_sha256':sha(ROOT/'EM_OPTION_LEARNING54_V1.md'),'fits':done,'heldout_read':False,'latent_truth_read':False},indent=2),encoding='utf-8')
 state('ALL_FOUR_NATIVE_EM_TRUE_FITS_SEALED',selection_seal_sha256=sha(seal))
except Exception:state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
