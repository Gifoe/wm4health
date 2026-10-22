"""Execute original vectors-only simulation; independently verify the oracle, not a model."""
import pathlib,sys,json,hashlib,datetime,traceback,importlib.util,subprocess,shutil,os
import numpy as np,torch,scipy
from scipy.integrate import solve_ivp
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'physics_qualification_v1'
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
PAYLOAD=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\physics_vectors_only_v1')
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 assert not (OUT/'RESULTS_V1.json').exists();PAYLOAD.mkdir(exist_ok=False)
 acquisition=json.loads((OUT/'ACQUISITION_V1.json').read_text(encoding='utf-8'))
 for row in acquisition['files']:assert sha((SRC/row['path']).read_bytes())==row['sha256']
 torch.set_num_threads(1);sys.path.insert(0,str(SRC/'pendulum/gen_data'))
 import generate_data as native
 from fingerprint import fingerprint,check
 refs=json.loads((SRC/'pendulum/gen_data/fingerprints.json').read_text())
 splits=[];all_errors=[]
 for split,cfg in native.SPLITS.items():
  data=native.generate(cfg,'cpu',render=False)
  native_match=check(data,split)
  rows={}
  for key,value in data.items():
   f=fingerprint(value);ref=refs[split][key]
   rows[key]=dict(generated=f,source_reference=ref,exact_match=f['sha256']==ref['sha256'])
  x=data['xt'].numpy();p=data['params'].numpy();labels=data['label'].numpy()
  assert x.shape==(cfg['n_init']*cfg['n_param'],60,2) and np.isfinite(x).all()
  assert np.array_equal(labels[:,0],np.tile(np.arange(cfg['n_param']),cfg['n_init']))
  assert np.array_equal(p.reshape(cfg['n_init'],cfg['n_param'],2),np.broadcast_to(p[:cfg['n_param']],(cfg['n_init'],cfg['n_param'],2)))
  assert np.array_equal(x[:,0].reshape(cfg['n_init'],cfg['n_param'],2),np.broadcast_to(x[::cfg['n_param'],0,None,:],(cfg['n_init'],cfg['n_param'],2)))
  indexes=np.linspace(0,len(x)-1,6,dtype=int);comparisons=[]
  # Source timepoint rounding is retained; an independent float64 solver controls integration error.
  ts=native.Dynamics('cpu','physics').t
  ts=torch.linspace(0.,.1*59,60).double().numpy()
  for i in indexes:
   omega,gamma=map(float,p[i])
   def rhs(t,z):return [z[1],-2*gamma*z[1]-omega**2*np.sin(z[0])+3*np.cos(np.pi*t)]
   sol=solve_ivp(rhs,(float(ts[0]),float(ts[-1])),x[i,0].astype(np.float64),t_eval=ts,method='DOP853',rtol=1e-10,atol=1e-12)
   assert sol.success and sol.y.shape==(2,60)
   err=float(np.max(np.abs(x[i].astype(np.float64)-sol.y.T)));assert err<.005,(split,int(i),err)
   comparisons.append(dict(index=int(i),max_absolute_state_error=err));all_errors.append(err)
  path=PAYLOAD/(split+'_vectors_only.npz')
  with path.open('xb') as f:np.savez_compressed(f,xt=x,params=p,label=labels)
  splits.append(dict(split=split,sequences=len(x),source_fingerprints=rows,native_existing_keys_all_match=native_match,rendered_yt_present=False,complete_model_input=False,independent_trajectories=comparisons,payload_sha256=sha(path.read_bytes())))
  print(json.dumps(dict(split=split,sequences=len(x),vector_fingerprint_matches=sum(r['exact_match'] for r in rows.values()),independent_max_error=max(r['max_absolute_state_error'] for r in comparisons))),flush=True)
 modules={name:importlib.util.find_spec(name) is not None for name in ['kaolin','gradsim','torchdiffeq']}
 standard_cuda=pathlib.Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA')
 vswhere=pathlib.Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
 vs=[]
 if vswhere.exists():
  cp=subprocess.run([str(vswhere),'-all','-products','*','-format','json','-utf8'],capture_output=True,timeout=20,check=True)
  vs=[dict(installationPath=x.get('installationPath'),displayName=x.get('displayName')) for x in json.loads(cp.stdout.decode('utf-8-sig'))]
 inventory=dict(modules_in_executed_cpu_runtime=modules,path_tools={x:shutil.which(x) for x in ['nvcc.exe','cl.exe','cmake.exe','ninja.exe']},standard_cuda_path_exists=standard_cuda.exists(),standard_cuda_directories=[x.name for x in standard_cuda.iterdir()] if standard_cuda.exists() else [],vswhere_present=vswhere.exists(),visual_studio_registrations=vs,exhaustive_disk_search=False,renderer_installation_attempted=False)
 dump('CAPABILITY_INVENTORY_V1.json',inventory)
 result=dict(status='NATIVE_VECTORS_ORACLE_QUALIFIED_RENDERED_MODEL_INPUT_UNQUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=acquisition['source_revision'],runtime=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,device='cpu'),splits=splits,total_sequences=sum(x['sequences'] for x in splits),independent_trajectory_checks=len(all_errors),independent_max_absolute_error=max(all_errors),frozen_threshold=.005,renderer_required=True,rendered_model_inputs_available=False,learned_model_evaluations=0,new_fits=0,novel_research_gap_established=False,local_payload_sync=False)
 for row in acquisition['files']:assert sha((SRC/row['path']).read_bytes())==row['sha256']
 dump('RESULTS_V1.json',result);print(json.dumps(dict(status=result['status'],sequences=result['total_sequences'],max_error=result['independent_max_absolute_error'],learned_model_evaluations=0)),flush=True)
except Exception as e:
 dump('EXECUTION_FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
