"""Control a CPU/source fingerprint mismatch using the same native simulator on GPU."""
import pathlib,sys,json,hashlib,datetime,traceback,importlib.util
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'physics_gpu_qualification_v1';OUT.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1')
PAYLOAD=ART/'physics_gpu_vectors_only_v1';PAYLOAD.mkdir(exist_ok=False)
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 assert torch.cuda.is_available();torch.set_num_threads(1)
 acq=json.loads((ROOT/'physics_qualification_v1/ACQUISITION_V1.json').read_text())
 for row in acq['files']:assert sha((SRC/row['path']).read_bytes())==row['sha256']
 cpu=json.loads((ROOT/'physics_qualification_v1/RESULTS_V1.json').read_text());assert cpu['independent_max_absolute_error']<.005
 sys.path.insert(0,str(SRC/'pendulum/gen_data'));import generate_data as native
 from fingerprint import fingerprint,check
 refs=json.loads((SRC/'pendulum/gen_data/fingerprints.json').read_text());rows=[]
 for split,cfg in native.SPLITS.items():
  data=native.generate(cfg,'cuda:0',render=False);ok=check(data,split);checks={}
  oldpath=ART/'physics_vectors_only_v1'/(split+'_vectors_only.npz')
  oldrow=next(x for x in cpu['splits'] if x['split']==split);assert sha(oldpath.read_bytes())==oldrow['payload_sha256']
  with np.load(oldpath) as old:
   for key,t in data.items():
    fp=fingerprint(t);diff=float(np.max(np.abs(t.numpy().astype(np.float64)-old[key].astype(np.float64))))
    assert diff<.005,(split,key,diff)
    checks[key]=dict(generated=fp,source_reference=refs[split][key],source_exact_match=fp['sha256']==refs[split][key]['sha256'],cpu_gpu_max_absolute_error=diff)
  path=PAYLOAD/(split+'_vectors_only.npz')
  with path.open('xb') as f:np.savez_compressed(f,**{k:v.numpy() for k,v in data.items()})
  rows.append(dict(split=split,checks=checks,native_existing_keys_all_match=ok,payload_sha256=sha(path.read_bytes()),complete_model_input=False))
  print(json.dumps(dict(split=split,source_matches=sum(x['source_exact_match'] for x in checks.values()),cpu_gpu_state_max_error=checks['xt']['cpu_gpu_max_absolute_error'])),flush=True)
 for row in acq['files']:assert sha((SRC/row['path']).read_bytes())==row['sha256']
 result=dict(status='GPU_SIMULATOR_CONTROL_CLOSED_RENDERER_UNQUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime=dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),numpy=np.__version__,renderer_modules={name:importlib.util.find_spec(name) is not None for name in ['kaolin','gradsim']}),splits=rows,source_revision=acq['source_revision'],native_simulator_calls=3,learned_model_evaluations=0,new_fits=0,rendered_model_inputs_available=False,local_payload_sync=False)
 dump('RESULTS_V1.json',result);print(json.dumps(dict(status=result['status'],learned_model_evaluations=0)),flush=True)
except Exception as e:
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
