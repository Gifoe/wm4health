"""Seven frozen diagnostic evaluations; distinguish runtime draws from arithmetic."""
import pathlib,sys,os,json,subprocess,datetime,hashlib,traceback
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'inference_parity_v1';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 protocol=sha(ROOT/'IDF_PARITY_ATTRIBUTION_PROTOCOL_V1.md');dump('MATRIX_V1.json',dict(protocol_sha256=protocol,primary_seed=1000,secondary_seeds=[42,1111],diagnostic_evaluations=7,optimizer_updates=0,original_cpu_results_preserved=True))
 cpu=read(ROOT/'runtime_cpu_recovery_v1/READY_V1.json')['python_path'];base=subprocess.check_output([cpu,'-I','-c','import sys;print(sys.base_prefix)'],text=True).strip();win=os.environ['SystemRoot'];cpuenv=os.environ.copy();cpuenv['PATH']=os.pathsep.join([str(pathlib.Path(cpu).parent),base,str(pathlib.Path(base)/'DLLs'),str(pathlib.Path(win)/'System32'),win])
 for k in ['CONDA_PREFIX','CONDA_DEFAULT_ENV']:cpuenv.pop(k,None)
 def run(tag,py,device,seed=1000,bank=None):
  env=(cpuenv if py==cpu else os.environ).copy();env.update(CUDA_VISIBLE_DEVICES='0' if device=='cuda' else '-1',PYTHONNOUSERSITE='1',PYTHONPATH='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');folder=OUT/tag;folder.mkdir(exist_ok=False);command=[py,'-I','-B','-X','utf8','-u',str(ROOT/'scripts/idf_noise_bank_worker_v1.py'),'--tag',tag,'--mode','replay' if bank else 'record','--seed',str(seed),'--device',device]
  if bank:command+=['--bank',bank]
  with (folder/'stdout.log').open('xb') as f:
   p=subprocess.Popen(command,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);code=p.wait()
  receipt=dict(exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(folder/'stdout.log'))
  with (folder/'exit.json').open('x') as f:json.dump(receipt,f,indent=2)
  assert code==0,(tag,code);return read(OUT/(tag+'_V1.json'))
 a=run('cpu24_native_s1000',cpu,'cpu');native=read(ROOT/'native_recipe_v1/idf_id_s1000_V1.json')
 with np.load(a['payload_path'],allow_pickle=False) as x,np.load(native['payload_path'],allow_pickle=False) as y:faithful=max(float(np.max(abs(x[k]-y[k]))) for k in x.files);assert faithful==0,('Noise recorder changed native CPU outputs',faithful)
 b=run('cpu28_native_s1000',sys.executable,'cpu');c=run('gpu28_native_s1000',sys.executable,'cuda');d=run('gpu28_cpu_bank_s1000',sys.executable,'cuda',bank=b['bank_path']);e=run('cpu28_gpu_bank_s1000',sys.executable,'cpu',bank=c['bank_path']);secondary=[run('gpu28_native_s'+str(seed),sys.executable,'cuda',seed=seed) for seed in [42,1111]]
 def compare(left,right):
  with np.load(left['payload_path'],allow_pickle=False) as x,np.load(right['payload_path'],allow_pickle=False) as y:
   assert np.array_equal(x['x'],y['x']);v=x['x_pred'].astype(np.float64);w=y['x_pred'].astype(np.float64);diff=v-w;relative_rms=float(np.sqrt(np.mean(diff**2))/max(np.sqrt(np.mean(v**2)),1e-12));maximum=float(abs(diff).max())
  m=left['native_metrics']['pred_mse'];n=right['native_metrics']['pred_mse'];return dict(left=left['tag'],right=right['tag'],prediction_max=maximum,relative_prediction_rms=relative_rms,relative_mse_difference=abs(m-n)/max(abs(m),1e-12),mse_left=m,mse_right=n)
 comparisons=[compare(a,b),compare(b,c),compare(b,d),compare(c,e)];rng_supported=comparisons[1]['relative_mse_difference']>.01 and all(x['relative_prediction_rms']<=1e-3 and x['relative_mse_difference']<=1e-3 for x in comparisons[2:])
 source=read(ROOT/'acquisition_recovery_v1/RESULTS_V1.json');src=pathlib.Path(source['source_root']);reference=[]
 for case in [c,*secondary]:
  ref=read(src/'mocap/outputs/refined'/('idf_id_s'+str(case['seed'])+'.json'))
  for key in ['recon_mse','pred_mse']:
   m=case['native_metrics'][key];r=ref[key];reference.append(dict(seed=case['seed'],metric=key,actual=m,reference=r,relative_difference=abs(m-r)/max(abs(r),1e-12),within_tolerance=abs(m-r)<=max(1e-4,.01*abs(r))))
 for row in source['files']:assert sha(src/row['path'])==row['sha256']
 result=dict(status='IDF_NATIVE_PARITY_ATTRIBUTION_EXECUTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_cpu_recording_max=faithful,cases=[a,b,c,d,e,*secondary],comparisons=comparisons,random_draw_explanation_supported_locally=rng_supported,gpu_reference_parity=reference,all_gpu_reference_metrics_within_tolerance=all(x['within_tolerance'] for x in reference),source_files_unchanged=True,diagnostic_evaluations=7,optimizer_updates=0,original_cpu_reference_parity_reclassified=False,external_vendor_runtime_uniquely_recovered=False,accepted_final_directions=0,protocol_sha256=protocol);dump('RESULTS_V1.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
except Exception as ex:dump('FAILURE_V1.json',dict(exception_type=type(ex).__name__,exception=str(ex),traceback=traceback.format_exc()));raise
