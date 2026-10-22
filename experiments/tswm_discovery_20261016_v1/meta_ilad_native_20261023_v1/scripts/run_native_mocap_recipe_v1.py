"""Execute the reviewed native data/evaluation recipe with pinned inputs and no fits."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='-1',MPLBACKEND='Agg',TQDM_DISABLE='1',PYTHONNOUSERSITE='1',PYTHONPATH='')
import pathlib,sys,json,hashlib,datetime,subprocess,urllib.request,concurrent.futures,threading,traceback,ast
import torch,numpy as np
torch.set_num_threads(1)
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'native_recipe_v1';OUT.mkdir(exist_ok=False);lock=threading.Lock()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
def event(stage,**kw):
 if 'utc' in kw:kw['source_utc']=kw.pop('utc')
 row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with lock:
  with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print(json.dumps(row,allow_nan=False),flush=True)
def run(name,script,args=(),wrapper=False):
 p=OUT/name;p.mkdir(exist_ok=False);command=[sys.executable,'-B','-X','utf8','-u']
 if wrapper:
  code="import sys,socket,runpy,os;socket.setdefaulttimeout(20);p=sys.argv[1];sys.path.insert(0,os.path.dirname(p));sys.argv=sys.argv[1:];runpy.run_path(p,run_name='__main__')";command+=['-c',code,str(script),*map(str,args)]
 else:command+=[str(script),*map(str,args)]
 with (p/'stdout.log').open('xb') as f:
  child=subprocess.Popen(command,cwd=script.parent,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
  with (p/'launch.json').open('x') as g:json.dump(dict(pid=child.pid,command=command,cwd=str(script.parent),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
  code=child.wait()
 receipt=dict(pid=child.pid,exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(p/'stdout.log'))
 with (p/'exit.json').open('x') as g:json.dump(receipt,g,indent=2)
 event('native_child_terminal',name=name,**receipt);assert code==0,(name,code);return receipt
try:
 acquisition=read(ROOT/'acquisition_recovery_v1/RESULTS_V1.json');SRC=pathlib.Path(acquisition['source_root']);runtime=read(ROOT/'runtime_cpu_recovery_v1/READY_V1.json');assert pathlib.Path(sys.executable).resolve()==pathlib.Path(runtime['python_path']).resolve();assert torch.__version__.split('+')[0]=='2.4.1' and np.__version__=='2.0.1'
 for x in acquisition['files']:assert sha(SRC/x['path'])==x['sha256']
 for x in acquisition['files']:
  if x['path'].endswith('.py'):ast.parse((SRC/x['path']).read_text())
 dump('SOURCE_EXECUTION_AUDIT_V1.json',dict(source_revision=acquisition['source_revision'],source_root=str(SRC),files_rehashed=len(acquisition['files']),reviewed_entries=['mocap/gen_data/download.py','mocap/gen_data/prepare_data.py','mocap/meta_ilad/evaluate.py','mocap/baselines/idf/evaluate.py','mocap/meta_ilad/dataloader.py'],source_unchanged=True,version='author refined MoCAP only',protocol_sha256=sha(ROOT/'MOCAP_NATIVE_PROTOCOL_V1.md')))
 storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');DATA=pathlib.Path(storage['data_root'])/ROOT.name/'mocap_refined_v1';DATA.mkdir(parents=True,exist_ok=False);RAW=DATA/'raw';RAW.mkdir();ART=pathlib.Path(storage['artifact_root'])/ROOT.name/'native_recipe_v1';ART.mkdir(parents=True,exist_ok=False);gen=SRC/'mocap/gen_data';ref=read(gen/'fingerprints.json');splits=read(gen/'splits.json');trials=sorted({t for split in splits.values() for part in split.values() for t in part});assert len(trials)==60 and set(trials)==set(ref['raw'])
 def fetch(trial,scheme):
  url=f'{scheme}://mocap.cs.cmu.edu/subjects/{trial[:2]}/{trial}.amc'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'native-mocap-reproduction'}),timeout=20) as f:b=f.read(8388609)
  assert len(b)<=8388608 and hashlib.sha256(b).hexdigest()==ref['raw'][trial],('raw checksum',trial)
  path=RAW/(trial+'.amc')
  with path.open('xb') as f:f.write(b)
  row=dict(trial=trial,source_url=url,sha256=sha(path),bytes=len(b));event('raw_trial_verified',**row);return row
 first=None;transport=[]
 for scheme in ['https','http']:
  try:first=fetch(trials[0],scheme);break
  except Exception as e:transport.append(dict(scheme=scheme,exception_type=type(e).__name__,exception=str(e)));event('raw_endpoint_probe_failed',**transport[-1])
 assert first,'Both official CMU endpoint probes failed; no model execution';scheme=first['source_url'].split(':')[0]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rawrows=[first]+list(pool.map(lambda t:fetch(t,scheme),trials[1:]))
 dump('RAW_DATA_V1.json',dict(rows=rawrows,transport_probes=transport,all60_checksums_match=True));run('native_raw_validator',gen/'download.py',['--raw-dir',RAW],wrapper=True);run('native_data_prepare',gen/'prepare_data.py',['--raw-dir',RAW,'--data-dir',DATA])
 checks=[]
 for split,parts in splits.items():
  for part,names in parts.items():
   path=DATA/split/f'data_{part}.pt';data=torch.load(path,map_location='cpu',weights_only=True);assert data['trials']==names
   for k,v in data.items():
    if torch.is_tensor(v):
     wanted=ref[f'{split}/{part}'][k];digest=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest();assert digest==wanted['sha256'] and list(v.shape)==wanted['shape'] and str(v.dtype)==wanted['dtype'];checks.append(dict(split=split,part=part,key=k,sha256=digest))
 assert len(checks)==24;dump('DATA_QUALIFICATION_V1.json',dict(status='RAW60_AND_ALL24_NATIVE_TENSOR_FINGERPRINTS_EXACT',rows=checks,data_root=str(DATA),raw_manifest_sha256=sha(OUT/'RAW_DATA_V1.json')))
 weightrows=[]
 for x in acquisition['files']:
  if x['checkpoint']:
   state=torch.load(SRC/x['path'],map_location='cpu',weights_only=True);assert isinstance(state,dict) and state and all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values());weightrows.append(dict(path=x['path'],sha256=x['sha256'],tensor_count=len(state),tensor_elements=sum(v.numel() for v in state.values()),weights_only_load=True))
 assert len(weightrows)==9;dump('CHECKPOINT_QUALIFICATION_V1.json',dict(rows=weightrows,all_finite=True))
 cells=[]
 for model in ['meta_ilad','idf']:
  for seed in [42,1000,1111]:
   for split in (['ID','OOD'] if model=='meta_ilad' else ['ID']):
    name=f'{model}_{split.lower()}_s{seed}';script=SRC/('mocap/meta_ilad/evaluate.py' if model=='meta_ilad' else 'mocap/baselines/idf/evaluate.py');ck=SRC/'mocap/checkpoints'/(name+'.pt');prefix=ART/name;args=['--ckpt',ck,'--data-dir',DATA,'--seed',seed,'--out',prefix]
    if model=='meta_ilad':args+=['--split',split]
    run(name,script,args);actual=read(str(prefix)+'.json');provided=read(SRC/'mocap/outputs/refined'/(name+'.json'));parity=[]
    for key in ['recon_mse','obs_mse','pred_mse']:
     if key in actual and key in provided:
      delta=abs(actual[key]-provided[key]);parity.append(dict(metric=key,actual=actual[key],reference=provided[key],absolute_difference=delta,relative_difference=delta/max(abs(provided[key]),1e-12),within_tolerance=delta<=max(1e-4,.01*abs(provided[key]))))
    with np.load(str(prefix)+'.npz',allow_pickle=False) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
    row=dict(name=name,model=model,split=split,seed=seed,metrics=actual,reference_metrics=provided,parity=parity,payload_path=str(prefix)+'.npz',payload_sha256=sha(str(prefix)+'.npz'),checkpoint_sha256=sha(ck),source_metric_json_sha256=sha(SRC/'mocap/outputs/refined'/(name+'.json')));cells.append(row);dump(name+'_V1.json',row);event('native_cell_complete',name=name,pred_mse=actual['pred_mse'],reference_parity=all(x['within_tolerance'] for x in parity))
 # One fixed-seed native replay, separately counted.
 first=next(x for x in cells if x['name']=='meta_ilad_id_s42');prefix=ART/'replay_meta_ilad_id_s42';script=SRC/'mocap/meta_ilad/evaluate.py';ck=SRC/'mocap/checkpoints/meta_ilad_id_s42.pt';run('replay_meta_ilad_id_s42',script,['--ckpt',ck,'--split','ID','--data-dir',DATA,'--seed',42,'--out',prefix])
 with np.load(first['payload_path'],allow_pickle=False) as a,np.load(str(prefix)+'.npz',allow_pickle=False) as b:replay={k:float(np.max(abs(a[k]-b[k]))) for k in a.files};assert max(replay.values())==0,replay
 # A qualification-only future-observation corruption; no benchmark metric from it.
 shadow=pathlib.Path(storage['data_root'])/ROOT.name/'future_boundary_shadow_v1';(shadow/'OOD').mkdir(parents=True,exist_ok=False);d=torch.load(DATA/'OOD/data_test.pt',weights_only=True);d['data']=d['data'].clone();d['data'][:,150:]+=17.;torch.save(d,shadow/'OOD/data_test.pt');prefix=ART/'future_boundary_meta_ilad_ood_s42';ck=SRC/'mocap/checkpoints/meta_ilad_ood_s42.pt';run('future_boundary_meta_ilad_ood_s42',script,['--ckpt',ck,'--split','OOD','--data-dir',shadow,'--seed',42,'--out',prefix]);base=next(x for x in cells if x['name']=='meta_ilad_ood_s42')
 with np.load(base['payload_path'],allow_pickle=False) as a,np.load(str(prefix)+'.npz',allow_pickle=False) as b:
  boundary={k:float(np.max(abs(a[k]-b[k]))) for k in ['x_pred','x_recon']};assert max(boundary.values())<=1e-6 and np.array_equal(a['x'][:,:75],b['x'][:,:75]) and not np.array_equal(a['x'][:,75:],b['x'][:,75:])
 for x in acquisition['files']:assert sha(SRC/x['path'])==x['sha256']
 dump('RESULTS_V1.json',dict(status='NATIVE_REFINED_MOCAP_RECIPE_EXECUTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=acquisition['source_revision'],source_files_unchanged=True,data_fingerprints_exact=True,unique_benchmark_cells=9,actual_evaluation_calls=11,new_fits=0,cells=cells,replay_maxima=replay,future_boundary_maxima=boundary,all_reference_metrics_within_tolerance=all(p['within_tolerance'] for c in cells for p in c['parity']),protocol_sha256=sha(ROOT/'MOCAP_NATIVE_PROTOCOL_V1.md'),original_paper_table2_reproduced=False,causal_mechanism_fidelity_evaluated=False,accepted_final_directions=0));event('native_recipe_complete',cells=9,evaluation_calls=11,fits=0)
except Exception as e:dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
