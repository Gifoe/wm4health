"""Fixed18-cell conditional matrix. Gate on Validation, stop only on execution failure."""
import pathlib,json,sys,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
selection=json.loads((ROOT/'outputs/decoded_validation_v1/RESULTS_V1.json').read_text());assert selection['decision']=='RUN_REGISTERED_ITRANSFORMER_COMPATIBILITY_CONTROLS'
OUT=ROOT/'outputs/strong_itransformer_v1';OUT.mkdir(exist_ok=False)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));ART=pathlib.Path(cfg['artifact_root'])/ROOT.name
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x') as f:json.dump(j,f,indent=2)
matrix=[dict(horizon=h,seed=s,mode=m) for h in (336,720) for s in (42,43,44) for m in ('latent','decoded','raw')]
dump('MATRIX_V1.json',dict(matrix=matrix,validation_gate_sha256=sha(ROOT/'outputs/decoded_validation_v1/RESULTS_V1.json'),protocol_sha256=sha(ROOT/'OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md'),epochs=100,test_selection=False))
rows=[]
for cell in matrix:
 mode=cell['mode'];H=cell['horizon'];seed=cell['seed'];tag=f'Electricity_iTransformer_H{H}_seed{seed}';folder=ART/('strong_it_'+mode+'_v1')/tag
 assert not folder.exists();log=OUT/(mode+'_'+tag+'_stdout.log')
 with (OUT/(mode+'_'+tag+'_launch.json')).open('x') as g:
  with log.open('xb') as f:
   p=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts/strong_itransformer_control_v1.py'),mode,str(seed),str(H)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
   json.dump(dict(pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**cell),g);g.flush();code=p.wait()
 row=dict(**cell,exit_code=code,log_sha256=sha(log),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 if code==0:
  m=json.loads((folder/'metrics.json').read_text());assert m['epochs']==100
  assert sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(folder/'predictions.npz')==m['predictions_sha256']
  row.update({k:m[k] for k in ['reproduced_mse','reproduced_mae','best_epoch','epochs','num_parameters','checkpoint_sha256','predictions_sha256']});row.update(metrics_path=str(folder/'metrics.json'),metrics_sha256=sha(folder/'metrics.json'))
  if mode=='decoded':
   ref=ART/'strong_it_latent_v1'/tag/'INITIALIZATION_V1.json';a=json.loads(ref.read_text());b=json.loads((folder/'INITIALIZATION_V1.json').read_text());assert a['model_tensor_sha256']==b['model_tensor_sha256'] and a['ae_tensor_sha256']==b['ae_tensor_sha256']
   row['paired_latent_decoded_initialization_byte_verified']=True
 dump(mode+'_'+tag+'_exit.json',row);rows.append(row);print(json.dumps(row),flush=True)
 (OUT/'progress.json').write_text(json.dumps(dict(utc=row['utc'],closed=len(rows),planned=18,last_cell=cell)),encoding='utf-8')
 if code:break
ok=len(rows)==18 and all(x['exit_code']==0 for x in rows)
dump('RESULTS_V1.json',dict(status='EIGHTEEN_OFFICIAL_ITRANSFORMER_CONTROLS_CLOSED' if ok else 'STRONG_CONTROL_ERROR_RETAINED',rows=rows,new_method=False))
sys.exit(0 if ok else 1)
