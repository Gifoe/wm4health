"""Run all registered decoded controls; no outcome-dependent changes within the matrix."""
import pathlib,json,sys,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decoded_control_v1';OUT.mkdir(exist_ok=False)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));ART=pathlib.Path(cfg['artifact_root'])/ROOT.name/'decoded_control_v1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x') as f:json.dump(j,f,indent=2)
matrix=[dict(horizon=h,seed=s) for h in (336,720) for s in (42,43,44)]
dump('MATRIX_V1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),matrix=matrix,protocol_sha256=sha(ROOT/'OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md')))
rows=[]
for cell in matrix:
 tag=f"Electricity_DLinear_H{cell['horizon']}_seed{cell['seed']}";folder=ART/tag;assert not folder.exists()
 log=OUT/(tag+'_stdout.log')
 with (OUT/(tag+'_launch.json')).open('x') as g:
  with log.open('xb') as f:
   p=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts/decoded_control_v1.py'),str(cell['seed']),str(cell['horizon'])],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
   json.dump(dict(pid=p.pid,**cell),g);g.flush();code=p.wait()
 row=dict(**cell,exit_code=code,log_sha256=sha(log),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 if code==0:
  m=json.loads((folder/'metrics.json').read_text());assert m['epochs']==100
  row.update({k:m[k] for k in ['reproduced_mse','reproduced_mae','best_epoch','epochs','checkpoint_sha256','predictions_sha256','num_parameters']});row.update(metrics_path=str(folder/'metrics.json'),metrics_sha256=sha(folder/'metrics.json'))
 dump(tag+'_exit.json',row);rows.append(row);print(json.dumps(row),flush=True)
 (OUT/'progress.json').write_text(json.dumps(dict(closed=len(rows),planned=6,last_cell=cell)),encoding='utf-8')
 if code:break
ok=len(rows)==6 and all(x['exit_code']==0 for x in rows)
dump('RESULTS_V1.json',dict(status='SIX_OFFICIAL_DECODED_CONTROLS_CLOSED' if ok else 'DECODED_CONTROL_ERROR_RETAINED',rows=rows,new_method=False))
sys.exit(0 if ok else 1)
