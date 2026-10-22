"""Fixed six-cell matrix. Preserve every failure, model and original baseline artifact."""
import os,sys,pathlib,json,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
ART=pathlib.Path(cfg['artifact_root'])/ROOT.name/'native_replay_v2';ART.mkdir(exist_ok=True,parents=True)
runtime='D:/baseline_reproduction_seed2021_v1/.venv/Scripts/python.exe'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
matrix=[{'seed':s,'horizon':h} for h in (336,720) for s in (42,43,44)]
dump('NATIVE_REPLAY_MATRIX_V2.json',dict(utc=now(),matrix=matrix,protocol_sha256=sha(ROOT/'PROTOCOL.md'),worker_sha256=sha(ROOT/'scripts/replay_latent_native_v2.py'),test_hyperparameter_tuning=False))
rows=[]
for cell in matrix:
    tag=f"Electricity_DLinear_H{cell['horizon']}_seed{cell['seed']}";log=ART/(tag+'_stdout.log')
    with log.open('xb') as f:
        child=subprocess.Popen([runtime,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts/replay_latent_native_v2.py'),str(cell['seed']),str(cell['horizon'])],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        dump(tag+'_launch.json',dict(pid=child.pid,utc=now(),**cell));code=child.wait()
    row=dict(**cell,exit_code=code,log_path=str(log),log_sha256=sha(log),utc=now())
    metric=ART/tag/'metrics.json'
    if code==0:
        assert metric.is_file();m=json.loads(metric.read_text());row.update({k:m[k] for k in ['reproduced_mse','reproduced_mae','best_epoch','epochs','checkpoint_sha256','predictions_sha256','data_sha256','source_commit','num_parameters','frozen_ae_parameters']})
        row.update(metrics_path=str(metric),metrics_sha256=sha(metric))
    dump(tag+'_exit.json',row);rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'native_replay_v2_progress.json').write_text(json.dumps(dict(utc=now(),closed=len(rows),planned=len(matrix),live_child_pid=None),indent=2),encoding='utf-8')
result=dict(utc=now(),status='ALL_SIX_NATIVE_REPLAYS_CLOSED' if all(r['exit_code']==0 for r in rows) else 'NATIVE_REPLAY_MATRIX_HAS_FAILURES',rows=rows,protocol_sha256=sha(ROOT/'PROTOCOL.md'),new_method=False,accepted_final_directions=0)
dump('NATIVE_REPLAY_RESULTS_V2.json',result);print(json.dumps({'status':result['status'],'closed':len(rows)}),flush=True)
sys.exit(0 if all(r['exit_code']==0 for r in rows) else 1)
