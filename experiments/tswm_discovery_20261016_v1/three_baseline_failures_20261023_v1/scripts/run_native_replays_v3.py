"""Isolated controller receipts; recover the completed first V2 worker without retraining."""
import sys,pathlib,json,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/native_replays_v3'
OUT.mkdir(exist_ok=False,parents=True)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
ART=pathlib.Path(cfg['artifact_root'])/ROOT.name/'native_replay_v2';ART.mkdir(exist_ok=True,parents=True)
runtime='D:/baseline_reproduction_seed2021_v1/.venv/Scripts/python.exe'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
matrix=[dict(seed=s,horizon=h) for h in (336,720) for s in (42,43,44)]
dump('MATRIX_V3.json',dict(utc=now(),matrix=matrix,protocol_sha256=sha(ROOT/'PROTOCOL.md'),worker_sha256=sha(ROOT/'scripts/replay_latent_native_v2.py'),old_errors_retained=True))
rows=[]
for cell in matrix:
    tag=f"Electricity_DLinear_H{cell['horizon']}_seed{cell['seed']}";folder=ART/tag;metric=folder/'metrics.json';log=ART/(tag+'_stdout.log')
    if metric.exists():
        assert cell==dict(seed=42,horizon=336),'Only the known completed orphan may be recovered'
        assert (folder/'SOURCE_UNCHANGED_AFTER_V1.json').exists()
        assert 'NATIVE_REPLAY_COMPLETE' in log.read_text(encoding='utf-8-sig')
        code=None;execution='RECOVERED_COMPLETED_V2_WORKER_NO_PROCESS_EXIT_RECEIPT'
    else:
        assert not folder.exists(),('Partial worker exists; inspect instead of overwrite',tag)
        # Exclusive receipt reservation precedes subprocess creation.
        launch=OUT/(tag+'_launch.json')
        with launch.open('x',encoding='utf-8') as g:
            with log.open('xb') as f:
                p=subprocess.Popen([runtime,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts/replay_latent_native_v2.py'),str(cell['seed']),str(cell['horizon'])],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
                json.dump(dict(pid=p.pid,utc=now(),**cell),g);g.flush();code=p.wait()
        execution='CAPTURED_WORKER_EXIT'
    row=dict(**cell,exit_code=code,execution=execution,log_sha256=sha(log),utc=now())
    if code in (0,None):
        m=json.loads(metric.read_text(encoding='utf-8-sig'))
        assert sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(folder/'predictions.npz')==m['predictions_sha256']
        assert json.loads((folder/'SOURCE_UNCHANGED_AFTER_V1.json').read_text())['upstream_unchanged']
        row.update({k:m[k] for k in ['reproduced_mse','reproduced_mae','best_epoch','epochs','checkpoint_sha256','predictions_sha256','data_sha256','source_commit','num_parameters','frozen_ae_parameters']})
        row.update(metrics_path=str(metric),metrics_sha256=sha(metric),scientific_artifacts_complete=True)
    else:row['scientific_artifacts_complete']=False
    dump(tag+'_closure.json',row);rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'progress.json').write_text(json.dumps(dict(utc=now(),closed=len(rows),planned=6),indent=2),encoding='utf-8')
    if not row['scientific_artifacts_complete']:break
ok=len(rows)==6 and all(r['scientific_artifacts_complete'] for r in rows)
obj=dict(utc=now(),status='ALL_SIX_NATIVE_REPLAY_ARTIFACTS_VERIFIED' if ok else 'MATRIX_INCOMPLETE_ERROR_RETAINED',rows=rows,worker_sha256=sha(ROOT/'scripts/replay_latent_native_v2.py'),protocol_sha256=sha(ROOT/'PROTOCOL.md'),new_method=False,test_hyperparameter_selection=False)
dump('RESULTS_V3.json',obj);print(json.dumps({'status':obj['status'],'closed':len(rows)}),flush=True)
sys.exit(0 if ok else 1)
