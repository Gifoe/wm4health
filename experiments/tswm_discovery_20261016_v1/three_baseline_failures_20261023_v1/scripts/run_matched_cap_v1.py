"""Twelve fixed100-epoch controls; stop on implementation failure, never reroll a cell."""
import pathlib,json,sys,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/matched_cap_v1';OUT.mkdir(exist_ok=False)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));ART=pathlib.Path(cfg['artifact_root'])/ROOT.name
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
matrix=[dict(mode=mode,seed=s,horizon=h) for h in (336,720) for s in (42,43,44) for mode in ('latent','raw')]
dump('MATRIX_V1.json',dict(utc=now(),matrix=matrix,epochs=100,protocol_sha256=sha(ROOT/'MATCHED_CAP_PROTOCOL_V1.md'),test_hyperparameter_tuning=False))
rows=[]
for cell in matrix:
    mode=cell['mode'];tag=f"Electricity_DLinear_H{cell['horizon']}_seed{cell['seed']}";folder=ART/('matched_cap_'+mode+'_v1')/tag
    assert not folder.exists(),('Existing scientific cell; inspect instead of restarting',str(folder))
    log=OUT/(mode+'_'+tag+'_stdout.log');launch=OUT/(mode+'_'+tag+'_launch.json')
    with launch.open('x') as g:
        with log.open('xb') as f:
            child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/('matched_cap_'+mode+'_v1.py')),str(cell['seed']),str(cell['horizon'])],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            json.dump(dict(pid=child.pid,utc=now(),**cell),g);g.flush();code=child.wait()
    row=dict(**cell,exit_code=code,utc=now(),log_sha256=sha(log))
    if code==0:
        m=json.loads((folder/'metrics.json').read_text(encoding='utf-8-sig'));assert m['epochs']==100 and m['early_stop_break_ignored']
        assert sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(folder/'predictions.npz')==m['predictions_sha256']
        row.update({k:m[k] for k in ['reproduced_mse','reproduced_mae','best_epoch','epochs','checkpoint_sha256','predictions_sha256','num_parameters','frozen_ae_parameters']});row.update(metrics_path=str(folder/'metrics.json'),metrics_sha256=sha(folder/'metrics.json'))
    dump(mode+'_'+tag+'_exit.json',row);rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'progress.json').write_text(json.dumps(dict(utc=now(),closed=len(rows),planned=12,last_cell=cell),indent=2),encoding='utf-8')
    if code:break
ok=len(rows)==12 and all(x['exit_code']==0 for x in rows)
dump('RESULTS_V1.json',dict(utc=now(),status='TWELVE_FIXED100_CONTROLS_CLOSED' if ok else 'MATCHED_CAP_ERROR_RETAINED',rows=rows,protocol_sha256=sha(ROOT/'MATCHED_CAP_PROTOCOL_V1.md'),new_method=False))
sys.exit(0 if ok else 1)
