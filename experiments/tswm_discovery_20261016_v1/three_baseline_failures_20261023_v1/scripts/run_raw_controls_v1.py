import pathlib,sys,json,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/raw_control_v1';OUT.mkdir(exist_ok=False)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));ART=pathlib.Path(cfg['artifact_root'])/ROOT.name/'raw_control_v1';ART.mkdir(exist_ok=True,parents=True)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
matrix=[dict(seed=s,horizon=h) for h in (336,720) for s in (42,43,44)];rows=[]
dump('MATRIX_V1.json',dict(utc=now(),matrix=matrix,protocol_sha256=sha(ROOT/'RAW_MODE_CONTROL_PROTOCOL_V1.md')))
for c in matrix:
    tag=f"Electricity_DLinear_H{c['horizon']}_seed{c['seed']}";folder=ART/tag;assert not folder.exists()
    launch=OUT/(tag+'_launch.json');log=ART/(tag+'_stdout.log')
    with launch.open('x') as g:
        with log.open('xb') as f:
            p=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts/ordinary_raw_control_v1.py'),str(c['seed']),str(c['horizon'])],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            json.dump(dict(pid=p.pid,utc=now(),**c),g);g.flush();code=p.wait()
    row=dict(**c,exit_code=code,log_sha256=sha(log))
    if code==0:
        m=json.loads((folder/'metrics.json').read_text(encoding='utf-8-sig'));row.update(m);row.update(metrics_path=str(folder/'metrics.json'),metrics_sha256=sha(folder/'metrics.json'))
    dump(tag+'_exit.json',row);rows.append(row);print(json.dumps(row),flush=True)
    if code:break
ok=len(rows)==6 and all(x['exit_code']==0 for x in rows)
dump('RESULTS_V1.json',dict(utc=now(),status='ALL_SIX_RAW_MODE_CONTROLS_CLOSED' if ok else 'RAW_CONTROL_ERROR_RETAINED',rows=rows,new_method=False))
sys.exit(0 if ok else 1)
