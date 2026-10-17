"""Independent source-controlled two-system audit paths and immutable receipts."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,o):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,allow_nan=False)
def paths():
    s=read(PROJECT/'STORAGE_LOCATION_V1.json');d={k:pathlib.Path(s[k])/'cycle60' for k in ['data_root','checkpoint_root','artifact_root']}
    for v in d.values():v.mkdir(parents=True,exist_ok=True)
    return d
def source_check():
    o=read(OUT/'DOWNLOAD60_V1.json')
    for row in o['files']:assert sha(row['server_path'])==row['sha256'],row['path']
    return o
