"""Portable path and integrity helpers for the unchanged upstream reproduction."""
import datetime
import hashlib
import json
import os
import pathlib
import sys

R=pathlib.Path(__file__).resolve().parents[1]
P=R.parent
cfg_path=pathlib.Path(os.environ.get('TSWM_STORAGE_CONFIG',str(P/'STORAGE_LOCATION_V1.json')))
STORAGE=json.loads(cfg_path.read_text(encoding='utf-8'))
UP=pathlib.Path(os.environ.get('TSWM_UPSTREAM_ROOT',str(P/'upstream/TSWM-Official-Repo')))
DATA=pathlib.Path(STORAGE['data_root'])/'greenhouse3_original_v1/experiment_1/TimeSeries'
CACHE=pathlib.Path(STORAGE['cache_root'])/'cycle53/native_windows_v1'
CKPT=pathlib.Path(STORAGE['checkpoint_root'])/'cycle53/native_v1'
ART=pathlib.Path(STORAGE['artifact_root'])/'cycle53/native_v1'
CODEC=CKPT/'shared_codec'
for path in [UP,UP/'rq2_latent_vs_ob',UP/'rq2_latent_vs_ob/common',UP/'RQ3_fusion_benchmark',UP/'RQ4_action_encoding',UP/'RQ7_causal_fusion']:
    sys.path.insert(0,str(path))

def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding='utf-8')
def check_source():
    records=json.loads((R/'outputs/native_probe_v2/source_manifest.json').read_text())
    for entry in records: assert sha(UP/entry['path'])==entry['sha256'],entry['path']
    return len(records)
