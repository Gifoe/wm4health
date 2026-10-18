"""Fresh-checkout reproduction helper: pinned public Python sources, no weights/data."""
import json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
target=ROOT/'cycle84'/'outputs'/'SOURCE_CONTRACT84_V1.json'
if target.exists():
    r=next(z for z in json.loads(target.read_text())['repositories'] if z['repository']=='ExistentialRobotics/cross_embodiment_transfer')
    assert r['commit']=='3988511598a39c8a8a6530055b72609bf1b8b373'
    print('Existing pinned C84 source registry retained; common85 validates file hashes.');raise SystemExit(0)
cfg=json.loads((ROOT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
cache=Path(cfg['cache_root'])/'cycle85'/'native_selected_v1';cache.mkdir(parents=True,exist_ok=True)
commit='3988511598a39c8a8a6530055b72609bf1b8b373'
expected={'utils.py':'636fddfaf498a9b209139ee93849909e3871ef007ee6dc353402b4fda18e2b42','td3.py':'a167fe0f5c5c95e0c019f494dcbbc3384622ce31900b0e365d0e205edcf80c14','bc.py':'bf1ff0638441715d10b904791ccf7188aa7f52ddd2b806c8d55acbf6805f59f1','align.py':'8bd0d3abef71d2497e740ca1fecc70d90a3296c20dd4a29f5bf438a9cf512a83'}
rows=[]
for name,digest in expected.items():
    p=cache/name
    if not p.exists():
        with urllib.request.urlopen('https://raw.githubusercontent.com/ExistentialRobotics/cross_embodiment_transfer/'+commit+'/'+name,timeout=40) as response:b=response.read()
        assert hashlib.sha256(b).hexdigest()==digest
        with p.open('xb') as f:f.write(b)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    rows.append(dict(path=name,server_cache=str(p),sha256=digest,bytes=p.stat().st_size))
target.parent.mkdir(parents=True,exist_ok=True)
with target.open('x',encoding='utf-8') as f:json.dump(dict(repositories=[dict(repository='ExistentialRobotics/cross_embodiment_transfer',commit=commit,selected=rows)]),f,indent=2)
print('Four pinned public source files verified; no checkpoint or dataset downloaded.')
