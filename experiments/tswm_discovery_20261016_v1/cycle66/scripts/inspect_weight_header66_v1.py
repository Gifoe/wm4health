import pathlib,json,struct,collections,math
ROOT=pathlib.Path(__file__).resolve().parents[1];cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));p=pathlib.Path(cfg['checkpoint_root'])/'cycle66'/'normwear2_3d1281c'/'model.safetensors'
with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];assert 0<n<10**7;header=json.loads(f.read(n))
rows={k:v for k,v in header.items() if k!='__metadata__'}
obj=dict(tensors=len(rows),state_elements=sum(math.prod(v['shape']) for v in rows.values()),top_level_prefix_counts=dict(collections.Counter(k.split('.')[0] for k in rows)),first_keys=list(rows)[:20],metadata=header.get('__metadata__'))
with (ROOT/'outputs'/'WEIGHT_HEADER66_V1.json').open('x') as f:json.dump(obj,f,indent=2)
print(json.dumps(obj))
