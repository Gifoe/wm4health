import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));p=pathlib.Path(cfg['checkpoint_root'])/'cycle66'/'normwear2_3d1281c'/'model.safetensors.ramrelay.part';rows=[]
for i in range(3):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 rows.append(h.hexdigest())
expected='44ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11';obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='VERIFIED' if all(x==expected for x in rows) else 'HASH_MISMATCH_DO_NOT_LOAD',server_path=str(p),bytes=p.stat().st_size,expected_sha256=expected,sha256_repeats=rows,model_execution=False)
with (ROOT/'outputs'/'PARTIAL_HASH_VERIFICATION66_V1.json').open('x') as f:json.dump(obj,f,indent=2)
print(json.dumps(obj))
