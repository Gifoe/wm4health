"""Server-only numeric file inspection; never load the unverified checkpoint."""
import pathlib,json,hashlib,datetime,sys,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
base=(pathlib.Path(cfg['checkpoint_root'])/'cycle66'/'normwear2_3d1281c').resolve()
assert base.drive.upper()=='G:' and base.is_relative_to(pathlib.Path(cfg['checkpoint_root']).resolve())
old=(base/'model.safetensors.ramrelay.part').resolve();candidate=(base/'model.safetensors.repair1.part').resolve();final=(base/'model.safetensors').resolve()
assert all(p.parent==base for p in [old,candidate,final]);size=748201812;expected='44ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11';chunk=4*1024*1024
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(chunk):h.update(b)
 return h.hexdigest()
def write(name,obj):
 with (ROOT/'outputs'/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)
mode=sys.argv[1]
if mode=='prepare':
 assert old.stat().st_size==size and not candidate.exists() and not final.exists()
 h=hashlib.sha256();rows=[]
 with old.open('rb') as src,candidate.open('xb') as dst:
  while b:=src.read(chunk):
   dst.write(b);h.update(b);rows.append(hashlib.sha256(b).hexdigest())
 assert digest(candidate)==h.hexdigest()
 obj=dict(utc=utc(),status='REPAIR_CANDIDATE_PREPARED_NOT_LOADABLE',original_sha256=h.hexdigest(),candidate_sha256=digest(candidate),chunk_bytes=chunk,chunk_sha256=rows,bytes=size,expected_sha256=expected,server_candidate=str(candidate),original_retained=True,model_execution=False)
 write('TRANSPORT_REPAIR_PREPARE66_V1.json',obj);print(json.dumps(obj))
elif mode=='chunk':
 i=int(sys.argv[2]);assert 0<=i<(size+chunk-1)//chunk
 with candidate.open('rb') as f:f.seek(i*chunk);b=f.read(chunk)
 print(json.dumps(dict(index=i,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))))
elif mode=='finalize':
 assert candidate.stat().st_size==size and digest(candidate)==expected and not final.exists()
 candidate.rename(final);assert digest(final)==expected
 obj=dict(utc=utc(),status='CHECKPOINT_SHA256_VERIFIED_NOT_EXECUTED',server_path=str(final),bytes=size,sha256=expected,original_bad_file_retained=True,model_execution=False,local_model_or_result_files_written=0)
 write('TRANSPORT_REPAIR_FINAL66_V1.json',obj);print(json.dumps(obj))
else:raise ValueError(mode)
