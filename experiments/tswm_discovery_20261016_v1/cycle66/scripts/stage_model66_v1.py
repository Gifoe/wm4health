"""Public pinned files to server G storage only; no execution or local sync."""
import pathlib,json,hashlib,datetime,urllib.request,os,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True);PROJECT=ROOT.parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(obj):
 p=OUT/'download_state66_v1.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'utc':utc(),**obj},indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
cfg=json.loads((PROJECT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));base=(pathlib.Path(cfg['checkpoint_root'])/'cycle66'/'normwear2_3d1281c').resolve();base.mkdir(exist_ok=True,parents=True)
repo='mosaic-laboratory/normwear2';rev='3d1281c9994140056edfbbdedab57f2e6493dfaa'
files=[('README.md',12093,'53867226d7832594b6f05e74f858a9142933c575'),('__init__.py',97,'aadc86b6bb4d9fc51cb56ab9fa8231dc31247325'),('config.json',567,'713267e2b8da62c7503e2fc6690925dc6430ee05'),('configuration_normwear.py',1548,'bf2a6c9081760f08ddcacad9f2ade1611f42b154'),('latent_bayesian.py',11051,'f3aabe09b14c396330f5db53b41d466e95431446'),('layers.py',20259,'61d9fb45c4414fa44305765692e5c17aab664c61'),('modeling_normwear.py',2019,'68565b29ec2c05bfaed3de9831a1fd5cda561279'),('normwear2.py',30697,'a8757e0360559e4a9afe918d9c8bf9a5071034b0'),('utils.py',853,'88c19c72c213915aecc802c9372d49d32a4bc1f4'),('model.safetensors',748201812,'44ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11')]
rows=[]
for name,size,digest in files:
 target=(base/name).resolve();assert base in target.parents;part=target.with_suffix(target.suffix+'.part');assert base in part.resolve().parents
 if target.exists():
  assert target.stat().st_size==size
 else:
  assert not part.exists(),'Partial download retained. Inspect and resume it explicitly; do not restart blindly.'
  opened=None
  for endpoint in ['https://hf-mirror.com','https://huggingface.co']:
   try:
    opened=urllib.request.urlopen(endpoint+'/'+repo+'/resolve/'+rev+'/'+name,timeout=90);break
   except Exception as ex:
    state(dict(status='TRY_NEXT_PUBLIC_ENDPOINT',file=name,error_type=type(ex).__name__))
  assert opened is not None,('No public endpoint opened',name)
  count=0
  with opened as response,part.open('xb') as out:
   while True:
    chunk=response.read(1024*1024)
    if not chunk:break
    out.write(chunk);count+=len(chunk)
    if count%(32*1024*1024)==0:state(dict(status='DOWNLOADING',file=name,bytes_received=count,expected_bytes=size))
  assert count==size,(name,count,size)
  if name=='model.safetensors':assert sha(part)==digest
  else:
   raw=part.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==digest
  assert part.resolve().parent==base and target.resolve().parent==base and not target.exists();part.replace(target)
 if name=='model.safetensors':assert sha(target)==digest
 else:
  raw=target.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==digest
 rows.append(dict(path=name,server_path=str(target),bytes=size,sha256=sha(target),source_object_id=digest,repository=repo,revision=rev));state(dict(status='PINNED_FILE_VERIFIED',file=name,completed_files=len(rows)))
receipt=dict(utc=utc(),status='PRETRAINED_MODEL_FILES_VERIFIED_NOT_EXECUTED',repository=repo,revision=rev,server_directory=str(base),files=rows,downloaded_bytes=sum(r['bytes'] for r in rows),parameters=187042577,license_declared_by_model_card='Apache-2.0',duplicate_bin_downloaded=False,raw_data_downloaded=False,model_execution=False,accepted_final_directions=0)
with (OUT/'MODEL_FILES66_V1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
state(dict(status=receipt['status'],files=len(rows),bytes=receipt['downloaded_bytes']));print(json.dumps(dict(status=receipt['status'],files=len(rows),bytes=receipt['downloaded_bytes'],receipt_sha256=sha(OUT/'MODEL_FILES66_V1.json'))))
