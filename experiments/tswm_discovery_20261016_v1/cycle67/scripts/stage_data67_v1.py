"""Selective public ZIP reads; keep numerical records only on server G."""
import pathlib,json,hashlib,datetime,urllib.request,io,zipfile,re,sys,csv,collections
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));base=(pathlib.Path(cfg['data_root'])/'cycle67'/'cgmacros_v1').resolve();base.mkdir(exist_ok=True,parents=True);assert base.drive.upper()=='G:'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
class RangeZip(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache={};self.block=262144;self.etag=None;self.requests=[]
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,n,whence=0):
  self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n;assert 0<=self.pos<=self.size;return self.pos
 def read(self,n=-1):
  n=min(self.size-self.pos,self.size if n<0 else n);parts=[]
  while n:
   block=self.pos//self.block
   if block not in self.cache:
    start=block*self.block;end=min(self.size,start+self.block)-1;req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=60) as r:
     assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{self.size}';etag=r.headers['ETag'];b=r.read();assert len(b)==end-start+1
     if self.etag is None:self.etag=etag
     assert self.etag==etag,'Archive identity changed during selective reads'
    self.cache[block]=b;self.requests.append(dict(start=start,end=end,bytes=len(b),sha256=sha(b)))
   b=self.cache[block];offset=self.pos%self.block;take=min(n,len(b)-offset);assert take>0;parts.append(b[offset:offset+take]);self.pos+=take;n-=take
  return b''.join(parts)
metadata=[]
for name in ['SHA256SUMS.txt','DataDictionary_CGMacros-00X.csv']:
 u='https://physionet.org/files/cgmacros/1.0.0/'+name
 with urllib.request.urlopen(u,timeout=60) as r:raw=r.read()
 p=base/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
 metadata.append(dict(name=name,url=u,sha256=sha(raw),bytes=len(raw)))
archive='https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip';stream=RangeZip(archive,657187340)
with zipfile.ZipFile(stream) as z:
 entries=z.infolist();csvs=[x for x in entries if x.filename.lower().endswith('.csv')];chosen=[x for x in csvs if re.search(r'(^|/)CGMacros-\d{3}\.csv$',x.filename,re.I)]
 assert len(chosen)==45,[(x.filename,x.file_size) for x in csvs]
 inventory=[dict(path=x.filename,uncompressed_bytes=x.file_size,compressed_bytes=x.compress_size,CRC32=f'{x.CRC:08x}',compression=x.compress_type,header_offset=x.header_offset) for x in entries];records=[]
 for info in sorted(chosen,key=lambda x:x.filename):
  raw=z.read(info);assert len(raw)==info.file_size;name=pathlib.PurePosixPath(info.filename).name;target=(base/name).resolve();assert target.parent==base
  if target.exists():assert target.read_bytes()==raw
  else:
   with target.open('xb') as f:f.write(raw)
  parsed=csv.reader(io.StringIO(raw.decode('utf-8-sig')));header=next(parsed);rows=list(parsed)
  records.append(dict(member=info.filename,path=str(target),bytes=len(raw),sha256=sha(raw),CRC32=f'{info.CRC:08x}',rows=len(rows),columns=header))
  print(json.dumps(dict(subject_file=name,rows=len(rows),files=len(records))),flush=True)
ap=pathlib.Path(cfg['artifact_root'])/'cycle67';ap.mkdir(exist_ok=True,parents=True);dump(ap/'SOURCE_ZIP_DIRECTORY67_V1.json',dict(utc=utc(),url=archive,bytes=657187340,etag=stream.etag,entries=inventory))
obj=dict(utc=utc(),status='45_PUBLIC_NUMERIC_PARTICIPANT_FILES_VERIFIED',dataset='CGMacros',version='1.0.0',license='CC-BY-NC-SA-4.0',source_page='https://physionet.org/content/cgmacros/1.0.0/',archive_url=archive,archive_bytes=657187340,archive_etag=stream.etag,archive_published_SHA256SUMS=(base/'SHA256SUMS.txt').read_text(),whole_archive_sha256_verified=False,reason='Selective numerical-member acquisition avoids food photographs; each ZIP member CRC and retained SHA256 verified, whole archive not acquired.',range_requests=stream.requests,network_bytes=sum(x['bytes'] for x in stream.requests),files=records,metadata=metadata,source_directory_artifact=str(ap/'SOURCE_ZIP_DIRECTORY67_V1.json'),source_directory_sha256=sha((ap/'SOURCE_ZIP_DIRECTORY67_V1.json').read_bytes()),model_execution=False,local_result_files_written=0)
dump(OUT/'DATA_SOURCE67_V1.json',obj);print(json.dumps(dict(status=obj['status'],files=len(records),network_bytes=obj['network_bytes'],columns=records[0]['columns'])),flush=True)
