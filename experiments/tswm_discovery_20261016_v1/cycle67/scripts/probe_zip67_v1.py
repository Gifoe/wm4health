import urllib.request,struct,json,pathlib,time,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];p=ROOT/'outputs'/'ZIP_END_PROBE67_V1.json';assert not p.exists();start=time.perf_counter();size=657187340;url='https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip'
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={size-22}-{size-1}'}),timeout=15) as r:
  assert r.status==206;b=r.read();assert len(b)==22 and b[:4]==b'PK\x05\x06';values=struct.unpack('<4s4H2LH',b)
  obj=dict(status='EOCD_VERIFIED',entries=values[4],central_directory_bytes=values[5],central_directory_offset=values[6],archive_comment_bytes=values[7],elapsed=time.perf_counter()-start,etag=r.headers['ETag'])
except Exception as e:obj=dict(status='PROBE_FAILED',error_type=type(e).__name__,error=str(e),elapsed=time.perf_counter()-start)
obj['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
with p.open('x') as f:json.dump(obj,f,indent=2)
print(json.dumps(obj))
