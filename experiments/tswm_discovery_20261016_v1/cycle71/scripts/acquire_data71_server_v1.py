"""Direct public-source prefix acquisition; preserve failed relay artifacts."""
import sys,pathlib,urllib.request,io,hashlib,json,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np
sel=read(ROOT/'SOURCE_SELECTION71_V1.json');src=PROJECT/'upstream'/'modular-metalearning_f830c6aa';sources=[]
for row in sel['files']:
 p=src/row['path'];b=p.read_bytes();assert len(b)==row['bytes'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob_sha1'];sources.append(dict(**row,sha256=sha(p),server_path=str(p)))
parent=storage('data');dest=parent/'verified_server_v1';assert not (OUT/'ACQUISITION71_V1.json').exists();dest.mkdir(exist_ok=False)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[];start=time.perf_counter()
for row in sel['data']['files']:
 url=sel['data']['origin']+row['name'];print('OPEN_PUBLIC_SOURCE',row['name'],flush=True)
 with opener.open(url,timeout=15) as f:
  assert int(f.headers['Content-Length'])==row['source_bytes'];v=np.lib.format.read_magic(f);shape,fort,dt=np.lib.format._read_array_header(f,v);assert list(shape)==row['original_shape'];n=int(np.prod(shape))*dt.itemsize if fort else int(np.prod(shape[1:]))*row['subset_rows']*dt.itemsize;raw=f.read(n);assert len(raw)==n
  arr=np.frombuffer(raw,dtype=dt).reshape(shape,order='F')[:row['subset_rows']].copy() if fort else np.frombuffer(raw,dtype=dt).reshape((row['subset_rows'],)+shape[1:]).copy();transport={k:f.headers.get(k) for k in ['Content-Length','ETag','Last-Modified']}
 b=io.BytesIO();np.save(b,arr,allow_pickle=False);payload=b.getvalue();p=dest/row['name'];expected=hashlib.sha256(payload).hexdigest()
 with p.open('xb') as out:out.write(payload)
 assert sha(p)==expected,('Direct write hash mismatch',str(p))
 rows.append(dict(**row,url=url,source_dtype=str(dt),source_fortran=fort,acquired_source_body_sha256=hashlib.sha256(raw).hexdigest(),acquired_source_body_bytes=len(raw),acquisition_kind='Direct server HTTPS; full label body then first rows' if fort else 'Direct server HTTPS; contiguous first-row prefix, remaining body not acquired',subset_shape=list(arr.shape),subset_dtype=str(arr.dtype),subset_sha256=expected,subset_bytes=len(payload),server_path=str(p),transport=transport));print(json.dumps(dict(data=row['name'],shape=list(arr.shape),bytes=len(payload),sha256=expected)),flush=True)
failed=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in parent.glob('*.npy')]
dump(OUT/'ACQUISITION71_V1.json',dict(utc=utc(),status='PINNED_SOURCES_AND_FIXED_SUBSETS_VERIFIED',repo=sel['repo'],revision=sel['revision'],source_root=str(src),data_root=str(dest),sources=sources,datasets=rows,failed_relay_files_preserved=failed,source_to_server_hashes_verified=True,original_full_state_payloads_downloaded=False,local_result_data_model_or_pdf_files_written=0,wall_seconds=time.perf_counter()-start))
print(json.dumps(dict(status='PINNED_SOURCES_AND_FIXED_SUBSETS_VERIFIED',receipt_sha256=sha(OUT/'ACQUISITION71_V1.json'),files=len(rows),seconds=time.perf_counter()-start)),flush=True)
