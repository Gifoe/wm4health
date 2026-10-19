"""Fetch public pinned source/metadata only; no model execution or weight copy."""
import pathlib,json,hashlib,urllib.request,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
REV='3212e42564493f520593e5414af4367fc4b49226';REPO='thuml/sundial-base-128m'
OUT=ROOT/'upstream_source/Sundial';OUT.mkdir(parents=True,exist_ok=True)
def fetch(path):
    errors=[]
    for host in ['https://huggingface.co','https://hf-mirror.com']:
        url=host+path
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TSWM-research-source-audit/1.0'}),timeout=25) as r:
                assert r.status==200;return r.read(),url
        except Exception as e:errors.append(type(e).__name__+':'+str(e))
    raise RuntimeError(errors)
data,url=fetch('/api/models/'+REPO+'/revision/'+REV+'?blobs=true');meta=json.loads(data);assert meta['sha']==REV
(OUT/'hf_metadata.json').write_bytes(data);files=[]
for row in meta['siblings']:
    name=row['rfilename']
    if not (name.endswith('.py') or name in ['config.json','generation_config.json','README.md','LICENSE']):continue
    assert '/' not in name and row['size']<2*1024*1024
    content,source=fetch('/'+REPO+'/resolve/'+REV+'/'+name)
    assert len(content)==row['size']
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    assert blob==row['blobId'],name
    path=OUT/name
    if path.exists():assert path.read_bytes()==content
    else:path.write_bytes(content)
    files.append({'path':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'git_blob':blob,'url':source})
manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo':REPO,'revision':REV,
          'metadata_url':url,'metadata_sha256':hashlib.sha256(data).hexdigest(),'files':files,
          'weights_downloaded':False,'no_remote_code_executed':True}
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'status':'source_only_verified','files':len(files),'bytes':sum(r['bytes'] for r in files),
     'weight_metadata':[r for r in meta['siblings'] if r['rfilename'].endswith('.safetensors')]}))
