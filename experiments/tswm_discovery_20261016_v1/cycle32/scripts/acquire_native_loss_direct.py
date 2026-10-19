"""Same pinned public files, normal TLS, process-local direct HTTP transport."""
import pathlib,json,hashlib,urllib.request,time
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'upstream/worldts_loss'
OUT.mkdir(parents=True,exist_ok=True)
REV='269426ea6217d7152a244da0e209d922acd46d4f'
prefix='DAG-bench/ts_benchmark/baselines/worldts/losses/'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
rows=[];start=time.perf_counter()
for name in ['latent_state_loss.py','sigreg_loss.py','visreg_loss.py']:
    url='https://raw.githubusercontent.com/decisionintelligence/WorldTS/'+REV+'/'+prefix+name
    req=urllib.request.Request(url,headers={'User-Agent':'TSWM-Research-Source-Audit'})
    with opener.open(req,timeout=30) as r:data=r.read(200000)
    assert 100<len(data)<100000
    text=data.decode('utf-8').replace('\r\n','\n')
    previous=ROOT.parent/'upstream/WorldTS'/prefix/name
    assert previous.read_text(encoding='utf-8').replace('\r\n','\n')==text
    path=OUT/name
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    rows.append({'path':name,'source_url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                 'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),
                 'existing_local_normalized_text_equal':True})
manifest={'repo':'decisionintelligence/WorldTS','revision':REV,'files':rows,'seconds':time.perf_counter()-start,
          'transport':'normal TLS, process-local direct opener; no environment proxy change',
          'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest),flush=True)
