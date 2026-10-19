"""One public, pinned safetensors file plus official demonstration source; normal TLS."""
import hashlib,json,pathlib,time,urllib.request,shutil,re
ROOT=pathlib.Path(__file__).resolve().parents[1];FOLDER=ROOT/'upstream/normwear2'
def main():
    started=time.monotonic();manifest=json.loads((FOLDER/'source_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['source_files']:assert hashlib.sha256((FOLDER/row['path']).read_bytes()).hexdigest()==row['sha256']
    expected=next(r for r in manifest['weights_not_downloaded'] if r['rfilename']=='model.safetensors')
    assert expected['size']==748201812 and expected['lfs']['sha256']=='44ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11'
    assert shutil.disk_usage(FOLDER).free>2*expected['size']
    base=manifest['endpoint'];prefix=base+'/'+manifest['repo']+'/resolve/'+manifest['revision']+'/'
    def request(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'tswm-public-reproduction/1.0'}),timeout=30)
    with request(base+'/api/models/'+manifest['repo']+'/revision/'+manifest['revision']+'?blobs=true') as r:info=json.load(r)
    demo=next(r for r in info['siblings'] if r['rfilename']=='demo_code.ipynb')
    with request(prefix+'demo_code.ipynb') as response:body=response.read(20_000_001)
    assert len(body)<=20_000_000
    gitsha=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest();assert gitsha==demo['blobId']
    (FOLDER/'demo_code.ipynb').write_bytes(body)
    demo_info=dict(path='demo_code.ipynb',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),git_blob=gitsha)
    path=FOLDER/'model.safetensors';partial=path.with_suffix('.safetensors.part')
    if path.exists():assert path.stat().st_size==expected['size'] and hashlib.sha256(path.read_bytes()).hexdigest()==expected['lfs']['sha256']
    else:
        assert not partial.exists(),'preserve and inspect existing partial download before retrying'
        h=hashlib.sha256();count=0
        with request(prefix+'model.safetensors') as response,partial.open('xb') as f:
            while data:=response.read(4*1024*1024):
                count+=len(data);assert count<=expected['size'];h.update(data);f.write(data)
        assert count==expected['size'] and h.hexdigest()==expected['lfs']['sha256']
        partial.replace(path)
    result=dict(status='weights_downloaded_and_verified_not_executed',repo=manifest['repo'],revision=manifest['revision'],endpoint=base,
                path=path.name,bytes=expected['size'],sha256=expected['lfs']['sha256'],demo=demo_info,seconds=time.monotonic()-started,
                script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((ROOT/'NORMWEAR_QUALIFICATION_PLAN.md').read_bytes()).hexdigest())
    (FOLDER/'weight_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
