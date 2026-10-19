"""Pin public model metadata and small source files only; never execute downloaded code."""
import hashlib,json,pathlib,re,time,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'upstream/normwear2';OUT.mkdir(parents=True,exist_ok=True)
REPO='mosaic-laboratory/normwear2'
def fetch(url,limit=2_000_000):
    request=urllib.request.Request(url,headers={'User-Agent':'tswm-research-source-audit/1.0'})
    with urllib.request.urlopen(request,timeout=25) as response:
        body=response.read(limit+1)
        if len(body)>limit:raise ValueError('small-source limit exceeded')
        return body
def main():
    started=time.monotonic();errors=[];metadata=None;endpoint=None
    for base in ['https://hf-mirror.com','https://huggingface.co']:
        try:
            metadata=json.loads(fetch(base+'/api/models/'+REPO+'?blobs=true'));endpoint=base;break
        except Exception as exc:errors.append(dict(endpoint=base,type=type(exc).__name__,message=str(exc)[:250]))
    if metadata is None:raise RuntimeError(json.dumps(errors))
    revision=metadata['sha'];assert re.fullmatch('[0-9a-f]{40}',revision)
    source=[];weights=[]
    for entry in metadata['siblings']:
        name=entry['rfilename'];p=pathlib.PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts
        if name.endswith(('.safetensors','.bin','.pt','.pth')):
            weights.append(entry);continue
        if p.suffix.lower() not in ['.py','.json','.md','.txt','.yaml','.yml'] and p.name.lower() not in ['license','notice']:continue
        payload=fetch(endpoint+'/'+REPO+'/resolve/'+revision+'/'+name)
        target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        digest=hashlib.sha256(payload).hexdigest()
        gitsha=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        if entry.get('blobId'):assert gitsha==entry['blobId'],name
        source.append(dict(path=name,bytes=len(payload),sha256=digest,git_blob=gitsha))
    result=dict(status='source_pinned_not_executed',repo=REPO,revision=revision,endpoint=endpoint,source_files=source,weights_not_downloaded=weights,
                endpoint_errors=errors,seconds=time.monotonic()-started,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    (OUT/'source_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
