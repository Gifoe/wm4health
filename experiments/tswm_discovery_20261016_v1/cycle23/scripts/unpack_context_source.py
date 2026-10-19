import base64,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'outputs/explore_context_source_payload.json').read_text(encoding='utf-8'))
rows=[]
for r in p['files']:
    assert r['encoding']=='base64'
    b=base64.b64decode(r['content']);blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    assert blob==r['sha']==r['expected_git_blob'],r['path']
    target=ROOT/'upstream/explorethecontext'/r['path'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==b
    else:target.write_bytes(b)
    rows.append(dict(path=r['path'],git_blob=blob,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
manifest=dict(repository=p['repository'],revision=p['revision'],files=rows,scope='Selected source interface audit only; no complete checkout or execution claimed.')
(ROOT/'outputs/explore_context_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),revision=p['revision'])))
