import base64,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];rows=[]
for name in ['explore_context_source_payload.json','explore_context_extra_source_payload.json']:
    p=json.loads((ROOT/'outputs'/name).read_text(encoding='utf-8'))
    for r in p['files']:
        b=base64.b64decode(r['content']);h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert h==r['sha']==r['expected_git_blob']
        f=ROOT/'upstream/explorethecontext'/r['path'];f.parent.mkdir(parents=True,exist_ok=True)
        if f.exists():assert f.read_bytes()==b
        else:f.write_bytes(b)
        rows.append(dict(path=r['path'],sha256=hashlib.sha256(b).hexdigest(),git_blob=h,bytes=len(b)))
(ROOT/'outputs/explore_context_all_source_manifest.json').write_text(json.dumps(dict(repository=p['repository'],revision=p['revision'],files=rows,scope='All non-notebook source Python plus README, requirements, licenses and published job commands; Git blob verified.'),indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows))))
