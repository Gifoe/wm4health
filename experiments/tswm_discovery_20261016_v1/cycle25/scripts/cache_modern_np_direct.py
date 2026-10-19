"""Pin small official public code and primary documents; never execute fetched code."""
import hashlib,json,pathlib,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/literature';OUT.mkdir(parents=True,exist_ok=True)
def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'tswm-research-audit','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=30) as r:return r.read()
repo='danrsm/flowNP';meta=json.loads(get('https://api.github.com/repos/'+repo));branch=meta['default_branch'];commit=json.loads(get(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha'];tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'));assert not tree.get('truncated')
target=ROOT/'upstream/flownp';target.mkdir(parents=True,exist_ok=True);records=[]
for f in tree['tree']:
    if f['type']!='blob' or pathlib.Path(f['path']).suffix.lower() not in ['.py','.yaml','.yml','.toml','.md','.txt']:continue
    assert f.get('size',0)<2000000 and '..' not in pathlib.PurePosixPath(f['path']).parts
    url=f"https://raw.githubusercontent.com/{repo}/{commit}/{f['path']}";content=get(url);assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==f['sha']
    p=target/f['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content);records.append(dict(path=f['path'],bytes=len(content),git_blob=f['sha'],sha256=hashlib.sha256(content).hexdigest(),url=url))
papers=[]
for name,url in [('flownp_2512.23853v1.html','https://arxiv.org/html/2512.23853v1'),('structured_inference_uai2023.html','https://proceedings.mlr.press/v216/tailor23a.html'),('conditioning_gap_2604.19312.html','https://arxiv.org/abs/2604.19312')]:
    content=get(url);assert len(content)<3000000;(OUT/name).write_bytes(content);papers.append(dict(file=name,url=url,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
result=dict(status='pass',repo=repo,revision=commit,default_branch=branch,license_api=meta.get('license'),files=records,papers=papers,weights_downloaded=False,executed_fetched_code=False)
(OUT/'modern_np_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(status='pass',repo=repo,revision=commit,files=len(records),source_bytes=sum(x['bytes'] for x in records),papers=len(papers))))

