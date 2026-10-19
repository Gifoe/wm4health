"""Pin public source snapshots before choosing any experimental direction."""
import concurrent.futures,hashlib,io,json,pathlib,requests,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPOS=['Muyiiiii/LatentTSF','decisionintelligence/WorldTS','HaochenZhang717/TSWM-Official-Repo','gatsby-sahani/rp-gssm','ARBrachet/PRO-DYN','thummd/dotime','complex-ai-lab/epi-cf-benchmark','zqiao11/TIME']
def one(repo):
    s=requests.Session();s.trust_env=False;s.headers['User-Agent']='tswm-research-audit'
    def get(url):
        r=s.get(url,timeout=90);r.raise_for_status();return r
    meta=get('https://api.github.com/repos/'+repo).json()
    commit=get('https://api.github.com/repos/'+repo+'/commits/'+meta['default_branch']).json()
    sha=commit['sha'];name=repo.split('/')[-1]
    d=ROOT/'upstream'/name;d.mkdir(parents=True,exist_ok=True)
    record={'repo':repo,'revision':sha,'commit_date':commit['commit']['committer']['date'],'license':meta.get('license'),'size_kb':meta['size'],'url':meta['html_url']}
    # Do not fetch large datasets or benchmark payloads during source screening.
    if name in ['TIME','epi-cf-benchmark'] or meta['size']>180000:
        tree=get('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1').json()
        (d/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        for f in ['README.md','requirements.txt','pyproject.toml']:
            r=s.get('https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+f,timeout=45)
            if r.status_code==200:(d/f).write_bytes(r.content)
        record['download_scope']='metadata/readme/tree only'
    else:
        data=get('https://codeload.github.com/'+repo+'/zip/'+sha).content
        record['archive_sha256']=hashlib.sha256(data).hexdigest();record['archive_bytes']=len(data)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in z.infolist():
                rel=pathlib.PurePosixPath(member.filename)
                if len(rel.parts)<2 or member.is_dir():continue
                out=d.joinpath(*rel.parts[1:]).resolve()
                assert out.is_relative_to(d.resolve())
                out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(z.read(member))
        record['download_scope']='exact source archive'
    (d/'SOURCE_PIN.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record
if __name__=='__main__':
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for f in concurrent.futures.as_completed({pool.submit(one,r):r for r in REPOS}):
            try:rec=f.result();records.append(rec);print(json.dumps(rec),flush=True)
            except Exception as e:print(type(e).__name__,str(e),flush=True);raise
    (ROOT/'outputs/source_registry.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
