"""Cache bounded primary documents; metadata-only checks for large model archives."""
import concurrent.futures,hashlib,json,pathlib,urllib.request,time
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/literature';OUT.mkdir(parents=True,exist_ok=True)
sources={
 'metaid_2610_13411.html':'https://arxiv.org/html/2610.13411v1',
 'conditioning_2604_19312.pdf':'https://arxiv.org/pdf/2604.19312v1',
 'martingale_2304_09431.pdf':'https://arxiv.org/pdf/2304.09431',
 'drift_coherence_icml2026.html':'https://proceedings.mlr.press/v306/kim26ah.html',
 'explore_context_aistats2021.html':'https://proceedings.mlr.press/v130/achterhold21a.html',
 'explore_context_aistats2021.pdf':'https://proceedings.mlr.press/v130/achterhold21a/achterhold21a.pdf',
 'explore_context_supplement.pdf':'https://proceedings.mlr.press/v130/achterhold21a/achterhold21a-supp.pdf'}
def fetch(item):
    name,url=item;path=OUT/name
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-provenance-audit/1.0'}),timeout=40) as r:
            content=r.read(12*1024*1024+1);assert len(content)<=12*1024*1024
            if name.endswith('.pdf'):assert content.startswith(b'%PDF')
            path.write_bytes(content)
            return dict(path=name,url=url,resolved_url=r.url,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),status='downloaded')
    except Exception as e:return dict(path=name,url=url,status='failed',error=repr(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(fetch,sources.items()))
archives=[]
for label,url in [('corrected_models','https://keeper.mpdl.mpg.de/f/988ed8adf071486c80dc/?dl=1')]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=25) as r:
            archives.append(dict(label=label,url=url,status=r.status,resolved_url=r.url,content_length=r.headers.get('Content-Length'),content_type=r.headers.get('Content-Type'),downloaded=False))
    except Exception as e:archives.append(dict(label=label,url=url,error=repr(e),downloaded=False))
(OUT/'manifest.json').write_text(json.dumps(dict(retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sources=rows,archive_metadata=archives),indent=2),encoding='utf-8')
print(json.dumps(dict(sources=rows,archive_metadata=archives)),flush=True)
