"""Retry metadata acquisition with Requests' verified CA bundle, never disable TLS."""
import concurrent.futures,hashlib,json,pathlib,requests,time
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/literature'
old=json.loads((OUT/'manifest.json').read_text());(OUT/'manifest_urllib_attempt1.json').write_text(json.dumps(old,indent=2),encoding='utf-8')
def fetch(row):
    if row['status']=='downloaded':return row
    try:
        s=requests.Session();s.trust_env=False
        r=s.get(row['url'],timeout=40);r.raise_for_status();b=r.content;assert len(b)<=12*1024*1024
        if row['path'].endswith('.pdf'):assert b.startswith(b'%PDF')
        (OUT/row['path']).write_bytes(b)
        return dict(path=row['path'],url=row['url'],resolved_url=r.url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='downloaded',tls_verification=True)
    except Exception as e:return dict(path=row['path'],url=row['url'],status='failed',error=repr(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,old['sources']))
meta=[]
for row in old['archive_metadata']:
    try:
        s=requests.Session();s.trust_env=False;r=s.head(row['url'],timeout=25,allow_redirects=True);r.raise_for_status()
        meta.append(dict(label=row['label'],url=row['url'],status=r.status_code,resolved_url=r.url,content_length=r.headers.get('Content-Length'),content_type=r.headers.get('Content-Type'),downloaded=False,tls_verification=True))
    except Exception as e:meta.append(dict(label=row['label'],url=row['url'],error=repr(e),downloaded=False))
result=dict(retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sources=rows,archive_metadata=meta,first_attempt_retained='manifest_urllib_attempt1.json')
(OUT/'manifest_requests_attempt2.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
