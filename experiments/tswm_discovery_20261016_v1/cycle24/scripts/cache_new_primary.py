import hashlib,json,pathlib,requests,time
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/literature';OUT.mkdir(exist_ok=True)
s=requests.Session();s.trust_env=False;url='https://arxiv.org/html/2607.18965v1'
r=s.get(url,timeout=40);r.raise_for_status();b=r.content;assert len(b)<5*1024*1024
f=OUT/'variational_meta_sysid_2607_18965v1.html';f.write_bytes(b)
receipt=dict(url=url,path=f.name,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),tls_verified=True,retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='Primary method/interface review; no native execution or mathematical correctness endorsement.')
(OUT/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
