"""Fetch an exact public release directly on server; normal TLS and content hash."""
import runtime_profile
import hashlib,json,pathlib,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
REV='ce49c06f05efeb50bd307940392af0d8c89a3796';REPO='software-competence-center-hagenberg/PCFM'
EXPECTED='746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755';SIZE=211418848

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 out=ROOT/'outputs/external/PCFM';out.mkdir(parents=True,exist_ok=True);final=out/'model.safetensors';attempts=[];started=time.monotonic()
 if final.exists():assert final.stat().st_size==SIZE and digest(final)==EXPECTED;print('Already verified complete file',flush=True);return
 for host,tag in [('https://hf-mirror.com','mirror'),('https://huggingface.co','official')]:
  url=f'{host}/{REPO}/resolve/{REV}/model.safetensors';part=out/f'model.{tag}.part'
  try:
   n=0;h=hashlib.sha256()
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ReproducibleResearchAudit'}),timeout=25) as r,part.open('wb') as f:
    while b:=r.read(65536):f.write(b);h.update(b);n+=len(b)
   assert n==SIZE and h.hexdigest()==EXPECTED,(n,h.hexdigest());part.replace(final)
   receipt=dict(status='verified',repository=REPO,revision=REV,url=url,path=final.name,bytes=n,sha256=EXPECTED,elapsed_s=time.monotonic()-started,attempts=attempts)
   (out/'acquisition.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True);return
  except Exception as e:attempts.append(dict(host=host,error=type(e).__name__+':'+str(e)));print(json.dumps(attempts[-1]),flush=True)
 (out/'remote_download_attempts.json').write_text(json.dumps(dict(status='remote_network_unavailable',attempts=attempts),indent=2),encoding='utf-8');raise RuntimeError('Direct public download unavailable;retain attempts,use exact-hash alternate transport')
if __name__=='__main__':main()
