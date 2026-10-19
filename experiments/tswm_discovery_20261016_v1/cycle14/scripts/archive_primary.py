"""Archive bounded primary-source documents without executing downloaded code."""
import runtime_profile
import datetime, hashlib, json, pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/literature'
SOURCES=[
 ('causal_normalization_2606.09954v1','https://arxiv.org/pdf/2606.09954v1','pdf'),
 ('revin_diagnostics_2510.04667v1','https://arxiv.org/pdf/2510.04667v1','pdf'),
 ('oneworld_2609.30946v1','https://arxiv.org/pdf/2609.30946v1','pdf'),
 ('deconfounding_rl_1812.10576_latest','https://arxiv.org/pdf/1812.10576','pdf'),
 ('mdn_vjepa_2610.13956v1','https://arxiv.org/pdf/2610.13956v1','pdf'),
 ('bayesian_context_readme','https://raw.githubusercontent.com/boschresearch/bayesian-context-aggregation/master/readme.md','md'),
]
def main():
 OUT.mkdir(parents=True,exist_ok=True); records=[]
 manifest=OUT/'manifest.json'
 if manifest.exists() and not (OUT/'initial_download_attempt.json').exists():
  (OUT/'initial_download_attempt.json').write_bytes(manifest.read_bytes())
 session=requests.Session();session.trust_env=False
 for name,url,ext in SOURCES:
  path=OUT/(name+'.'+ext)
  record={'name':name,'requested_url':url,'path':path.relative_to(ROOT).as_posix()}
  try:
   if path.exists():data=path.read_bytes();record['existing']=True
   else:
    with session.get(url,timeout=(8,35),stream=True) as response:
     response.raise_for_status()
     record['resolved_url']=response.url;record['content_type']=response.headers.get('Content-Type')
     parts=[];size=0
     for part in response.iter_content(1024*1024):
      size+=len(part);assert size<=20*1024*1024,'Exceeded document size limit'
      parts.append(part)
     data=b''.join(parts)
    assert len(data)<=20*1024*1024,'Exceeded bounded primary-document download'
    if ext=='pdf':assert data.startswith(b'%PDF-'),'Not a PDF; do not treat challenge/HTML as paper'
    if ext=='md':assert b'Bayesian Context Aggregation' in data
    path.write_bytes(data)
   record.update(status='archived',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  except Exception as error:record.update(status='unavailable',error=type(error).__name__+': '+str(error)[:250])
  records.append(record);print(json.dumps({'document':name,'status':record['status']}),flush=True)
 payload={'sources':records,'downloaded_code_executed':False,'scope':'Primary evidence archive; no paper-result reproduction implied',
          'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 manifest.write_text(json.dumps(payload,indent=2),encoding='utf-8')
if __name__=='__main__':main()
