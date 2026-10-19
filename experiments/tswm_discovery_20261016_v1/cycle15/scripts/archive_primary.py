"""Archive primary papers for the persistent-context novelty boundary."""
import requests,pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
PAPERS=[('mts3_latest','https://arxiv.org/pdf/2310.18534'),('hiprssm_v3','https://arxiv.org/pdf/2206.14697v3'),
 ('pets_2018','https://proceedings.neurips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models.pdf'),
 ('explore_context_2021','https://proceedings.mlr.press/v130/achterhold21a/achterhold21a.pdf')]
def main():
 out=ROOT/'outputs/literature';out.mkdir(parents=True,exist_ok=True);s=requests.Session();s.trust_env=False;rows=[]
 for name,url in PAPERS:
  p=out/(name+'.pdf');row={'name':name,'requested_url':url,'path':p.relative_to(ROOT).as_posix()}
  try:
   if not p.exists():
    response=s.get(url,timeout=(8,30));response.raise_for_status();data=response.content
    assert len(data)<15_000_000 and data.startswith(b'%PDF');p.write_bytes(data);row['resolved_url']=response.url
   data=p.read_bytes();row.update(status='archived',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  except (requests.RequestException,AssertionError,OSError) as e:row.update(status='unavailable',error=repr(e)[:200])
  rows.append(row);print(json.dumps({'paper':name,'status':row['status']}),flush=True)
 (out/'manifest.json').write_text(json.dumps({'papers':rows,'latest_MTS3_revision':'Versionless publisher arXiv URL pinned by content SHA256; no guessed revision',
                                          'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2),encoding='utf-8')
if __name__=='__main__':main()
