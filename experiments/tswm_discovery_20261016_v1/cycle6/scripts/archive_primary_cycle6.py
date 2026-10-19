"""Archive pinned primary papers; only public small documents, never credentials."""
import concurrent.futures,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCES=[
 ('child_2510.18310v1','https://arxiv.org/pdf/2510.18310v1'),
 ('ada_diffuser_2605.16054v1','https://arxiv.org/pdf/2605.16054v1'),
 ('caring_2401.14535v1','https://arxiv.org/pdf/2401.14535v1'),
 ('survae_2007.02731v1','https://arxiv.org/pdf/2007.02731v1'),
 ('exploding_inverses_aistats2021','https://proceedings.mlr.press/v130/behrmann21a/behrmann21a.pdf'),
 ('self_consistency_2501.13483v6','https://arxiv.org/pdf/2501.13483v6'),
 ('branch_jepa_2607.05238v3','https://arxiv.org/pdf/2607.05238v3'),
 ('uwm_jepa_2605.25313v1','https://arxiv.org/pdf/2605.25313v1'),
 ('comet_2404.15109v1','https://arxiv.org/pdf/2404.15109v1'),
 ('mechanistic_world_models_2607.12474v1','https://arxiv.org/pdf/2607.12474v1'),
 ('world_models_pieces_2606.24842v1','https://arxiv.org/pdf/2606.24842v1'),
 ('thermoforce_2607.03942v1','https://arxiv.org/pdf/2607.03942v1'),
 ('biosignal_position_2610.17510v1','https://arxiv.org/pdf/2610.17510v1'),
]
def fetch(row):
 name,url=row;folder=ROOT/'outputs/literature';folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.pdf')
 if not path.exists():
  session=requests.Session();session.trust_env=False
  with session.get(url,timeout=(12,45),stream=True) as response:
   response.raise_for_status();parts=[];size=0
   for chunk in response.iter_content(131072):
    size+=len(chunk)
    if size>12_000_000:raise ValueError('Small-paper size cap exceeded')
    parts.append(chunk)
  raw=b''.join(parts);assert raw.startswith(b'%PDF-'),url;path.write_bytes(raw)
 raw=path.read_bytes()
 return {'name':name,'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def main():
 rows=[];errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(fetch,row):row for row in SOURCES}
  for future,row in futures.items():
   try:rows.append(future.result());print('Archived',row[0],flush=True)
   except Exception as error:errors.append({'name':row[0],'url':row[1],'error':repr(error)});print('Unavailable',row[0],repr(error),flush=True)
 result={'papers':rows,'errors':errors};(ROOT/'outputs/literature/registry.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'archived':len(rows),'unavailable':len(errors)}))
if __name__=='__main__':main()
