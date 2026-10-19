"""Read-only, commit-pinned official baseline source snapshot; never execute it."""
import datetime,hashlib,json,pathlib,requests,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/bayesian_context_source'
def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists()
 session=requests.Session();session.trust_env=False
 base='https://api.github.com/repos/boschresearch/bayesian-context-aggregation'
 def fetch(url):
  response=session.get(url,timeout=(8,30));response.raise_for_status()
  assert len(response.content)<2_000_000
  return response
 repo=fetch(base).json();branch=repo['default_branch']
 commit=fetch(base+'/commits/'+branch).json()['sha'];assert len(commit)==40
 tree=fetch(base+'/git/trees/'+commit+'?recursive=1').json();assert not tree.get('truncated')
 entries=[x for x in tree['tree'] if x['type']=='blob' and (pathlib.PurePosixPath(x['path']).suffix in ['.py','.md','.txt','.yaml','.yml'] or pathlib.PurePosixPath(x['path']).name.upper().startswith('LICENSE'))]
 assert len(entries)<100
 records=[];size=0
 with zipfile.ZipFile(OUT/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as bundle:
  for entry in entries:
   relative=entry['path'];assert '..' not in pathlib.PurePosixPath(relative).parts
   data=fetch('https://raw.githubusercontent.com/boschresearch/bayesian-context-aggregation/'+commit+'/'+relative).content
   size+=len(data);assert size<3_000_000
   bundle.writestr(relative,data)
   records.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
   if relative.lower() in ['readme.md','license','license.txt']:(OUT/pathlib.PurePosixPath(relative).name).write_bytes(data)
 payload={'repository':repo['html_url'],'default_branch':branch,'revision':commit,'archived':repo['archived'],
          'license_spdx':(repo.get('license') or {}).get('spdx_id'),'files':records,'count':len(records),'bytes':size,
          'source_zip_sha256':hashlib.sha256((OUT/'source.zip').read_bytes()).hexdigest(),
          'downloaded_source_executed':False,'scope':'Official source audit only; no installation or GP/dynamics reproduction',
          'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'manifest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 print(json.dumps({'revision':commit,'files':len(records),'bytes':size,'source_executed':False}))
if __name__=='__main__':main()
