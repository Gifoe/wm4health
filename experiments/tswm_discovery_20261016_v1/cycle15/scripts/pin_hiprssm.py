"""Bounded primary-source intake; never execute fetched source or launchers."""
import datetime,hashlib,json,pathlib,re,requests,zipfile,base64
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/hiprssm_source'
REPO='ALRhub/HiP-RSSM'
def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists()
 session=requests.Session();session.trust_env=False
 def fetch(url):
  response=session.get(url,timeout=(10,30));response.raise_for_status()
  assert len(response.content)<3_000_000
  return response
 base='https://api.github.com/repos/'+REPO
 metadata=fetch(base).json();branch=metadata['default_branch'];commit='f4adef0ce5ed4b37c9847973df320f72965bd330'
 tree=fetch(base+'/git/trees/'+commit+'?recursive=1').json();assert not tree.get('truncated')
 entries=[e for e in tree['tree'] if e['type']=='blob' and not any(p in ['logs','wandb','.idea','__pycache__'] for p in pathlib.PurePosixPath(e['path']).parts)
          and (pathlib.PurePosixPath(e['path']).suffix.lower() in ['.py','.md','.yaml','.yml','.txt'] or pathlib.PurePosixPath(e['path']).name.upper().startswith('LICENSE'))]
 selected={'metaWorldModels/ssm/kalmanOps/contextual_predict.py','metaWorldModels/ssm/contextualRSSM/contextualRKN.py',
           'metaWorldModels/ssm/contextualRSSM/acRKNContextualLayer.py','metaWorldModels/hiprssm.py','data/mobileDataSeq.py',
           'experiments/mobileRobot/conf/model/default.yaml','readme.md','LICENSE','utils/TimeDistributed.py'}
 entries=[e for e in entries if e['path'] in selected or e['path'].endswith('/__init__.py')]
 assert len(entries)<500
 records=[];excluded=[];total=0;agents=[]
 # Do not print or retain potential hardcoded credential values.
 secret=re.compile(rb'(?:wandb\.login\s*\([^)]*key\s*=\s*[\"\x27][A-Za-z0-9_-]{20,}|(?:API_KEY|api_key|access_token)\s*[=:]\s*[\"\x27][A-Za-z0-9_-]{24,})')
 source=ROOT/'upstream/HiP-RSSM';source.mkdir(parents=True,exist_ok=True)
 if (OUT/'source.zip').exists():
  failed=OUT/'source_failed_attempt.zip';assert not failed.exists()
  (OUT/'source.zip').replace(failed)
 with zipfile.ZipFile(OUT/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as bundle:
  for entry in entries:
   relative=entry['path'];assert '..' not in pathlib.PurePosixPath(relative).parts
   local=source/relative
   if local.is_file():
    data=local.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['sha']
   else:
    blob=fetch(base+'/git/blobs/'+entry['sha']).json();assert blob['encoding']=='base64'
    data=base64.b64decode(blob['content'])
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['sha']
   record={'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   if secret.search(data):record['reason']='Potential hardcoded credential; not retained or executed';excluded.append(record);continue
   total+=len(data);assert total<12_000_000
   bundle.writestr(relative,data);p=source/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);records.append(record)
   if p.name=='AGENTS.md':agents.append(relative)
 payload={'repository':metadata['html_url'],'revision':commit,'default_branch':branch,'license_spdx':(metadata.get('license') or {}).get('spdx_id'),
          'files':records,'excluded_files':excluded,'AGENTS_files':agents,'bytes':total,'count':len(records),
          'source_zip_sha256':hashlib.sha256((OUT/'source.zip').read_bytes()).hexdigest(),'downloaded_source_executed':False,
          'snapshot_scope':'Selected original source for context-dependence audit; not a complete original training checkout',
          'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'manifest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 print(json.dumps({'revision':commit,'files':len(records),'bytes':total,'excluded_file_paths':[x['path'] for x in excluded],'AGENTS_files':agents}))
if __name__=='__main__':main()
