"""Public, read-only prerecorded-data metadata. Never operates remote hardware."""
import base64,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='juangamella/causal-chamber'
def main():
 session=requests.Session();session.trust_env=False
 def get(url):
  response=session.get(url,timeout=(10,30));response.raise_for_status();return response.json()
 commit=get('https://api.github.com/repos/'+REPO+'/commits/main')['sha']
 tree=get(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1');assert not tree['truncated']
 out=ROOT/'outputs';out.mkdir(parents=True,exist_ok=True)
 (out/'chamber_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
 rows=[]
 for row in tree['tree']:
  name=row['path'];parts=name.split('/');lower=name.lower()
  take=name in ['README.md','LICENSE'] or (len(parts)>=3 and parts[0]=='datasets' and parts[1] in ['wt_test_v1','wt_walks_v1','wt_intake_impulse_v1','wt_changepoints_v1','wt_pressure_control_v1','wt_validate_v1'] and name.endswith(('.md','.yaml','.yml','.json','.txt')))
  if not take or row['type']!='blob' or row.get('size',0)>300000:continue
  item=get(f'https://api.github.com/repos/{REPO}/git/blobs/'+row['sha']);assert item['encoding']=='base64';raw=base64.b64decode(item['content'])
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
  path=ROOT/'sources/chamber'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
  rows.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});print('Metadata',name,flush=True)
 pin={'repository':REPO,'revision':commit,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'README/dataset metadata only, public prerecorded datasets; no RemoteLab access or hardware command','files':rows}
 (ROOT/'sources/chamber/SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
 print(json.dumps({'commit':commit,'files':len(rows),'tree_entries':len(tree['tree'])}))
if __name__=='__main__':main()
