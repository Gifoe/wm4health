"""TLS-verified GitHub API snapshots of explicitly selected public source files."""
import base64,datetime,hashlib,json,pathlib,time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPOS={'ep_idm':'LeoToso/JEPA_control','acro':'manantomar/agent-centric-representations','ac_mtm':'jackboyla/action-contrastive-jepa'}
def get(url):
 for attempt in range(3):
  session=requests.Session();session.trust_env=False
  try:
   r=session.get(url,timeout=(5,15));r.raise_for_status();return r.json()
  except requests.RequestException:
   if attempt==2:raise
   time.sleep(.5)
def main():
 for key,repo in REPOS.items():
  out=ROOT/'outputs';out.mkdir(exist_ok=True);treepath=out/f'{key}_tree.json'
  if treepath.exists():tree=json.loads(treepath.read_text(encoding='utf-8'))
  else:
   metadata=get('https://api.github.com/repos/'+repo)
   revision=get('https://api.github.com/repos/'+repo+'/commits/'+metadata['default_branch'])['sha']
   tree=get('https://api.github.com/repos/'+repo+'/git/trees/'+revision+'?recursive=1')
   treepath.write_text(json.dumps(tree,indent=2),encoding='utf-8')
  assert not tree.get('truncated',False)
  paths=[x['path'] for x in tree['tree'] if x['type']=='blob']
  print(json.dumps({'repo':repo,'revision':tree['sha'],'tree_files':len(paths)}),flush=True)
  if key=='ep_idm':wanted=['README.md','models/jepa_world_model.py','models/sensorimotor_world_model.py','training/trainer.py','losses/prediction.py','configs/cartpole_jepa_rollout_endpoint_inverse_act1.yaml']
  elif key=='acro':wanted=['README.md','acro.py']
  else:wanted=['README.md','jepa.py','module.py','train.py','tests/test_masked_transition.py','config/train/lewm_masked_action_nce.yaml','config/train/lewm_masked_ms_inverse.yaml','config/train/lewm_masked_endpoint_action_nce.yaml']
  wanted=list(dict.fromkeys(wanted));assert len(wanted)<25,(key,wanted)
  folder=ROOT/'sources'/key;rows=[]
  for name in wanted:
   row=next(x for x in tree['tree'] if x['path']==name);path=folder/name
   if path.exists():raw=path.read_bytes()
   else:
    blob=get(row['url']);assert blob['encoding']=='base64';raw=base64.b64decode(blob['content'])
   assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
   rows.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
   print(key,name,len(raw),flush=True)
  pin={'repository':repo,'revision':tree['sha'],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Selected source files only. Complete tree inventoried; no launcher executed; no weights or dataset downloaded.','files':rows}
  folder.mkdir(parents=True,exist_ok=True);(folder/'SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
if __name__=='__main__':main()
