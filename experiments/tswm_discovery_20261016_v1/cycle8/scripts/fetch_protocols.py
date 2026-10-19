"""Archive pinned generator text for audit; never execute these protocols."""
import hashlib,json,pathlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 pin=json.loads((ROOT/'sources/chamber/SOURCE_PIN.json').read_text());tree=json.loads((ROOT/'outputs/chamber_tree.json').read_text())['tree'];s=requests.Session();s.trust_env=False
 for row in tree:
  p=row['path'];parts=p.split('/')
  if not (len(parts)>=3 and parts[0]=='datasets' and parts[1] in ['wt_walks_v1','wt_test_v1'] and p.endswith(('.py','.csv')) and row.get('size',0)<100000):continue
  path=ROOT/'sources/chamber'/p
  if path.exists():raw=path.read_bytes()
  else:
   response=s.get('https://raw.githubusercontent.com/'+pin['repository']+'/'+pin['revision']+'/'+p,timeout=(5,12));response.raise_for_status();raw=response.content
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
  if not any(r['path']==p for r in pin['files']):pin['files'].append({'path':p,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  print('Pinned protocol text',p,flush=True)
 (ROOT/'sources/chamber/SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
if __name__=='__main__':main()
