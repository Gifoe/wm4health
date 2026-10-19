"""Archive public firmware text as timing evidence; never compile or flash."""
import hashlib,json,pathlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 pin=json.loads((ROOT/'sources/chamber/SOURCE_PIN.json').read_text());name='hardware/arduino/wt_mk_1/wt_mk_1.ino';tree=json.loads((ROOT/'outputs/chamber_tree.json').read_text())['tree'];row=next(r for r in tree if r['path']==name);s=requests.Session();s.trust_env=False
 r=s.get('https://raw.githubusercontent.com/'+pin['repository']+'/'+pin['revision']+'/'+name,timeout=(5,15));r.raise_for_status();raw=r.content;assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
 path=ROOT/'sources/chamber'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 if not any(p['path']==name for p in pin['files']):pin['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 (ROOT/'sources/chamber/SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8');print('Pinned firmware text',len(raw))
if __name__=='__main__':main()
