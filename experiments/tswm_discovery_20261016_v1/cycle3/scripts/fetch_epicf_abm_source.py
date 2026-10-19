"""Fetch only small code/config files at the pinned simulator commit."""
import datetime,hashlib,json,pathlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='complex-ai-lab/epi-diff-abm'
REV='824ca2a9785038eaec4e277903856d796ac4adb3'
def main():
 s=requests.Session();s.trust_env=False
 tree=json.loads((ROOT/'outputs'/'epi-diff-abm_tree.json').read_text())
 rows=[]
 for item in tree['tree']:
  if item['type']!='blob' or pathlib.Path(item['path']).suffix.lower() not in {'.py','.md','.yaml','.yml','.json','.txt','.sh'}:continue
  assert item.get('size',0)<1000000
  url=f'https://raw.githubusercontent.com/{REPO}/{REV}/{item["path"]}'
  response=s.get(url,timeout=45);response.raise_for_status()
  path=ROOT/'sources'/'epi-diff-abm'/item['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
  rows.append({'path':item['path'],'sha256':hashlib.sha256(response.content).hexdigest(),'bytes':len(response.content),'url':url})
 pin={'repository':REPO,'revision':REV,'scope':'only code/config/docs; large CSV data and bytecode omitted','fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows}
 (ROOT/'sources'/'epi-diff-abm'/'SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
 registry=[json.loads((ROOT/'sources'/'epi-cf-benchmark'/'SOURCE_PIN.json').read_text()),pin]
 (ROOT/'outputs'/'source_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
 print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'revision':REV}))
if __name__=='__main__':main()
