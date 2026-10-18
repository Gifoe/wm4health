"""Server-only immutable compact evidence and exact licensed-source snapshot."""
from common61 import *
import base64,math

def main():
 closure=read(OUT/'CLOSURE61_V1.json');source=sources();e=read(OUT/'EVALUATION61_V1.json');seal=read(OUT/'MODEL_SEAL61_V1.json')
 assert closure['status']=='C61_LEARNED_EXISTING_DURATION_CONTROL_SCREEN_CLOSED'
 for name,h in closure['reports_sha256'].items():assert sha(ROOT/name)==h
 assert closure['evaluation_sha256']==sha(OUT/'EVALUATION61_V1.json')
 assert len(seal['models'])+len(seal['failed'])==12
 for row in e['records']+e['oracle']:assert sha(row['arrays'])==row['arrays_sha256']
 selected=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.ps1']]+list((ROOT/'scripts').glob('*.py'))+[pathlib.Path(r['server_path']) for r in source['files']]
 for p in OUT.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl','.log','.csv'] and not any(x.startswith('publication_') for x in p.relative_to(OUT).parts):selected.append(p)
 dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
 for p in sorted(set(selected)):
  public=p.relative_to(ROOT).as_posix();raw=p.read_bytes();original=hashlib.sha256(raw).hexdigest()
  try:
   text=raw.decode('utf-8-sig');transform='Remove UTF8 BOM' if raw.startswith(b'\xef\xbb\xbf') else None
  except UnicodeDecodeError:
   public+='.original_bytes.json';text=json.dumps({'original_path':p.relative_to(ROOT).as_posix(),'original_bytes':len(raw),'original_sha256':original,'encoding':'base64','content':base64.b64encode(raw).decode(),'reason':'Lossless preservation without guessing a terminal encoding'},indent=2);transform='Lossless base64 log envelope'
  data=text.encode('utf-8');target=dest/'files'/public;target.parent.mkdir(exist_ok=True,parents=True)
  with target.open('xb') as f:f.write(data)
  rows.append({'path':public,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_sha256':original,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':transform})
 result={'utc':utc(),'status':'C61_CLOSED_EXISTING_DURATION_CONTROL_PUBLICATION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE61_V1.json'),'decision':closure['decision'],'source_objects_verified':len(source['files']),'predeclared_attempts':12,'real_domains':0,'scope':'Native existing HSMM/HMM, numerical qualification, supported adapter correction, fixed unlabeled learning and complete heldout density/survival replay. This is a controlled prior-art exclusion, not a new world-model framework or real-data result. Raw trajectories/audit labels/checkpoints/prediction arrays stay on G storage with hashes.','no_local_result_files_written':True,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
 dump(dest/'manifest.json',result);print(json.dumps({'status':result['status'],'files':len(rows),'bytes':result['bytes'],'parts':result['total_content_parts'],'manifest_sha256':sha(dest/'manifest.json')}),flush=True)

if __name__=='__main__':main()
