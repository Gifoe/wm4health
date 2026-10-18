from common63 import *
import math,base64,os
c=read(OUT/'CLOSURE63_V1.json');assert c['status']=='CLOSED';src=sources()
for name,h in c['reports'].items():assert sha(ROOT/name)==h
assert c['audit_sha256']==sha(OUT/'AUDIT63_V1.json') and c['replay_sha256']==sha(OUT/'INDEPENDENT_REPLAY63_V1.json')
patterns=json.loads(os.environ['C63_PRIVATE_EXPORT_PATTERNS_JSON'])
selected=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.ps1']]+list((ROOT/'scripts').glob('*.py'))+[pathlib.Path(r['server_path']) for r in src['files']]
selected += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.log'] and not p.name.startswith('PUBLICATION_RECEIPT')]
dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
for p in sorted(set(selected)):
 raw=p.read_bytes();original=sha(p);public=p.relative_to(ROOT).as_posix()
 try:text=raw.decode('utf-8-sig');transform='Remove UTF8 BOM' if raw.startswith(b'\xef\xbb\xbf') else None
 except UnicodeDecodeError:
  public+='.original_bytes.json';text=json.dumps({'original_path':p.relative_to(ROOT).as_posix(),'original_sha256':original,'encoding':'base64','content':base64.b64encode(raw).decode()},indent=2);transform='lossless base64 envelope'
 for needle in patterns:assert needle not in text,(public,'private export pattern')
 b=text.encode('utf-8');target=dest/'files'/public;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append({'path':public,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'original_sha256':original,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':transform})
dump(dest/'manifest.json',{'utc':utc(),'status':'C63_CLOSED_NATIVE_COMPONENT_CONTRACT_AUDIT','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE63_V1.json'),'licensed_source_files':len(src['files']),'trained_models':0,'optimizer_updates':0,'heldout_scores':0,'weight_fixtures':3,'accepted_final_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','raw_arrays_and_weights':'server_G_only','no_local_result_files_written':True})
print(json.dumps({'status':'PUBLICATION_READY','files':len(rows),'bytes':sum(r['bytes'] for r in rows),'manifest_sha256':sha(dest/'manifest.json')}))
