from common64 import *
import math,os
c=read(OUT/'CLOSURE64_V1.json');assert c['status']=='CLOSED';source_check()
for n,h in c['reports'].items():assert sha(ROOT/n)==h
for n,h in c['artifacts'].items():assert sha(OUT/n)==h
patterns=json.loads(os.environ['C64_PRIVATE_EXPORT_PATTERNS_JSON'])
selected=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.ps1']]+list((ROOT/'scripts').glob('*.py'))
selected += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.log'] and not p.name.startswith('PUBLICATION_RECEIPT')]
dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
for p in sorted(set(selected)):
 raw=p.read_bytes();original=sha(p);public=p.relative_to(ROOT).as_posix();text=raw.decode('utf-8-sig')
 for needle in patterns:assert needle not in text,(public,'private export pattern')
 b=text.encode('utf-8');target=dest/'files'/public;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append({'path':public,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'original_sha256':original,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':'Remove UTF8 BOM' if raw.startswith(b'\xef\xbb\xbf') else None})
dump(dest/'manifest.json',{'utc':utc(),'status':'C64_CLOSED_GAUSSIAN_ATTRIBUTION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE64_V1.json'),'upstream_source_files_redistributed':0,'upstream_source_receipt_sha256':sha(OUT/'SOURCES64_V1.json'),'trained_neural_models':0,'accepted_final_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','raw_arrays_and_weights':'server_G_only','no_local_result_files_written':True})
print(json.dumps({'status':'PUBLICATION_READY','files':len(rows),'bytes':sum(r['bytes'] for r in rows),'manifest_sha256':sha(dest/'manifest.json')}))
