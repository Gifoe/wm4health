"""Server-only compact publication; original source-model bytes are not mirrored."""
from common62 import *
import math,base64,os
c=read(OUT/'CLOSURE62_V1.json');assert c['status']=='CLOSED'
for name,h in c['reports'].items():assert sha(ROOT/name)==h
e=read(OUT/'EVALUATION62_V1.json');v=read(OUT/'INDEPENDENT_REPLAY62_V1.json');assert v['evaluation_sha256']==sha(OUT/'EVALUATION62_V1.json')
private_patterns=json.loads(os.environ['C62_PRIVATE_EXPORT_PATTERNS_JSON']);assert len(private_patterns)>=4
exclude={'export62_v1.py','export62_v2.py'}
selected=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.ps1']]+[p for p in (ROOT/'scripts').glob('*.py') if p.name not in exclude]
selected += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.log'] and not p.name.startswith('PUBLICATION_RECEIPT')]
dest=OUT/'publication_v2';dest.mkdir(exist_ok=False);rows=[]
for p in sorted(set(selected)):
 raw=p.read_bytes();original=sha(p);public=p.relative_to(ROOT).as_posix()
 try:text=raw.decode('utf-8-sig');transform='Remove UTF8 BOM' if raw.startswith(b'\xef\xbb\xbf') else None
 except UnicodeDecodeError:
  public+='.original_bytes.json';text=json.dumps({'original_path':p.relative_to(ROOT).as_posix(),'original_sha256':original,'encoding':'base64','content':base64.b64encode(raw).decode()},indent=2);transform='lossless base64 log'
 for needle in private_patterns:assert needle not in text,(public,'private verification-pattern material')
 b=text.encode('utf-8');target=dest/'files'/public;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append({'path':public,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'original_sha256':original,'git_blob_sha1':blob(b),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':transform})
dump(dest/'manifest.json',{'utc':utc(),'status':'C62_CLOSED_PARTIAL_PRO_DYN_REPRODUCTION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE62_V1.json'),'formal_fit_count':8,'planned_fit_count':9,'failed_formal_attempts':3,'accepted_final_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','raw_arrays_and_weights':'server_G_only','upstream_source_publication':'hashes_and_pinned_links_only_no_license_file_in_source','excluded_private_packaging_helpers':[{'path':'scripts/'+name,'sha256':sha(ROOT/'scripts'/name),'reason':'Contains explicit private verification patterns; superseded by environment-configured export62_v3.'} for name in sorted(exclude)],'no_local_result_files_written':True})
print(json.dumps({'status':'PUBLICATION_SNAPSHOT_READY','files':len(rows),'bytes':sum(r['bytes'] for r in rows),'manifest_sha256':sha(dest/'manifest.json')}))
