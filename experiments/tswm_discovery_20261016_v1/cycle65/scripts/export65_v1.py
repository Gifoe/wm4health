from common65 import *
import math,os
c=read(OUT/'CLOSURE65_V1.json');assert c['status']=='CLOSED';src=source_check()
for n,h in {**c['reports'],**c['artifacts']}.items():assert sha(ROOT/n if n in c['reports'] else OUT/n)==h
patterns=json.loads(os.environ['C65_PRIVATE_EXPORT_PATTERNS_JSON']);selected=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.ps1']]+list((ROOT/'scripts').glob('*.py'))+[pathlib.Path(x['server_path']) for x in src['files']]+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.log','.csv'] and not p.name.startswith('PUBLICATION_RECEIPT')]
dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
for p in sorted(set(selected)):
 raw=p.read_bytes();text=raw.decode('utf-8-sig');public=p.relative_to(ROOT).as_posix();transform='Remove UTF8 BOM' if raw.startswith(b'\xef\xbb\xbf') else None
 if p.name=='RUNTIME_ISOLATION_NOTE65_V1.md':text='> Superseded source-origin attribution: read RUNTIME_ISOLATION_NOTE65_V2.md. The original text is retained below as historical evidence.\n\n'+text;transform='Prepend superseded-provenance notice; original bytes retained on server'
 for needle in patterns:assert needle not in text,(public,'private export pattern')
 b=text.encode('utf-8');target=dest/'files'/public;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b);rows.append(dict(path=public,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),original_sha256=sha(p),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),parts=math.ceil(len(text)/3500),server_snapshot_file=str(target),publication_transform=transform))
dump(dest/'manifest.json',dict(utc=utc(),status='C65_CLOSED_NATIVE_LEARNED_GENERATOR_ATTRIBUTION',files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),total_content_parts=sum(r['parts'] for r in rows),closure_sha256=sha(OUT/'CLOSURE65_V1.json'),licensed_upstream_files=len(src['files']),trained_models=18,successful_optimizer_updates=144000,discarded_original_optimizer_updates_lower_bound=8000,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',raw_arrays_and_weights='server_G_only',no_local_result_files_written=True));print(json.dumps(dict(status='PUBLICATION_READY',files=len(rows),bytes=sum(r['bytes'] for r in rows),manifest_sha256=sha(dest/'manifest.json'))))
