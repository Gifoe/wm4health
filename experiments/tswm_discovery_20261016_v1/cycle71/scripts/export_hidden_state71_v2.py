"""Recover publication staging only; never repeat the closed ledger mutation."""
import sys,pathlib,json,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
closure=read(OUT/'HIDDEN_STATE_CLOSURE71_V1.json');q=read(OUT/'HIDDEN_STATE_QUALIFICATION71_V3.json')
assert closure['status']=='CLOSED_EXISTING_HIDDEN_STATE_CAPABILITY_QUALIFICATION_NOT_NEW_METHOD'
assert read(OUT/'hidden_state71_v3_exit.json')['exit_code']==0
assert sha(OUT/'HIDDEN_STATE_QUALIFICATION71_V3.json')==closure['result_sha256']
assert sha(q['array_path'])==q['array_sha256'] and sha(q['checkpoint']['path'])==q['checkpoint']['sha256']
for name,proof in closure['ledgers'].items():assert sha(ROOT/name)==proof['after_sha256'] and proof['prefix_preserved']
files=list(closure['ledgers'])+['HIDDEN_STATE_QUALIFICATION71_V1.md','HIDDEN_STATE_FINDING71_V1.md','scripts/export_hidden_state71_v2.py','outputs/HIDDEN_STATE_QUALIFICATION71_V3.json','outputs/HIDDEN_STATE_CLOSURE71_V1.json']
for v in (1,2,3):files+=['scripts/qualify_hidden_state71_v'+str(v)+'.py','scripts/capture_hidden_state71_v'+str(v)+'.py']+['outputs/hidden_state71_v'+str(v)+suffix for suffix in ['_launch.json','_exit.json','_stdout.log']]
patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON']);assert len(patterns)>=3
staged=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'Private export pattern')
 b=raw.decode('utf-8-sig').encode();staged.append((rel,b))
dest=OUT/'publication_hidden_state_v2';assert not dest.exists();dest.mkdir();rows=[]
for rel,b in staged:
 target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=closure['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),raw_arrays_weights_and_live_training_logs_included=False,eval_rows_accessed=0,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0,staging_recovery='V1 staging rejected its own hardcoded private-pattern list before publication. The failed local/server exporter and partial snapshot are preserved, excluded from public files; V2 takes patterns from private process environment and validates all files before staging. Scientific results and append-only ledgers were not repeated or changed.',failed_staging_source_sha256=sha(ROOT/'scripts'/'export_hidden_state71_v1.py'))
dump(dest/'manifest.json',manifest)
print(json.dumps(dict(status=closure['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'))),flush=True)
