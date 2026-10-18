"""Correct one report table boundary in a new snapshot, preserve original bytes."""
import sys,pathlib,hashlib,json,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
previous=OUT/'publication_v1';m=read(previous/'manifest.json');dest=OUT/'publication_v2';assert not dest.exists();staged=[];changes=[]
for r in m['files']:
 raw=pathlib.Path(r['server_snapshot_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
 if r['path']=='experimental_results.md':
  old=b'free-running steps. | Control';new=b'free-running steps.\n\n| Control';assert raw.count(old)==1;updated=raw.replace(old,new);changes.append(dict(path=r['path'],original_sha256=r['sha256'],reviewed_sha256=hashlib.sha256(updated).hexdigest(),change='Insert blank line before Markdown table; no numeric or scientific change'));raw=updated
 staged.append((r['path'],raw))
rel='scripts/review_publication74_v1.py';staged.append((rel,(ROOT/rel).read_bytes()));patterns=json.loads(os.environ['C74_PRIVATE_EXPORT_PATTERNS_JSON'])
for rel,b in staged:
 for needle in patterns:assert needle.encode() not in b,(rel,'Private export pattern')
dest.mkdir();rows=[]
for rel,b in staged:
 target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b);rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
out=dict(utc=utc(),status=m['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),raw_data_weights_or_predictions_included=False,official_test_numeric_values_parsed=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0,review_changes=changes,original_snapshot_manifest_sha256=sha(previous/'manifest.json'),original_snapshot_preserved=True)
dump(dest/'manifest.json',out);print(json.dumps(dict(status=out['status'],files=len(rows),bytes=out['bytes'],manifest_sha256=sha(dest/'manifest.json'),reviewed_tables=1)),flush=True)
