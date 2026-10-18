"""Preserve the initial snapshot and add canonical audit disclosures in a new snapshot."""
import sys,pathlib,hashlib,shutil,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
def main():
 old=OUT/'publication_v1';dest=OUT/'publication_v2';oldmanifest=read(old/'manifest.json');assert not dest.exists();dest.mkdir();pub=[]
 for row in oldmanifest['files']:
  raw=pathlib.Path(row['server_snapshot_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];p=dest/'files'/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);pub.append(dict(row,server_snapshot_file=str(p)))
 original=(ROOT/'FEATURE_ADAPTATION_SCREEN81_V1.md').read_text(encoding='utf-8');amendment=(ROOT/'NUMERICAL_VERIFICATION81_V1.md').read_text(encoding='utf-8');report=original+'\n\n---\n\n'+amendment
 with (ROOT/'FEATURE_ADAPTATION_SCREEN81_V2.md').open('x',encoding='utf-8') as f:f.write(report)
 additions=['README.md','README81_V2.md','NUMERICAL_VERIFICATION81_V1.md','FEATURE_ADAPTATION_SCREEN81_V2.md','outputs/SOLVER_DIAGNOSTIC81_V1.json','scripts/finalize_snapshot81_v1.py']
 for rel in additions:
  raw=(ROOT/rel).read_text(encoding='utf-8-sig').encode();assert all(token.encode() not in raw for token in json.loads(os.environ.get('C81_PRIVATE_EXPORT_PATTERNS_JSON','[]'))),(rel,'private export pattern');p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);pub.append(dict(path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),server_snapshot_file=str(p),encoding='utf-8'))
 assert len({r['path'] for r in pub})==len(pub)
 dump(dest/'manifest.json',dict(utc=utc(),files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),prior_snapshot_manifest_sha256=sha(old/'manifest.json'),canonical_checker='scripts/verify81_v3.py',canonical_report='FEATURE_ADAPTATION_SCREEN81_V2.md',numerical_experiment_unchanged=True,large_data_weights_prediction_arrays_included=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0));print(json.dumps(dict(status='FINAL_SNAPSHOT_READY',files=len(pub),manifest_sha256=sha(dest/'manifest.json'))),flush=True)
if __name__=='__main__':main()
