"""Private immutable-results inventory plus hash-verified compact archive."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 for folder in ['pcfm_native','ced_screen']:
  result=ROOT/f'outputs/{folder}/results.json';verification=load(result.with_name('verification.json'));assert verification['status']=='passed' and verification['results_sha256']==sha(result)
 replay=load(ROOT/'outputs/pcfm_native/neural_replay.json');assert replay['status']=='passed' and replay['calls_replayed']==28
 for job in ['pcfm_native_v1','ced_screen_v1']:
  status=load(ROOT/f'outputs/jobs/{job}/status.json');assert status['status']=='completed' and status['exit_code']==0
 for name in ['literature_review','baseline_registry','experimental_results','negative_evidence','hypothesis_space','research_ideas','final_research_direction']:assert (ROOT/f'{name}.md').is_file()
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['ARTIFACT_MANIFEST.json','ARCHIVE_RECEIPT.json','cycle28_compact.zip'] and not p.name.endswith(('.tmp','.download'))]
 excluded={'.npz','.npy','.pt','.pkl','.safetensors','.bin'}
 entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),included_in_compact=p.suffix not in excluded) for p in sorted(files)]
 manifest=dict(scope='Private cycle28 complete inventory;raw weights/numeric arrays server-only',server_root=str(ROOT),entries=entries,accepted_final_directions=0,goal_status='ACTIVE_UNACHIEVED',
  native_results_sha256=sha(ROOT/'outputs/pcfm_native/results.json'),ced_results_sha256=sha(ROOT/'outputs/ced_screen/results.json'),
  prior_dependency='Existing cycle16 ICL checkpoint/source;referenced exact hashes in CED results,not copied or modified')
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 path=ROOT/'cycle28_compact.zip'
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for entry in entries:
   if entry['included_in_compact']:z.write(ROOT/entry['path'],entry['path'])
  z.write(ROOT/'ARTIFACT_MANIFEST.json','ARTIFACT_MANIFEST.json')
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  for entry in entries:
   if entry['included_in_compact']:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
 receipt=dict(status='verified',archive=path.name,archive_bytes=path.stat().st_size,archive_sha256=sha(path),manifest_sha256=sha(ROOT/'ARTIFACT_MANIFEST.json'),
  total_files=len(entries),compact_files=sum(e['included_in_compact'] for e in entries)+1,server_only_files=sum(not e['included_in_compact'] for e in entries),server_only_bytes=sum(e['bytes'] for e in entries if not e['included_in_compact']))
 (ROOT/'ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
