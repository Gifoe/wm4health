"""Hash-preserving private compact archive. Raw numeric arrays remain on the server."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 result=ROOT/'outputs/controlled_em_study/results.json';verify=json.loads((ROOT/'outputs/controlled_em_study/verification.json').read_text())
 assert verify['status']=='passed' and verify['results_sha256']==sha(result)
 for name in ['controlled_em_amended_v1','controlled_em_verify_v1']:
  status=json.loads((ROOT/f'outputs/jobs/{name}/status.json').read_text());assert status['status']=='completed' and status['exit_code']==0
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['ARTIFACT_MANIFEST.json','cycle27_compact.zip','ARCHIVE_RECEIPT.json'] and not p.name.endswith(('.tmp','.download'))]
 entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),included_in_compact=p.suffix not in ['.npz','.npy','.pt','.pkl','.bin']) for p in sorted(files)]
 manifest=dict(scope='Private complete cycle27 file hashes;raw numeric arrays remain on server',raw_location=str(ROOT),entries=entries,
  accepted_final_directions=0,goal_status='ACTIVE_UNACHIEVED',results_sha256=sha(result))
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 archive=ROOT/'cycle27_compact.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for e in entries:
   if e['included_in_compact']:z.write(ROOT/e['path'],e['path'])
  z.write(ROOT/'ARTIFACT_MANIFEST.json','ARTIFACT_MANIFEST.json')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for e in entries:
   if e['included_in_compact']:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
 receipt=dict(status='verified',archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
  manifest_sha256=sha(ROOT/'ARTIFACT_MANIFEST.json'),n_files=len(entries),compact_files=sum(e['included_in_compact'] for e in entries)+1,
  raw_numeric_files=sum(not e['included_in_compact'] for e in entries),raw_numeric_bytes=sum(e['bytes'] for e in entries if not e['included_in_compact']))
 (ROOT/'ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
