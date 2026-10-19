"""Hash completed research artifacts; this does not mark discovery achieved."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 entries=[]
 files=[p for p in ROOT.iterdir() if p.is_file() and p.name not in ['ARTIFACT_MANIFEST.json']]
 for folder in ['scripts','tests','outputs','visual-composer']:
  files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('.tmp','.download')))
 for p in sorted(set(files)):
  before=p.stat();h=hashlib.sha256()
  with p.open('rb') as f:
   while chunk:=f.read(1024*1024):h.update(chunk)
  after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p.relative_to(ROOT))
  entries.append({'path':p.relative_to(ROOT).as_posix(),'bytes':after.st_size,'sha256':h.hexdigest()})
 status=json.loads((ROOT/'outputs/task_status.json').read_text())
 result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Completed cycle2 artifacts, including server-retained data/model arrays. Overall research discovery goal is NOT complete. Runtime environments and the parent pinned source checkouts are dependencies, not duplicated here.',
  'objective_achieved':False,'model_training_cells_completed':status['native_initial_training_completed']+status['native_extended_training_completed']+status['generative_training_completed']+status['existing_RPM_exact_training_completed'],
  'entries':entries,'total_bytes':sum(r['bytes'] for r in entries)}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'files':len(entries),'bytes':result['total_bytes'],'completed_training_cells':result['model_training_cells_completed'],'objective_achieved':False}))
if __name__=='__main__':main()
