"""Read-only source-stage integrity and runtime inventory, not a science run."""
import hashlib,importlib.metadata,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 pin=json.loads((ROOT/'sources/lens/SOURCE_PIN.json').read_text())
 for row in pin['files']:
  p=ROOT/'sources/lens'/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 packages={}
 for package in ['torch','jax','jaxlib','flax','brax','optax','numpy']:
  try:packages[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:packages[package]=None
 report={'revision':pin['revision'],'selected_source_files_verified':len(pin['files']),'AGENTS_files':pin['AGENTS_files'],'packages':packages,'python':sys.version,'native_training_executed':False,'scientific_evaluation_executed':False,'full_original_protocol_reproduced':False,'objective_achieved':False}
 (ROOT/'outputs/source_stage_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
