"""Extract an inspected source bundle and verify every pinned text file."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 destination=ROOT/'sources';destination.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(ROOT/'data'/'source_bundle.zip') as archive:
  for item in archive.infolist():
   if item.is_dir():continue
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   target=destination.joinpath(*rel.parts);assert target.resolve().is_relative_to(destination.resolve())
   content=archive.read(item)
   if target.exists():assert target.read_bytes()==content
   else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
 registry=[]
 for folder in destination.iterdir():
  if not (folder/'SOURCE_PIN.json').exists():continue
  pin=json.loads((folder/'SOURCE_PIN.json').read_text())
  for row in pin['files']:assert hashlib.sha256((folder/row['path']).read_bytes()).hexdigest()==row['sha256']
  registry.append(pin)
 out=ROOT/'outputs';out.mkdir(parents=True,exist_ok=True);(out/'source_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
 print([(p['repository'],p['revision']) for p in registry])
if __name__=='__main__':main()
