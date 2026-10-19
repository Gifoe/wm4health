"""Safely unpack and verify the pinned small source bundle; never execute it."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 dst=ROOT/'sources';dst.mkdir(exist_ok=True,parents=True)
 archives=sorted((ROOT/'data').glob('*source_bundle.zip'))
 for archive in archives:
  with zipfile.ZipFile(archive) as z:
   for item in z.infolist():
    if item.is_dir():continue
    relative=pathlib.PurePosixPath(item.filename);assert '..' not in relative.parts and not relative.is_absolute()
    path=dst.joinpath(*relative.parts);assert path.resolve().is_relative_to(dst.resolve())
    content=z.read(item)
    if path.exists():assert path.read_bytes()==content,'Refuse source modification '+str(path)
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
  print('Source bundle verified',archive.name,hashlib.sha256(archive.read_bytes()).hexdigest())
 registry=[]
 for source in ['epi-diff-abm','epi-cf-benchmark','lenepa']:
  if not (dst/source/'SOURCE_PIN.json').exists():continue
  pin=json.loads((dst/source/'SOURCE_PIN.json').read_text())
  for item in pin.get('files',[]):
   assert hashlib.sha256((dst/source/item['path']).read_bytes()).hexdigest()==item['sha256']
  registry.append(pin)
  print(source,pin['revision'])
 (ROOT/'outputs'/'source_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
if __name__=='__main__':main()
