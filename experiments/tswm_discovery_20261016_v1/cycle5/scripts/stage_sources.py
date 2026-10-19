import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'data'/'sources.zip'
 if '--package' in sys.argv:
  with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in sorted((ROOT/'sources').rglob('*')):
    if p.is_file():z.write(p,p.relative_to(ROOT/'sources').as_posix())
  print('Packaged source bytes',path.stat().st_size);return
 dest=ROOT/'sources';dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   p=dest.joinpath(*rel.parts);assert p.resolve().is_relative_to(dest.resolve());data=z.read(item)
   if p.exists():assert p.read_bytes()==data
   else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 pins=[]
 for folder in dest.iterdir():
  pin=json.loads((folder/'SOURCE_PIN.json').read_text())
  for row in pin['files']:assert hashlib.sha256((folder/row['path']).read_bytes()).hexdigest()==row['sha256']
  pins.append(pin)
 (ROOT/'outputs').mkdir(parents=True,exist_ok=True)
 (ROOT/'outputs'/'source_registry.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
 print('Verified pinned source files',sum(len(p['files']) for p in pins))
if __name__=='__main__':main()
