import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 archive=ROOT/'data/source_bundle.zip'
 if '--package' in sys.argv:
  with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in sorted((ROOT/'sources').rglob('*')):
    if p.is_file():z.write(p,p.relative_to(ROOT/'sources').as_posix())
  print('Packaged',archive.stat().st_size);return
 dest=ROOT/'sources';dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   path=dest.joinpath(*rel.parts);assert path.resolve().is_relative_to(dest.resolve());raw=z.read(item)
   if path.exists():assert path.read_bytes()==raw
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 pin=json.loads((dest/'actswm/SOURCE_PIN.json').read_text())
 for row in pin['files']:assert hashlib.sha256((dest/'actswm'/row['path']).read_bytes()).hexdigest()==row['sha256']
 print('Verified',len(pin['files']),'native source files')
if __name__=='__main__':main()

