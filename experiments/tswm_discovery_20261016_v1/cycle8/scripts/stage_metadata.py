import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 archive=ROOT/'data/metadata_bundle.zip'
 if '--package' in sys.argv:
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
   for path in sorted((ROOT/'sources').rglob('*')):
    if path.is_file():z.write(path,path.relative_to(ROOT).as_posix())
   z.write(ROOT/'outputs/chamber_tree.json','outputs/chamber_tree.json')
  print('Packaged metadata',archive.stat().st_size);return
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   path=ROOT.joinpath(*rel.parts);assert path.resolve().is_relative_to(ROOT.resolve());raw=z.read(item)
   if path.exists():assert path.read_bytes()==raw
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 pin=json.loads((ROOT/'sources/chamber/SOURCE_PIN.json').read_text())
 for row in pin['files']:assert hashlib.sha256((ROOT/'sources/chamber'/row['path']).read_bytes()).hexdigest()==row['sha256']
 print('Verified',len(pin['files']),'pinned source/metadata files')
if __name__=='__main__':main()
