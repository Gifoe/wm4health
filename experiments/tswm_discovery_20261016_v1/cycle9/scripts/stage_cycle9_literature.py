import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 archive=ROOT/'data/literature_bundle.zip'
 if '--package' in sys.argv:
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
   for path in sorted((ROOT/'outputs/literature').glob('*')):
    if path.is_file():z.write(path,path.relative_to(ROOT).as_posix())
  print('literature bundle',archive.stat().st_size);return
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts;path=ROOT.joinpath(*rel.parts);assert path.resolve().is_relative_to(ROOT.resolve());raw=z.read(item)
   if path.exists():assert path.read_bytes()==raw
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 registry=json.loads((ROOT/'outputs/literature/registry.json').read_text(encoding='utf-8'));assert not registry['errors']
 for row in registry['papers']:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
 print('verified primary PDFs',len(registry['papers']))
if __name__=='__main__':main()
