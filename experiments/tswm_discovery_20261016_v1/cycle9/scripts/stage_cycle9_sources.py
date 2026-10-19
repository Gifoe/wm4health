import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 archive=ROOT/'data/source_bundle.zip'
 if '--package' in sys.argv:
  archive.parent.mkdir(exist_ok=True)
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
   for key in ['ep_idm','acro','ac_mtm']:
    pinpath=ROOT/f'sources/{key}/SOURCE_PIN.json';pin=json.loads(pinpath.read_text(encoding='utf-8'));z.write(pinpath,pinpath.relative_to(ROOT).as_posix())
    for row in pin['files']:
     path=ROOT/'sources'/key/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];z.write(path,path.relative_to(ROOT).as_posix())
    tree=ROOT/f'outputs/{key}_tree.json';z.write(tree,tree.relative_to(ROOT).as_posix())
  print('source bundle',archive.stat().st_size);return
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts;path=ROOT.joinpath(*rel.parts);assert path.resolve().is_relative_to(ROOT.resolve());raw=z.read(item)
   if path.exists():assert path.read_bytes()==raw
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 for key in ['ep_idm','acro','ac_mtm']:
  pin=json.loads((ROOT/f'sources/{key}/SOURCE_PIN.json').read_text(encoding='utf-8'))
  for row in pin['files']:assert hashlib.sha256((ROOT/'sources'/key/row['path']).read_bytes()).hexdigest()==row['sha256']
 print('all selected native files verified')
if __name__=='__main__':main()
