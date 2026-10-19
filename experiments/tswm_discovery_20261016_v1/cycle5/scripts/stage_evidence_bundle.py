"""Securely mirror already archived public evidence; reject changed existing files."""
import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 archive=ROOT/'data'/'followup_evidence.zip'
 if '--package' in sys.argv:
  files=[p for name in ['metailad','timebench'] for p in (ROOT/'sources'/name).rglob('*') if p.is_file()]+[p for p in (ROOT/'outputs'/'literature').rglob('*') if p.is_file()]+[ROOT/'outputs'/'followup_source_registry.json']
  with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
  print('Packaged evidence',len(files),'files',archive.stat().st_size,'bytes');return
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   path=ROOT.joinpath(*rel.parts);assert path.resolve().is_relative_to(ROOT.resolve());raw=z.read(item)
   if path.exists():assert path.read_bytes()==raw,rel
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 for name in ['metailad','timebench']:
  folder=ROOT/'sources'/name;pin=json.loads((folder/'SOURCE_PIN.json').read_text())
  for row in pin['files']:assert hashlib.sha256((folder/row['path']).read_bytes()).hexdigest()==row['sha256']
 literature=json.loads((ROOT/'outputs'/'literature'/'registry.json').read_text())
 for row in literature['papers']:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
 print('Verified source pins and',len(literature['papers']),'primary papers')
if __name__=='__main__':main()
