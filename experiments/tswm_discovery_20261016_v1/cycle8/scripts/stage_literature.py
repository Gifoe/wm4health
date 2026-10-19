"""Package locally archived primary PDFs and verify their private server mirror."""
import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 archive=ROOT/'data/literature_bundle.zip'
 if '--package' in sys.argv:
  with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in sorted((ROOT/'outputs/literature').rglob('*')):
    if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
  print('Packaged',archive.stat().st_size,'bytes');return
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   path=ROOT.joinpath(*rel.parts);assert path.resolve().is_relative_to(ROOT.resolve());raw=z.read(item)
   if path.exists():assert path.read_bytes()==raw,rel
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 papers=json.loads((ROOT/'outputs/literature/registry.json').read_text())
 for row in papers['papers']:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
 print('Verified',len(papers['papers']),'primary PDFs')
if __name__=='__main__':main()
