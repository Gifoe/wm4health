"""Bundle local primary PDFs, or stage and verify their server mirror."""
import hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 bundle=ROOT/'data'/'primary_papers.zip'
 if '--package' in sys.argv:
  bundle.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED) as z:
   for p in sorted((ROOT/'outputs'/'literature').glob('*')):
    if p.is_file():z.write(p,p.name)
  print('Packaged',bundle.stat().st_size);return
 target=ROOT/'outputs'/'literature';target.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(bundle) as z:
  for item in z.infolist():
   rel=pathlib.PurePosixPath(item.filename);assert len(rel.parts)==1 and not rel.is_absolute()
   path=target/item.filename;data=z.read(item)
   if path.exists():assert path.read_bytes()==data
   else:path.write_bytes(data)
 rows=json.loads((target/'registry.json').read_text())
 for row in rows:assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256']
 print(json.dumps({'primary_pdfs':len(rows),'all_server_sha256_verified':True}))
if __name__=='__main__':main()
