"""Verify and materialize only the newly archived source/primary-paper files."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 receipt=json.loads((ROOT/'outputs/source/metaid_fim_boundary_receipt.json').read_text())
 archive=ROOT/receipt['path'];assert sha(archive.read_bytes())==receipt['sha256']
 expected={x['path']:x['sha256'] for x in receipt['contents']};assert len(expected)==receipt['files']
 with zipfile.ZipFile(archive) as z:
  assert set(z.namelist())==set(expected)
  destinations={name:(ROOT/name).resolve() for name in expected}
  for name,p in destinations.items():
   assert p.is_relative_to(ROOT.resolve()) and not p.exists(),name
  for name,p in destinations.items():
   data=z.read(name);assert sha(data)==expected[name];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 assert all(sha(p.read_bytes())==expected[name] for name,p in destinations.items())
 record={'verified_files':len(expected),'archive_sha256':receipt['sha256'],'new_source_executed':False}
 (ROOT/'outputs/source/metaid_materialization.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))
if __name__=='__main__':main()
