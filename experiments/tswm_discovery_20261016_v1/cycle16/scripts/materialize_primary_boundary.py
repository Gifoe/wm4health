"""Materialize the earlier verified boundary archive without executing source."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 receipt=json.loads((ROOT/'outputs/source/boundary_primary_archive_receipt.json').read_text());archive=ROOT/receipt['path']
 assert digest(archive.read_bytes())==receipt['sha256'];expected={r['path']:r['sha256'] for r in receipt['contents']}
 with zipfile.ZipFile(archive) as z:
  assert set(z.namelist())==set(expected)
  for name,want in expected.items():
   path=(ROOT/name).resolve();assert path.is_relative_to(ROOT.resolve());data=z.read(name);assert digest(data)==want
   if path.exists():assert digest(path.read_bytes())==want,name
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 record={'verified_files':len(expected),'archive_sha256':receipt['sha256'],'new_source_executed':False}
 out=ROOT/'outputs/source/boundary_primary_materialization.json';assert not out.exists();out.write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))
if __name__=='__main__':main()
