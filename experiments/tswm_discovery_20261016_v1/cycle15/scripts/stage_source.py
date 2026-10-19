"""Materialize only the pinned audited source archive into a new cycle folder."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 m=json.loads((ROOT/'outputs/source/manifest.json').read_text())
 assert m['revision']=='ab971c0ec43df7664c4b727ca69d5b8dd0cab5da' and not m['excluded_files'] and not m['AGENTS_files']
 archive=ROOT/'outputs/source/source.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['source_zip_sha256']
 target=ROOT/'upstream/MTS3';assert not target.exists()
 with zipfile.ZipFile(archive) as bundle:
  assert len(bundle.namelist())==m['count']
  for row in m['files']:
   rel=pathlib.PurePosixPath(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
   p=target.joinpath(*rel.parts);assert p.resolve().is_relative_to(target.resolve())
   data=bundle.read(row['path']);assert hashlib.sha256(data).hexdigest()==row['sha256']
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 print(json.dumps({'source_files_verified':m['count'],'source_executed':False}))
if __name__=='__main__':main()
