import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 dest=ROOT/'sources'
 with zipfile.ZipFile(ROOT/'data'/'vjepa_source_bundle.zip') as archive:
  for item in archive.infolist():
   if item.is_dir():continue
   rel=pathlib.PurePosixPath(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   target=dest.joinpath(*rel.parts);assert target.resolve().is_relative_to(dest.resolve());data=archive.read(item)
   if target.exists():assert target.read_bytes()==data
   else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 rows=[]
 for folder in sorted(dest.iterdir()):
  if not (folder/'SOURCE_PIN.json').exists():continue
  pin=json.loads((folder/'SOURCE_PIN.json').read_text())
  for row in pin['files']:assert hashlib.sha256((folder/row['path']).read_bytes()).hexdigest()==row['sha256']
  rows.append(pin)
 (ROOT/'outputs'/'source_registry.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 print([(r['repository'],r['revision']) for r in rows])
if __name__=='__main__':main()
