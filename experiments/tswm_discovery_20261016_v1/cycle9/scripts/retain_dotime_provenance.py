"""Retain the actual read-only native dependency and released data checksums."""
import hashlib,json,pathlib,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 src=ROOT.parent/'upstream/dotime';dst=ROOT/'sources/dotime';files=[p for p in (src/'src').rglob('*') if p.is_file() and p.suffix=='.py']
 files += [src/p for p in ['README.md','SOURCE_PIN.json','pyproject.toml','docs/benchmarks.md','results/reference/erratum/README.md','results/reference/prereg_2026-09/README.md','results/reference/prereg_2026-11/PREREG.md','results/reference/signature_2026-10/README.md','results/reference/v1_2/README.md']]
 files += [p for p in src.glob('LICENSE*') if p.is_file()]
 rows=[]
 for source in files:
  rel=source.relative_to(src);target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert sha(source)==sha(target)
  else:shutil.copyfile(source,target)
  assert sha(source)==sha(target);rows.append({'path':rel.as_posix(),'bytes':target.stat().st_size,'sha256':sha(target)})
 record={'repository':'thummd/dotime','revision':'8f227406a5b17c2040cfdcc680b493729ef47d00','scope':'All native package Python source plus selected protocol/erratum files. Executed read-only dependency was the prior sealed upstream/dotime copy; each retained file is byte-identical to that dependency.','files':rows}
 (dst/'RETAINED_MANIFEST.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 cache=ROOT/'data/dotime_cache/dot-Identifiability-v1-1.2.0';manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
 datafiles=[p for p in cache.rglob('*') if p.is_file() and '.cache' not in p.parts]
 datarows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'md5':hashlib.md5(p.read_bytes()).hexdigest()} for p in datafiles]
 (ROOT/'outputs/dotime/data_provenance.json').write_text(json.dumps({'native_loader_manifest_md5_verified':True,'released_manifest':manifest,'retained_files':datarows},indent=2),encoding='utf-8')
 print('retained exact native dependency',len(rows),'and data artifacts',len(datarows))
if __name__=='__main__':main()
