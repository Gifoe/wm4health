"""Safely restore reviewed literature/search artifacts and record runtime detail."""
import runtime_profile
import datetime,hashlib,json,pathlib,sys,zipfile
import torch,numpy,pandas
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 with zipfile.ZipFile(ROOT/'outputs/literature_bundle.zip') as z:
  assert len(z.infolist())<30
  for row in z.infolist():
   rel=pathlib.PurePosixPath(row.filename);assert not rel.is_absolute() and '..' not in rel.parts
   assert row.filename=='literature_review.md' or row.filename.startswith(('outputs/literature/','literature-search-20261016-acquisition-policy/'))
   p=ROOT.joinpath(*rel.parts);assert p.resolve().is_relative_to(ROOT.resolve());assert row.file_size<15_000_000
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(row))
 reg=json.loads((ROOT/'outputs/literature/registry.json').read_text(encoding='utf-8'));assert len(reg['papers'])==6 and not reg['errors']
 for row in reg['papers']:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
 detail={'python':sys.version,'prefix':sys.prefix,'torch':torch.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,
  'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
  'matmul_precision':torch.get_float32_matmul_precision(),'scientific_main_environment_modified':False,
  'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'post-fit detail probe;raw per-run environment files retained unchanged'}
 (ROOT/'outputs/runtime_detail.json').write_text(json.dumps(detail,indent=2),encoding='utf-8')
 print(json.dumps({'primary_papers_verified':6,'search_artifacts_staged':True,'python':sys.version.split()[0]}))

if __name__=='__main__':main()
