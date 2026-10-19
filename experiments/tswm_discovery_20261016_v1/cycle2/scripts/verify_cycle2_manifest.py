"""Verify the checkpoint without writing into the hashed artifact tree."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--allow-absent',action='store_true');args=parser.parse_args();manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text());verified=[];absent=[]
 for item in manifest['entries']:
  path=ROOT/item['path']
  if not path.exists():absent.append(item['path']);continue
  h=hashlib.sha256()
  with path.open('rb') as f:
   while chunk:=f.read(1024*1024):h.update(chunk)
  assert path.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256'],item['path'];verified.append(item['path'])
 assert not absent or args.allow_absent,absent
 print(json.dumps({'verified_files':len(verified),'absent_files':len(absent),'absent_allowed':args.allow_absent,'all_present_files_match':True,'manifest_training_cells':manifest['model_training_cells_completed'],'overall_research_objective_achieved':False}))
if __name__=='__main__':main()
