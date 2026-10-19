"""Integrity closure of this screen, explicitly not achievement of discovery."""
from common47 import *
def main():
 v=read(R/'outputs/eval_v1/verification.json');assert v['status']=='PASS_FROZEN_FINITE_GRID_SCREEN';assert read(R/'outputs/evaluation_supervisor_v1.json')['status']=='FULL_EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE';assert read(R/'outputs/figures_v1/visual_qa.json')['status']=='VISUALLY_INSPECTED_PASS';a=read(R/'outputs/eval_v1/analysis.json');checks=[]
 def check(p,digest):assert sha(p)==digest,p;checks.append({'path':p.relative_to(R).as_posix() if p.is_relative_to(R) else str(p),'sha256':digest})
 for name,ent in read(R/'data/MANIFEST.json')['files'].items():check(R/'data'/name,ent['sha256'])
 for folder in sorted((R/'outputs/fits').iterdir()):
  f=read(folder/'fit_summary.json');cfg=read(folder/'config.json');check(folder/'selected.pt',f['selected_sha256']);check(folder/'last.pt',f['last_sha256']);check(folder/'schedule.npz',f['schedule_sha256']);check(R/'PROTOCOL.md',cfg['protocol_sha256']);check(R/'scripts/common47.py',cfg['common_sha256']);check(R/'scripts/train47.py',cfg['source_sha256'])
 for key in ['raw_manifest.json','model_bundles.json']:
  for ent in read(R/'outputs/eval_v1'/key):check(R/'outputs/eval_v1'/ent['path'],ent['sha256'])
 for src in ['HyperNSDE','TESAR-CDE']:
  for ent in read(R/'source_archive'/src/'SOURCE_TREE.json')['files']:check(R/'source_archive'/src/ent['path'],ent['sha256'])
 for name,digest in read(R/'REPORT_MANIFEST.json')['files'].items():check(R/name,digest)
 for ent in read(R/'outputs/figures_v1/visual_qa.json')['exports']:check(R/'outputs/figures_v1'/ent['path'],ent['sha256'])
 for ent in read(R/'ROOT_INDEX_RECEIPTS.json')['files']:assert sha(R.parent/ent['path'])==ent['after']
 result={'status':'EVIDENCE_SNAPSHOT_VERIFIED_GOAL_UNACHIEVED','utc':now(),'checks':len(checks),'artifacts':checks,'formal_fits':16,'formal_updates':a['formal_updates'],'raw_groups':54,'model_origin_policy_entries':13824,'unique_physical_EVAL_trajectories':512,'selected_source_unchanged':True,'new_framework_validated':False,'accepted_final_directions':0,'C44_original_query_protocol_incomplete':True,'remaining_goal_requirements':['A distinct useful framework-level learning interaction','Current primary-source differentiation from strong complete models','Fresh independent-domain and scalable neural evidence'],'source_sha256':sha(__file__)};dump(R/'SNAPSHOT_CLOSURE.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)
if __name__=='__main__':main()
