"""Evidence integrity only; explicitly not the research completion criterion."""
from common46 import *
import datetime
def main():
 verification=read(R/'outputs/eval_v1/verification.json');assert verification['status']=='PASS_FOR_FROZEN_CONTROLLED_SCREEN';assert read(R/'outputs/train_val_diagnostics_v1/verification.json')['status']=='PASS_TRAIN_VAL_DIAGNOSTICS_ONLY';assert read(R/'outputs/evaluation_supervisor_v1.json')['status']=='EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE';assert read(R/'outputs/report_supervisor_v1.json')['status']=='REPORTS_COMPLETE_VISUAL_QA_PENDING';qa=read(R/'outputs/figures_v1/visual_qa.json');assert qa['status']=='VISUALLY_INSPECTED_PASS';checks=[]
 def check(p,expected):assert sha(p)==expected,p;checks.append({'path':str(p.relative_to(R)) if p.is_relative_to(R) else str(p),'sha256':expected})
 data=read(R/'data/MANIFEST.json')
 for name,expected in data['files'].items():check(R/'data'/name,expected)
 for folder in sorted((R/'outputs/fits').iterdir()):
  fit=read(folder/'fit_summary.json');assert fit['status']=='FIT_COMPLETE_EVAL_UNOPENED';check(folder/'selected.pt',fit['selected_sha256']);check(folder/'last.pt',fit['last_sha256']);check(folder/'schedule.npz',fit['schedule_sha256'])
 for ent in read(R/'outputs/eval_v1/raw_manifest.json'):check(R/'outputs/eval_v1'/ent['path'],ent['sha256'])
 for ent in qa['exports']:check(R/'outputs/figures_v1'/ent['path'],ent['sha256'])
 for ent in read(R/'ROOT_INDEX_RECEIPTS.json')['files']:assert sha(R.parent/ent['path'])==ent['after']
 for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md','SCIENTIFIC_DECISION.md','PRIORITY_DECISION.md','DIAGNOSTIC_INTERPRETATION.md','RESEARCH_BRANCH_NOTES.md']:
  p=R/name;assert p.exists() and p.stat().st_size>100;check(p,sha(p))
 manifest=read(SRC/'SOURCE_TREE.json')
 for ent in manifest['files']:assert sha(SRC/ent['path'])==ent['sha256']
 result={'status':'EVIDENCE_SNAPSHOT_VERIFIED_GOAL_UNACHIEVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':len(checks),'artifacts':checks,'formal_fits':8,'formal_updates':27250,'scored_origins':verification['verified_origins'],'raw_groups':verification['raw_groups'],'author_core_unchanged':True,'accepted_final_directions':0,'new_method_effect_established':False,'C44_full_query_protocol_still_incomplete':True,'remaining_goal_requirements':['A distinct valuable world-model research question and concrete learning interaction','Current primary-source differentiation from complete ordinary methods','Independent held-out and cross-domain evidence'],'source_sha256':sha(__file__)};dump(R/'SNAPSHOT_CLOSURE.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)
if __name__=='__main__':main()
