"""Original frozen forecast functions with the recovered campaign's terminal gate."""
import sys,pathlib,ast,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import evaluate_free_simulation71_v1 as base
from common71_v1 import *

def campaign_gate():
 assert read(OUT/'campaign71_v4_exit.json')['exit_code']==0
 c=read(OUT/'CAMPAIGN71_V4.json');assert c['status']=='THREE_NATIVE_SUBSET_FITS_CLOSED_AFTER_DRIVER_RECOVERY_TRAIN_VAL_ONLY'
 assert c['module_updates']==96000 and c['reserved_eval_rows_accessed']==0 and [r['seed'] for r in c['fits']]==list(base.SEEDS)
 assert read(OUT/'RESUME_QUALIFICATION71_V2.json')['status']=='PASS_CHECKPOINT_CONTINUATION_EQUIVALENCE'
 for row in c['fits']:
  result=read(OUT/row['result_file']);assert read(OUT/row['exit_file'])['exit_code']==0
  assert result['epochs']==2000 and result['last']['epoch']==1999 and result['module_updates']==result['proposal_updates']==32000
  assert sha(OUT/row['result_file'])==row['result_sha256'] and result['selected']==row['selected'] and sha(row['selected']['path'])==row['selected']['sha256']
 return c

if __name__=='__main__':
 campaign_gate();p=ROOT/'scripts'/'evaluate_free_simulation71_v1.py';assert sha(p)=='a7c0ffab12752976c72c45d06977edef0c4982d96686e703d2f6ba8f15153769';tree=ast.parse(p.read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');changes=0
 for n in ast.walk(fn):
  if isinstance(n,ast.Constant) and n.value=='CAMPAIGN71_V3.json':n.value='CAMPAIGN71_V4.json';changes+=1
 assert changes==1
 helpers={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in ('main','campaign_gate')}
 dump(OUT/'ASSESSMENT_CAMPAIGN_GATE_ADAPTER71_V2.json',dict(utc=utc(),original_evaluator_sha256=sha(p),forecast_helper_bodies_unchanged=helpers,changes=['Verify recovered campaign terminal receipts','Reference CAMPAIGN71_V4.json in output provenance'],protocol_sha256=sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md'),reserved_eval_rows_accessed=0))
 base.campaign_gate=campaign_gate;exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p)+'::campaign_gate_v2','exec'),base.__dict__);base.main()
