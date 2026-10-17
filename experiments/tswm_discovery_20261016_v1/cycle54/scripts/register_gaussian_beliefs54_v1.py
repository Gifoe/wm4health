"""Seal the additional read-only Gaussian diagnostic before heldout scoring."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert not (OUT/'evaluation54.json').exists()
q=json.loads((OUT/'QUALIFICATION_GAUSSIAN_BELIEF54_V1.json').read_text())
assert q['status']=='PASS_INDEPENDENT_DUMMY_GAUSSIAN_DENSITY_INTEGRATION_AND_AFFINE_INVARIANCE'
assert q['formal_data_or_model_reads']==q['optimizer_updates']==0
assert q['math_script_sha256']==sha(ROOT/'scripts/gaussian_belief_math54_v1.py')
files=['GAUSSIAN_BELIEF_AUDIT54_V1.md','scripts/gaussian_belief_math54_v1.py','scripts/qualify_gaussian_belief54_v1.py','scripts/audit_gaussian_beliefs54_v1.py','scripts/watch_gaussian_beliefs54_v1.py']
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ADDITIONAL_FIXED_HORIZON_GAUSSIAN_DIAGNOSTIC_REGISTERED_BEFORE_HELDOUT','heldout_opened_at_registration':False,'files_sha256':{p:sha(ROOT/p) for p in files},'qualification_sha256':sha(OUT/'QUALIFICATION_GAUSSIAN_BELIEF54_V1.json'),'additional_optimizer_updates':0,'changes_to_original_training_selection_or_scores':False,'accepted_final_directions':0}
p=OUT/'GAUSSIAN_BELIEF_REGISTRATION54_V1.json'
with p.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({'status':record['status'],'registration_sha256':sha(p)}),flush=True)
