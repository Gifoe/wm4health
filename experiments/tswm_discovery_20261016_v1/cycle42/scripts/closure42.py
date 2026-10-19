"""Bounded artifact closure; overall research objective remains unachieved."""
import pathlib,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent;O=R/'outputs';F=O/'fits_v1'
load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'closure_verification.json').exists()
M=load(R/'REPORT_MANIFEST.json');checks=[]
for name,digest in M['reports'].items():assert sha(R/name)==digest;checks.append('report:'+name)
for path,key in [(O/'analysis_v1/analysis.json','analysis_sha256'),(O/'verification_v1.json','verification_sha256'),(F/'results.json','results_sha256'),(R/'PROTOCOL.md','protocol_sha256')]:assert sha(path)==M[key];checks.append(key)
q=load(O/'qualification_v1/qualification.json');r=load(F/'results.json');v=load(O/'verification_v1.json');a=load(O/'analysis_v1/analysis.json')
assert q['status']==r['status']==v['status']==a['status']=='PASS'
assert q['common_sha256']==r['common_sha256']==sha(R/'scripts/common42.py')
assert q['source_sha256']==sha(R/'scripts/qualify42.py') and r['training_source_sha256']==sha(R/'scripts/train42.py')
assert r['fits']==v['independently_rebuilt_models']==32 and r['formal_updates']==48000
assert load(F/'status.json')['status']==load(O/'supervisor_v1.json')['status']=='completed'
manifest=[]
for file in sorted(F.rglob('*')):
 if file.is_file() and file.suffix in ['.pt','.npz','.npy']:
  manifest.append({'path':file.relative_to(R).as_posix(),'bytes':file.stat().st_size,'sha256':sha(file)})
assert len(manifest)==76,len(manifest)
for row in r['rows']:
 label=f"w{row['world']}_n{row['n_per_seen_cell']}_i{row['init_seed']}_o{row['obs_mod']}d{row['dyn_mod']}"
 assert sha(F/label/'model.pt')==row['checkpoint_sha256'] and sha(F/label/'scores.npz')==row['score_sha256']
for x in load(R/'ROOT_INDEX_RECEIPTS.json')['files']:assert sha(B/x['path'])==x['sha256'];checks.append('root:'+x['path'])
for name in ['factorization.png','factorization.pdf']:assert (O/'analysis_v1'/name).stat().st_size>100;checks.append('figure:'+name)
assert a['accepted_final_directions']==0 and a['goal']=='ACTIVE_UNACHIEVED' and not a['old_real_test_scored']
(R/'RAW_ARTIFACT_MANIFEST.json').write_text(json.dumps({'files':manifest},indent=2),encoding='utf-8')
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'raw_files':len(manifest),'raw_bytes':sum(x['bytes'] for x in manifest),'formal_fits':32,'formal_updates':48000,'raw_sequence_scores':v['raw_sequence_scores'],'next_distribution_cells':v['next_distribution_cells'],'max_independent_nll_abs':v['max_sequence_nll_abs'],'new_method_fits':0,'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED','old_real_test_scored':False,'synthetic_evaluation_scored':True,'raw_manifest_sha256':sha(R/'RAW_ARTIFACT_MANIFEST.json'),'verifier_sha256':sha(pathlib.Path(__file__))}
(R/'closure_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
