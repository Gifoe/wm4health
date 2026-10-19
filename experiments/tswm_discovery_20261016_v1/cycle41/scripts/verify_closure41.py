"""Close evidence artifacts, never equate cycle completion with research success."""
import pathlib,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
M=read(R/'REPORT_MANIFEST.json');checks=[]
for name,digest in M['reports'].items():assert sha(R/name)==digest;checks.append('report:'+name)
N=R/'outputs/native_v1';D=R/'outputs/controlled_v2';n=read(N/'qualification.json');v=read(D/'verification_v2.json');c=read(D/'result.json')
assert n['status']==v['status']==c['status']=='PASS' and v['cross_rows']==2304 and v['independent_cross_metric_cells']==9216
assert n['no_accuracy_or_convergence_claim'] and read(R/'outputs/supervisor_v1.json')['status']=='completed'
for name,key in [('fixture.npz','fixture_sha256'),('native_minimal.pt','checkpoint_sha256'),('native_samples.npz','samples_sha256')]:assert sha(N/name)==n[key];checks.append('native:'+name)
for item in read(N/'source_manifest.json')['files']:assert sha(R/'upstream/lifts'/item['path'])==item['sha256'];checks.append('native_source:'+item['path'])
assert len(read(N/'source_manifest.json')['files'])==26
assert sha(D/'raw_controls.npz')==c['raw_sha256']==v['raw_sha256']
assert sha(R/'CONTROLLED_PLAN.md')==c['protocol_sha256']
assert M['native_qualification_sha256']==sha(N/'qualification.json') and M['control_verification_sha256']==sha(D/'verification_v2.json')
assert M['analysis_sha256']==sha(R/'outputs/analysis_v1/analysis.json')
for name in ['mainstream_state41.md','critic_state41.md','constructive_state41.md']:assert (R/name).stat().st_size>5000;checks.append('independent_note:'+name)
for x in read(R/'ROOT_INDEX_RECEIPTS.json')['files']:assert sha(B/x['path'])==x['sha256'];checks.append('root:'+x['path'])
assert (R/'outputs/controlled_v1/failure.json').exists() and (R/'outputs/controlled_v2/verification.json').exists()
for name in ['anchor_information.png','anchor_information.pdf']:assert (R/'outputs/analysis_v1'/name).stat().st_size>100;checks.append('figure:'+name)
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(checks),'checks':checks,'native_example_fit':1,'native_example_updates':15,'new_comparative_neural_fits':0,'control_rows':2304,'independent_control_cells':9216,'control_max_abs':v['max_statistic_abs_difference'],'independent_gaussian_predictions':12,'gaussian_max_abs':v['max_dense_gaussian_abs_difference'],'old_real_test_scored':False,'synthetic_known_law_controls_scored':True,'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED','verifier_sha256':sha(pathlib.Path(__file__))}
assert not (R/'closure_verification.json').exists();(R/'closure_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
