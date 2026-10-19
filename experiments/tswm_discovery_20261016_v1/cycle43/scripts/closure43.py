"""Close bounded native qualification artifacts; never claim research completion."""
import pathlib,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent;O=R/'outputs';load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'closure_verification.json').exists();M=load(R/'REPORT_MANIFEST.json');checks=[]
for name,digest in M['reports'].items():assert sha(R/name)==digest;checks.append('report:'+name)
q=load(O/'native_v3/qualification.json');v=load(O/'verification_v1.json');assert q['status']=='PASS_WITH_NARROW_METADATA_REPAIR' and v['status']=='PASS'
assert q['discarded_updates']==50 and not q['comparative_effects_measured'] and not v['comparative_effects_measured']
assert v['independent_scheduler_values']==9000 and v['max_schedule_abs']==v['checkpoint_replay_max_abs']==v['independent_within_level_resume_max_abs']==0
assert M['qualification_sha256']==sha(O/'native_v3/qualification.json')==v['qualification_sha256']
assert M['verification_sha256']==sha(O/'verification_v1.json')
assert M['source_receipts_sha256']==sha(R/'source_archive/spf/CACHE_RECEIPTS.json') and M['runtime_adapter_sha256']==sha(R/'runtime_source/spf/ADAPTER_MANIFEST.json')
for x in load(R/'source_archive/spf/CACHE_RECEIPTS.json')['files']:assert sha(R/'source_archive/spf'/x['path'])==x['sha256'];checks.append('original:'+x['path'])
assert len(load(R/'source_archive/spf/CACHE_RECEIPTS.json')['files'])==29
for x in load(R/'runtime_source/spf/ADAPTER_MANIFEST.json')['files']:
 assert sha(R/'runtime_source/spf'/x['path'])==x['sha256']
 if not x['import_only_change']:assert x['sha256']==x['original_sha256']
 checks.append('runtime:'+x['path'])
for x in load(R/'assets/WHEEL_RECEIPTS.json')['files']:assert sha(R/'assets'/x['file'])==x['sha256'];checks.append('wheel:'+x['file'])
for path,key in [(O/'native_v3/preflight.pt','checkpoint_sha256'),(O/'native_v3/qualification_samples.npz','samples_sha256')]:assert sha(path)==q[key];checks.append(key)
assert sha(O/'native_v3/independent_api_replay.npz')==v['replay_raw_sha256'];checks.append('independent_api_replay')
assert load(O/'native_v2/qualification.json')['status']=='FAIL' and (O/'native_v2/original_api_failures.json').exists();checks.append('preserved_original_failure')
for x in load(R/'ROOT_INDEX_RECEIPTS.json')['files']:assert sha(B/x['path'])==x['sha256'];checks.append('root:'+x['path'])
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'check_count':len(checks),'checks':checks,'original_source_files':29,'runtime_import_only_modified_files':1,'discarded_preflight_updates':100,'formal_comparative_fits':0,'independent_scheduler_values':9000,'max_schedule_abs':v['max_schedule_abs'],'checkpoint_replay_max_abs':v['checkpoint_replay_max_abs'],'within_scale_resume_max_abs':v['independent_within_level_resume_max_abs'],'cross_scale_prefix_max_abs':v['cross_level_prefix_max_abs'],'comparative_effects_measured':False,'old_real_test_scored':False,'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED','verifier_sha256':sha(pathlib.Path(__file__))}
(R/'closure_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
