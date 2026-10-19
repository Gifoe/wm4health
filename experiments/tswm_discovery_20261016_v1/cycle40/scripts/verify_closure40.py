"""Current-cycle bounded closure; never declare research objective achieved."""
import pathlib,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
M=read(R/'REPORT_MANIFEST.json');checks=[]
for name,digest in M['reports'].items():assert sha(R/name)==digest;checks.append('report:'+name)
V=read(R/'outputs/verification.json');I=read(R/'outputs/information_verification.json');D=read(R/'outputs/raw_data_verification.json')
assert V['status']==I['status']==D['status']=='PASS' and V['fits']==8 and V['raw_path_archives']==20
assert len(I['checks'])==24 and not V['test_scored'] and not I['test_scored'] and not D['test_scored']
assert M['verification_sha256']==sha(R/'outputs/verification.json') and M['analysis_sha256']==sha(R/'outputs/analysis_v1/analysis.json')
assert M['information_verification_sha256']==sha(R/'outputs/information_verification.json') and M['raw_data_verification_sha256']==sha(R/'outputs/raw_data_verification.json')
assert read(R/'outputs/supervisor_v1.json')['status']=='failed' and read(R/'outputs/supervisor_v2.json')['status']=='completed'
assert read(R/'outputs/closure_worker_v2.json')['stages'][0]['exit_code']==1
assert read(R/'outputs/closure_worker_v3.json')['status']=='completed'
for receipt in read(R/'ROOT_INDEX_RECEIPTS.json')['files']:
 assert sha(B/receipt['path'])==receipt['sha256'];checks.append('root:'+receipt['path'])
L=R/'source_archive/lifts';cache=read(L/'CACHE_RECEIPTS.json')
assert not cache['executed'] and len(cache['files'])==24
for file in cache['files']:assert sha(L/file['path'])==file['sha256'];checks.append('cached_source:'+file['path'])
for name in ['all_origin_scores.csv','mean_quality_cost.csv','fixed_tail_quality.png','fixed_tail_quality.pdf']:
 assert (R/'outputs/analysis_v1'/name).stat().st_size>100;checks.append('artifact:'+name)
assert '<!-- cycle40-complete-index-start -->' in (B/'CURRENT_RESEARCH_STATUS.md').read_text(encoding='utf-8')
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'count':len(checks),'fits':8,'formal_updates':16000,'discarded_synthetic_preflight_updates':200,'score_cells':V['independent_score_cells'],'max_score_difference':V['max_score_difference'],'unavailable_future_corruption_checks':24,'source_cache_files':24,'test_scored':False,'accepted_directions':0,'goal':'ACTIVE_UNACHIEVED','verifier_sha256':sha(pathlib.Path(__file__))}
assert not (R/'closure_verification.json').exists();(R/'closure_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
