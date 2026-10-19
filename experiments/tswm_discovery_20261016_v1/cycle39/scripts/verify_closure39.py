"""Final bounded evidence checks, not a completion audit of the research goal."""
import pathlib,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
checked=[];M=read(R/'REPORT_MANIFEST.json')
for name,digest in M['reports'].items():assert sha(R/name)==digest;checked.append('report:'+name)
for x in read(R/'SOURCE_RECEIPTS.json'):
    assert x['saved'] and sha(R/'source_archive'/x['file'])==x['sha256'];checked.append('source:'+x['file'])
assert len(read(R/'SOURCE_RECEIPTS.json'))==9
for name in ['mainstream_queries39.md','critic_queries39.md','constructive_queries39.md']:
    assert (R/name).stat().st_size>1000;checked.append('independent_note:'+name)
for x in read(R/'ROOT_INDEX_RECEIPTS.json')['files']:assert sha(B/x['path'])==x['sha256'];checked.append('root:'+x['path'])
Q=R/'outputs/native_queries_v2';K=R/'outputs/kernel_controls_v1';V=read(Q/'verification.json');W=read(K/'verification.json');A=R/'outputs/analysis_v1'
assert V['passed'] and W['passed'] and V['results_sha256']==sha(Q/'result.json') and W['results_sha256']==sha(K/'result.json')
assert M['query_verification_sha256']==sha(Q/'verification.json') and M['kernel_verification_sha256']==sha(K/'verification.json')
assert M['analysis_sha256']==sha(A/'analysis.json')
assert read(R/'outputs/supervisor_v2.json')['status']=='completed' and read(R/'outputs/supervisor_kernel_v1.json')['status']=='completed'
assert read(R/'outputs/supervisor_v1.json')['status']=='failed' and (R/'outputs/native_queries_v1/failure.json').exists()
assert not (R/'outputs/native_queries_v1/result.json').exists()
for name in ['query_moments.png','query_moments.pdf','ordinary_kernel_controls.png','ordinary_kernel_controls.pdf',
             'all_query_moments.csv','all_kernel_scores.csv']:assert (A/name).stat().st_size>100;checked.append('artifact:'+name)
assert '<!-- cycle39-complete-index-start -->' in (B/'CURRENT_RESEARCH_STATUS.md').read_text(encoding='utf-8')
assert V['moment_rows_verified']==864 and W['score_cells_verified']==256 and not V['test_scored'] and not W['test_scored']
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence_checks':len(checked),'checks':checked,
     'query_moment_rows':864,'kernel_score_cells':256,'max_query_summary_abs':V['max_summary_abs'],'max_kernel_score_abs':W['max_score_abs'],
     'raw_contexts':8,'rep_type':'Monte Carlo repeats, not training seeds','neural_updates':0,'ar_history_fits':8,
     'test_scored':False,'accepted_directions':0,'goal':'ACTIVE_UNACHIEVED','verifier_sha256':sha(pathlib.Path(__file__))}
p=R/'closure_verification.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
