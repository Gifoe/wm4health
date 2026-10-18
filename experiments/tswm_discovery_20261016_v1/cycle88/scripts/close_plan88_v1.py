from common88_v1 import *
from finalize_plan88_v4 import public_text

def main():
    native=read(O/'PAIRED_PLAN_RESULTS88_V1.json');assert all(x['ordinary_fix_passes_original_rejection_gate'] for x in native['ordinary_controls'])
    assert read(O/'VERIFICATION88_V2.json')['metric_max_abs_difference']==0
    assert read(O/'RESEARCH_DECISION88_V1.json')['paired_response_comparable']==False
    closure='''Closed C88 decision: **retire the new-method candidate based on fixing action anticipation**. The original registered ordinary-control criterion is satisfied. The stronger supplementary post-edit response check fails and remains unresolved; no all-metrics simulation-fidelity success is claimed. Do not restart this route or tune its edit cases to salvage novelty. See DECISION_RECONCILIATION88_V1.md. Overall research goal ACTIVE/UNACHIEVED; accepted directions0.

'''
    changes=[]
    targets=[C/'README.md',C/'final_research_direction.md',C/'RUN_BOOK88_V1.md',ROOT/'CURRENT_RESEARCH_STATUS_V88.md']
    for path in targets:
        before=path.read_text(encoding='utf-8');old_sha=sha(path)
        if path.parent==ROOT:
            with (ROOT/'CURRENT_RESEARCH_STATUS_V88_BEFORE_CLOSURE.md').open('x',encoding='utf-8') as f:f.write(before)
        with path.open('w',encoding='utf-8',newline='\n') as f:f.write(closure+before)
        changes.append(dict(document=str(path.relative_to(ROOT)),before_sha256=old_sha,after_sha256=sha(path)))
    runbook=C/'RUN_BOOK88_V1.md'
    with runbook.open('a',encoding='utf-8') as f:f.write('\nFinal closure/publication step: close_plan88_v1.py preserves publication_v1, adds the explicit primary/supplementary decision distinction, and builds publication_v2. The executed expanded verifier is verify_plan88_v3.py; earlier verifier/finalizer source versions remain provenance files, not alternative claimed executions. GRU paired metrics are in outputs/GRU_RESULTS88_V1.json.\n')
    accounting=['# C88 observed compute accounting','','| Arm | Parameters | Updates | Observed elapsed seconds |','|---|---:|---:|---:|']
    for arm in ['stock','lagged','acrkn','gru']:
        fit=read(O/(f'PLAN_FIT88_V1_{arm}.json' if arm!='gru' else 'GRU_FIT88_V1.json'))
        accounting.append(f"| {arm} | {fit['parameters']} | {fit['updates']} | {fit['elapsed_seconds']:.4f} |")
    accounting+=['','These are actual run timers under a concurrent independent campaign, not isolated GPU throughput benchmarks. Equal updates do not mean equal compute. The ordinary GRU is dramatically faster in this implementation; no efficiency novelty or general model ranking is claimed. Stock/lagged manager-index controls have the same parameter count, initialization and minibatches, which is the decisive comparison for this candidate.','', 'The raw GRU network reacts to arbitrary replacement values at positions flagged missing. The declared preprocessing always zero-fills those positions before inference, so the perturbation is a separate descriptive robustness check, not evidence of future-observation access in the executed task. Native Kalman models ignore those invalid measurements directly.']
    with (C/'COMPUTE_ACCOUNTING88_V1.md').open('x',encoding='utf-8') as f:f.write('\n'.join(accounting)+'\n')
    write(O/'RESEARCH_CLOSURE88_V1.json',dict(utc=utc(),status='C88_CLOSED_PRIMARY_METHOD_ROUTE_RETIRED_SUPPLEMENTARY_RESPONSE_UNRESOLVED',primary_registered_candidate_retired=True,supplementary_response_criterion_passed=False,full_intervention_fidelity_claim=False,scientific_neural_fits=4,updates_per_fit=1000,accepted_research_directions=0,goal_status='ACTIVE_UNACHIEVED',do_not_restart_or_tune_c88=True,next_action='Check native feasibility and closest work for cross-query mechanism-edit consistency; no new expensive campaign before a discriminating test. Preserve independent original-budget C71.',documents_updated=changes,original_preclosure_snapshot_preserved=True))
    previous=read(O/'publication_v1/manifest.json');pub=O/'publication_v2';pub.mkdir(parents=True,exist_ok=True)
    rels=[r['path'] for r in previous['files']]+['scripts/close_plan88_v1.py','COMPUTE_ACCOUNTING88_V1.md','outputs/RESEARCH_CLOSURE88_V1.json']
    manifest=[]
    for rel in sorted(rels):
        src=C/rel;dest=pub/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        raw=(json.dumps(public_text(read(src)),indent=2,ensure_ascii=False,allow_nan=False) if src.suffix=='.json' else public_text(src.read_text(encoding='utf-8'))).encode('utf-8')
        with dest.open('xb') as f:f.write(raw)
        manifest.append(dict(path=rel,server_snapshot_file=str(dest),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),bytes=len(raw),encoding='utf-8'))
    write(pub/'manifest.json',dict(utc=utc(),project='tswm_discovery_20261016_v1',cycle=88,files=manifest,raw_data_weights_or_predictions_included=False,primary_method_candidate_retired=True,supplementary_response_passed=False,goal_achieved=False))
    log(dict(stage='c88_primary_route_closed',publication_manifest_sha256=sha(pub/'manifest.json'),files=len(manifest),overall_goal='ACTIVE_UNACHIEVED'))

if __name__=='__main__':main()
