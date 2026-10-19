"""Canonical artifact seal of a completed negative research gate."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def canonical(p):
    rel=p.relative_to(ROOT)
    return (p.is_file() and not any(x in rel.parts for x in ['runtime_dependencies','runtime_numba_cache','__pycache__'])
        and p.name not in ['ARTIFACT_MANIFEST.json','MIRROR_VERIFICATION.json']
        and not p.name.endswith(('.partial','.download','.tmp','.pyc','.nbi','.nbc')))
def main():
    assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
    expected={'native_metaid_cross_system_gate':('failed',1),'native_metaid_cross_system_gate_attempt1':('failed',1),
        'isolated_dependencies':('completed',0),'isolated_odeintw':('completed',0),
        'native_metaid_cross_system_gate_attempt2':('completed',0),'native_gate_independent_verification':('completed',0)}
    for name,pair in expected.items():
        status=json.loads((ROOT/'outputs/jobs'/name/'status.json').read_text())
        assert (status['status'],status['exit_code'])==pair
    verify=json.loads((ROOT/'outputs/native_gate_attempt2/independent_verification.json').read_text());assert verify['pass']
    for name,digest in verify['verified_array_sha256'].items():
        assert hashlib.sha256((ROOT/'outputs/native_gate_attempt2'/name).read_bytes()).hexdigest()==digest
    status={'cycle':17,'gate':'COMPLETED_NATIVE_BASELINE_AND_ORDINARY_CONTROL_NO_NEW_METHOD',
        'accepted_novel_directions':0,'overall_objective_achieved':False,
        'calibration_systems':256,'probe_systems':256,'new_training':False,
        'interpretation':'Substantial ordinary covariance repair; generic joint-uncertainty claim rejected as new method.',
        'limitations':['Native full-sequence future-containing normalization.','One system family and released checkpoint.',
            'Residual joint miscalibration not fully explained or claimed solved.'],
        'dependencies':'Sibling cycle16 source/weight, plus isolated package hashes in outputs/*install_report.json.'}
    (ROOT/'CYCLE_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.rglob('*')) if canonical(p)]
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(rows),
        'total_bytes':sum(p['bytes'] for p in rows),'files':rows,
        'excluded':['runtime_dependencies (reinstall pinned distributions)','runtime_numba_cache','__pycache__',
            '*.partial','*.download','*.tmp','*.pyc','*.nbi','*.nbc','ARTIFACT_MANIFEST.json','MIRROR_VERIFICATION.json'],
        'referenced_not_copied':'cycle16 native source and public checkpoint already sealed and mirrored',
        'overall_objective_achieved':False}
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['count','total_bytes','overall_objective_achieved']}))
if __name__=='__main__':main()
