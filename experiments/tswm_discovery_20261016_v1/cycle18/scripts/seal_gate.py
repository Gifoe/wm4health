"""Seal native DynaMix demonstration evidence without marking discovery complete."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
    for name in ['native_dynamix_interface','native_dynamix_three_case_demo','native_dynamix_demo_independent_verification']:
        status=json.loads((ROOT/'outputs/jobs'/name/'status.json').read_text());assert status['status']=='completed' and status['exit_code']==0
    verification=json.loads((ROOT/'outputs/native_demo/independent_verification.json').read_text());assert verification['pass']
    for name,digest in verification['verified_arrays_sha256'].items():assert sha(ROOT/'outputs/native_demo'/name)==digest
    for name in ['manifest.json','demo_manifest.json']:
        metadata=json.loads((ROOT/'outputs/source/DynaMix'/name).read_text())
        for row in metadata['files']:assert sha(ROOT/'upstream/DynaMix-python'/row['path'])==row['sha256']
    status={'cycle':18,'gate':'COMPLETED_NATIVE_DYNAMIX_THREE_CASE_DEMONSTRATION',
        'new_training':False,'source_modified':False,'published_54_system_aggregate_reproduced':False,
        'native_cases':3,'stochastic_rollouts':9,'horizon_each':10000,
        'decision':'Positive baseline evidence on native examples; no new failure or method established.',
        'MASE_note':'Current Python notebook argument order kept alongside true-denominator diagnostic; no assertion about published Julia tables.',
        'accepted_novel_directions':0,'overall_objective_achieved':False}
    (ROOT/'CYCLE_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ['ARTIFACT_MANIFEST.json','MIRROR_VERIFICATION.json'] or p.name.endswith(('.partial','.download','.tmp','.pyc')):continue
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(files),
        'total_bytes':sum(r['bytes'] for r in files),'files':files,
        'excluded':['__pycache__','*.partial','*.download','*.tmp','*.pyc','ARTIFACT_MANIFEST.json','MIRROR_VERIFICATION.json'],
        'overall_objective_achieved':False}
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['count','total_bytes','overall_objective_achieved']}))
if __name__=='__main__':main()
