"""Read scientific checks and create/verify a cycle-local immutable file manifest."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gate():
    for relative in ['outputs/event_audit_attempt1/independent_verification.json','outputs/attribution/independent_verification.json','outputs/system_id_control/independent_verification.json']:
        assert json.loads((ROOT/relative).read_text())['pass'],relative
    for folder in (ROOT/'outputs/jobs').iterdir():
        status=json.loads((folder/'status.json').read_text());assert status['status'] in ['completed','failed']
        if folder.name=='retained_native_path_event_audit':
            assert status['exit_code']==1 and 'Out of range float values are not JSON compliant: nan' in (folder/'stderr.log').read_text()
        else:assert status['status']=='completed' and status['exit_code']==0,folder.name
        assert sha(folder/'script_snapshot.py')==status['script_sha256']
    for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:
        assert (ROOT/name).is_file()
    decision=json.loads((ROOT/'outputs/research_decision.json').read_text());assert not decision['objective_achieved'] and decision['accepted_novel_directions']==0
    array_receipt=json.loads((ROOT/'outputs/array_mirror_receipt.json').read_text());assert array_receipt['count']==23
    for row in array_receipt['files']:assert sha(ROOT/row['path'])==row['sha256']
    assert sha(ROOT.parent/'cycle18/ARTIFACT_MANIFEST.json')=='1820dda9443e66613563a81df7bf118feaca9c9c9e6a0ef31baa48b7abf224a6'
    for name in ['lorenz63','chua','sprottM']:
        assert sha(ROOT/'outputs/event_audit'/(name+'_analogue.npz'))==sha(ROOT/'outputs/event_audit_attempt1'/(name+'_analogue.npz'))
def inventory():
    return [dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name!='ARTIFACT_MANIFEST.json' and '__pycache__' not in p.parts]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--create',action='store_true');args=parser.parse_args();gate()
    path=ROOT/'ARTIFACT_MANIFEST.json';files=inventory()
    if args.create:
        assert not path.exists();assert not any(row['path'].endswith(('.download','.tmp')) for row in files)
        manifest=dict(cycle=19,files=files,count=len(files),total_bytes=sum(r['bytes'] for r in files),
            accepted_novel_directions=0,overall_objective_achieved=False,scope='Completed exploratory audit, not an accepted novel research direction')
        path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    else:
        manifest=json.loads(path.read_text());assert files==manifest['files']
    print(json.dumps(dict(pass_=True,count=manifest['count'],bytes=manifest['total_bytes'],manifest_sha256=sha(path),objective_achieved=False)))
if __name__=='__main__':main()
