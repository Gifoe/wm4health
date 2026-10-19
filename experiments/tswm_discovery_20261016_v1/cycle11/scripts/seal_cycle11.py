"""Seal completed evidence; the discovery objective remains unachieved."""
import datetime, hashlib, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(1048576):
            h.update(b)
    return h.hexdigest()

def main():
    assert not (ROOT / 'ARTIFACT_MANIFEST.json').exists()
    verification = json.loads((ROOT / 'outputs/evidence_verification.json').read_text(encoding='utf-8'))
    assert verification['all_checks_passed'] and verification['oracle_grid_rows_replayed'] == 60
    assert verification['scientific_jobs_complete'] and not verification['novel_direction_accepted']
    learned = json.loads((ROOT / 'outputs/score_learning/results.json').read_text(encoding='utf-8'))
    assert len(learned['rows']) == 12
    assert len(list((ROOT / 'outputs/score_learning').glob('*.pt'))) == 12
    status = json.loads((ROOT / 'outputs/task_status.json').read_text(encoding='utf-8'))
    assert not status['objective_achieved'] and status['cycle11']['fits'] == 12
    literature = json.loads((ROOT / 'outputs/literature/registry.json').read_text(encoding='utf-8'))
    assert len(literature['papers']) == 6 and not literature['errors']
    for row in literature['papers']:
        assert sha(ROOT / row['path']) == row['sha256']
    for name in ['literature_review.md', 'baseline_registry.md', 'experimental_results.md',
                 'negative_evidence.md', 'hypothesis_space.md', 'research_ideas.md',
                 'final_research_direction.md']:
        assert (ROOT / name).is_file()
    assert 'general time-series world model' in (ROOT / 'final_research_direction.md').read_text(encoding='utf-8')
    assert 'cycle11-score-learning completion and verification' in (ROOT / 'progress/experiment-log.md').read_text(encoding='utf-8')
    files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and p.name != 'ARTIFACT_MANIFEST.json'
             and not p.name.endswith(('.partial', '.download', '.tmp'))]
    rows = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
             'sha256': sha(p)} for p in sorted(files)]
    manifest = {'root_directory': 'cycle11', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'files': rows, 'count': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                'objective_achieved': False,
                'decision': 'Retain exact score witness and negative learning evidence; no accepted novel direction. General time-series world-model scope takes priority.',
                'exclusions': ['Bytecode and incomplete transfers; runtime outside this cycle; no full native-model reproduction claimed.']}
    (ROOT / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for row in rows:
        assert sha(ROOT / row['path']) == row['sha256']
    print(json.dumps({'count': len(rows), 'bytes': manifest['bytes'],
                      'all_server_hashes_verified': True, 'objective_achieved': False}))

if __name__ == '__main__':
    main()
