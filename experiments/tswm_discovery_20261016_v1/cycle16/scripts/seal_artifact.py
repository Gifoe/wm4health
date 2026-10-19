"""Seal the completed cycle 16 gate; this does not complete the research objective."""
import datetime, hashlib, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDED_NAMES = {'ARTIFACT_MANIFEST.json', 'MIRROR_VERIFICATION.json'}
EXCLUDED_SUFFIXES = ('.partial', '.download', '.tmp', '.pyc')

def canonical(path):
    return (path.is_file() and '__pycache__' not in path.parts
            and path.name not in EXCLUDED_NAMES
            and not path.name.endswith(EXCLUDED_SUFFIXES))

def main():
    assert not (ROOT / 'ARTIFACT_MANIFEST.json').exists(), 'Already sealed'
    expected = {
        'acquire_metaid_release_weight': ('failed', 1),
        'completed_pilot_integrity': ('completed', 0),
        'data_and_references': ('completed', 0),
        'diagnostic_smoke_step100': ('completed', 0),
        'learned_covariance_diagnostic': ('completed', 0),
        'metaid_checkpoint_interface': ('completed', 0),
        'native_mts3_NP_seed2021_attempt1': ('completed', 0),
        'preflight_attempt0': ('failed', 1),
        'preflight_attempt1': ('failed', 1),
        'preflight_attempt2': ('completed', 0),
    }
    observed = {}
    for job, pair in expected.items():
        status = json.loads((ROOT / 'outputs/jobs' / job / 'status.json').read_text())
        assert (status['status'], status['exit_code']) == pair, (job, status)
        observed[job] = {'status': status['status'], 'exit_code': status['exit_code']}
    abandoned = json.loads((ROOT / 'outputs/jobs/native_mts3_NP_seed2021/launch_observation.json').read_text(encoding='utf-8-sig'))
    assert not abandoned['SupervisorPresent'] and not abandoned['TrainingChildPresent']
    for path in ['outputs/analysis/completed_pilot_verification.json',
                 'outputs/diagnostics/final_step001100/independent_verification.json',
                 'outputs/metaid_preflight/results.json', 'REPRODUCIBILITY.md']:
        assert (ROOT / path).is_file(), path
    status = {
        'cycle': 16, 'scientific_gate': 'COMPLETED_NO_GO_FOR_COVARIANCE_REPAIR_LINE',
        'overall_research_objective_achieved': False, 'accepted_novel_directions': 0,
        'test_evaluated': False, 'completed_training_updates': 1500,
        'selected_validation_step': 1100,
        'primary_result': 'Post-hoc joint covariance replacement did not improve validation NLL.',
        'scope_limit': 'One transferred native MTS3 fit; no claim about all corrected retraining or world models.',
        'next_baseline': 'Released probabilistic meta-system-ID checkpoint; native interface only verified.',
        'expected_terminal_jobs': observed,
        'abandoned_launch': 'Preserved with absence observation; no exit code invented.',
        'source_dependencies': {'MTS3': '../cycle15/upstream/MTS3',
            'MTS3_revision': 'ab971c0ec43df7664c4b727ca69d5b8dd0cab5da'},
    }
    (ROOT / 'CYCLE_STATUS.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    files = []
    for path in sorted(ROOT.rglob('*')):
        if canonical(path):
            files.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'All canonical files below cycle16; parent environment and sibling MTS3 source are referenced separately.',
        'excluded': ['__pycache__', '*.partial', '*.download', '*.tmp', '*.pyc', *sorted(EXCLUDED_NAMES)],
        'transport_partials_retained': True, 'count': len(files),
        'total_bytes': sum(item['bytes'] for item in files), 'files': files,
        'overall_research_objective_achieved': False}
    (ROOT / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in ['count', 'total_bytes', 'overall_research_objective_achieved']}))

if __name__ == '__main__':
    main()
