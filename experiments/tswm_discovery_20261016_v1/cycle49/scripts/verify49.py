"""Independent NumPy replay of every saved future, observed score and energy."""
import datetime
import hashlib
import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parent / 'cycle29/scripts'))
import runtime_profile
import numpy as np
from independent49 import replay, observed_score, energy


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    out = R / 'outputs/eval_v1'
    assert not (out / 'verification.json').exists()
    manifest = read(R / 'data/MANIFEST.json')
    for ent in manifest['files']:
        assert sha(R / 'data' / ent['path']) == ent['sha256']
    opening = read(out / 'OPENING_RECEIPT.json')
    assert opening['protocol_sha256'] == sha(R / 'PROTOCOL.md') and opening['EVAL_used_for_selection'] is False
    summary = read(out / 'summary.json')
    assert summary['all_rows_sha256'] == sha(out / 'all_rows.json')
    assert summary['raw_manifest_sha256'] == sha(out / 'raw_manifest.json')
    bundles = {}
    for ent in read(out / 'model_bundles.json'):
        assert sha(out / ent['path']) == ent['sha256']
        assert sha(R / 'outputs/fits' / ent['model'] / 'selected.pt') == ent['selected_sha256'] == opening['selected'][ent['model']]
        with np.load(out / ent['path']) as z:
            bundles[ent['model']] = {k: z[k].copy() for k in z.files}
    lookup = {(r['model'], r['reveal'], r['density']): r for r in summary['groups']}
    row_lookup = {(r['model'], r['reveal'], r['density'], r['physical_id']): r for r in read(out / 'all_rows.json')}
    max_model, max_energy, max_score, max_mse = 0., 0., 0., 0.
    scored = 0
    for ent in read(out / 'raw_manifest.json'):
        assert sha(out / ent['path']) == ent['sha256']
        assert sha(out / ent['probe_path']) == ent['probe_sha256']
        z = np.load(out / ent['path'])
        data = np.load(R / 'data' / ent['case'] / 'eval_observed.npz')
        physical = np.load(R / 'data' / ent['case'] / 'eval_physical_truth.npz')['signal']
        ids = data['physical_id']
        reveal, horizon = ent['reveal'], ent['horizon']
        assert np.array_equal(ids, z['physical_id']) and np.array_equal(data['density'], z['density'])
        assert np.array_equal(z['counts'], data['counts'][:, reveal:24 + reveal])
        assert np.array_equal(z['sums'], data['sums'][:, reveal:24 + reveal])
        assert np.array_equal(z['target_counts'], data['counts'][:, 24 + reveal:])
        assert np.array_equal(z['target_sums'], data['sums'][:, 24 + reveal:])
        assert np.array_equal(z['truth'], physical[ids, 24 + reveal:])
        pr = np.load(out / ent['probe_path'])
        computed = {key: [] for key in ['energy', 'tail12_energy', 'mse', 'tail12_mse', 'observation_score', 'tail12_observation_score']}
        for start in range(0, len(ids), 8):
            sl = slice(start, start + 8)
            prediction = z['prediction'][sl].astype('float64')
            truth = z['truth'][sl].astype('float64')
            independent = replay(bundles[ent['model']], z['counts'][sl], z['sums'][sl], z['density'][sl], z['noise'][sl])
            error = float(np.max(abs(prediction - independent)))
            max_model = max(max_model, error)
            assert np.allclose(prediction, independent, atol=3e-4, rtol=3e-4), (ent['path'], start, 'model replay', error)
            values = {'energy': energy(prediction, truth), 'tail12_energy': energy(prediction[:, :, -12:], truth[:, -12:]),
                      'mse': np.mean((prediction.mean(1) - truth)**2, (1, 2)),
                      'tail12_mse': np.mean((prediction[:, :, -12:].mean(1) - truth[:, -12:])**2, (1, 2)),
                      'observation_score': observed_score(prediction, z['target_counts'][sl], z['target_sums'][sl], z['density'][sl], pr['count'][:, -horizon:], pr['mark'][:, -horizon:]),
                      'tail12_observation_score': observed_score(prediction[:, :, -12:], z['target_counts'][sl, -12:], z['target_sums'][sl, -12:], z['density'][sl], pr['count'][:, -12:], pr['mark'][:, -12:])}
            for key, value in values.items():
                difference = float(np.max(abs(value - z[key][sl])))
                if 'energy' in key:
                    max_energy = max(max_energy, difference)
                    assert difference < 2e-8, (ent['path'], start, key, difference)
                elif 'score' in key:
                    max_score = max(max_score, difference)
                    assert difference < 2e-5, (ent['path'], start, key, difference)
                else:
                    max_mse = max(max_mse, difference)
                    assert difference < 2e-5, (ent['path'], start, key, difference)
                computed[key].extend(value.tolist())
        for d in [.25, 1., 4.]:
            mask = z['density'] == d
            group = lookup[(ent['model'], reveal, d)]
            for key, value in computed.items():
                assert abs(group[key] - float(np.mean(np.asarray(value)[mask]))) < (2e-8 if 'energy' in key else 2e-5)
        for j, pid in enumerate(ids):
            row = row_lookup[(ent['model'], reveal, float(z['density'][j]), int(pid))]
            for key, value in computed.items():
                assert abs(row[key] - value[j]) < (2e-8 if 'energy' in key else 2e-5)
        scored += len(ids)
        print(json.dumps({'verified_model': ent['model'], 'reveal': reveal, 'contexts': len(ids), 'cumulative_contexts': scored, 'max_model_error': max_model}), flush=True)
    initials, schedules = {}, {}
    for folder in sorted((R / 'outputs/fits').iterdir()):
        fit, config = read(folder / 'fit_summary.json'), read(folder / 'config.json')
        assert sha(folder / 'selected.pt') == fit['selected_sha256'] == opening['selected'][folder.name]
        assert sha(folder / 'last.pt') == fit['last_sha256']
        assert sha(folder / 'schedule.npz') == fit['schedule_sha256'] == config['schedule_sha256']
        assert config['protocol_sha256'] == sha(R / 'PROTOCOL.md') and config['data_manifest_sha256'] == sha(R / 'data/MANIFEST.json')
        assert config['world_sha256'] == sha(R / 'scripts/world49_v2.py')
        assert config['measurement_sha256'] == sha(R / 'scripts/measurement49.py')
        assert config['clean_physical_truth_used'] is False and config['eval_used_for_selection'] is False
        key = fit['case'], fit['seed']
        initials.setdefault(key, set()).add(fit['initial_predictive_checksum'])
        schedules.setdefault(key, set()).add(fit['schedule_sha256'])
    assert len(initials) == 4 and all(len(v) == 1 for v in initials.values()) and all(len(v) == 1 for v in schedules.values())
    assert scored == 12288 and len(summary['groups']) == 48
    result = {'status': 'PASS_FRESH_CONDITIONAL_WORLD_SCREEN', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'raw_groups': 16, 'summary_groups': 48, 'model_context_reveal_entries': scored,
              'unique_physical_paths': 512, 'paired_density_contexts': 1536, 'independent_real_domains': 0,
              'max_independent_model_error': max_model, 'max_energy_error': max_energy, 'max_observed_score_error': max_score,
              'max_mean_mse_error': max_mse, 'predictive_initializations_and_schedules_matched': True,
              'summary_sha256': sha(out / 'summary.json'), 'source_sha256': sha(pathlib.Path(__file__)),
              'scope': 'All saved innovations, every model future and every scored entry replayed independently in NumPy; source data hashes checked. No physical latent identification.'}
    (out / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
