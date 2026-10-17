"""Report every pre-registered cost point; no budget or model selection on EVAL."""
import csv
import datetime
import hashlib
import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parent / 'cycle29/scripts'))
import runtime_profile
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    folder = R / 'outputs/cost_frontier_v1'
    assert not (folder / 'paired_analysis.json').exists()
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'PASS_PREREGISTERED_TIME_FRONTIER_WITH_INDEPENDENT_REPLAY'
    assert sha(folder / 'all_rows.json') == summary['all_rows_sha256']
    assert sha(folder / 'raw_manifest.json') == summary['raw_manifest_sha256']
    for ent in read(folder / 'raw_manifest.json'):
        path = (folder / ent['path']).resolve()
        assert path.is_relative_to(R.resolve()) and sha(path) == ent['sha256']
        if not ent['reused_primary_verified']:
            assert sha(folder / ent['weight_path']) == ent['weights_sha256']
            assert ent['independent_all_path_and_score_replay'] is True
    rows = read(folder / 'all_rows.json')
    lookup = {(r['case'], r['arm'], r['seed'], r['budget_seconds'], r['density'], r['physical_id']): r for r in rows}
    aggregate, intervals = [], []
    rng = np.random.default_rng(20265003)
    for case in ['rotating', 'bistable']:
        resampling = rng.integers(0, 256, size=(5000, 256))
        for budget in [60, 120, 180]:
            for density in [.25, 1., 4.]:
                groups = [g for g in summary['groups'] if g['case'] == case and g['budget_seconds'] == budget and g['density'] == density]
                if len(groups) != 4:
                    continue
                values = {}
                for arm in ['rb_score', 'full_clock_elbo']:
                    sub = [g for g in groups if g['arm'] == arm]
                    values[arm] = {m: float(np.mean([g[m] for g in sub])) for m in ['energy', 'mse']}
                aggregate.append({'case': case, 'budget_seconds': budget, 'density': density,
                                  'rb_energy': values['rb_score']['energy'], 'elbo_energy': values['full_clock_elbo']['energy'],
                                  'energy_rb_minus_elbo': values['rb_score']['energy'] - values['full_clock_elbo']['energy'],
                                  'energy_relative_difference_pct': 100 * (values['rb_score']['energy'] / values['full_clock_elbo']['energy'] - 1),
                                  'rb_mse': values['rb_score']['mse'], 'elbo_mse': values['full_clock_elbo']['mse']})
            for metric in ['energy', 'mse']:
                delta = []
                for pid in range(256):
                    entries = []
                    for seed in [2026491, 2026492]:
                        for density in [.25, 1., 4.]:
                            left = lookup.get((case, 'rb_score', seed, budget, density, pid))
                            right = lookup.get((case, 'full_clock_elbo', seed, budget, density, pid))
                            if left is not None and right is not None:
                                entries.append(left[metric] - right[metric])
                    if len(entries) == 6:
                        delta.append(float(np.mean(entries)))
                if len(delta) != 256:
                    continue
                delta = np.array(delta)
                lo, hi = np.quantile(delta[resampling].mean(-1), [.025, .975])
                intervals.append({'case': case, 'budget_seconds': budget, 'metric': metric,
                                  'rb_minus_elbo': float(delta.mean()), 'episode_bootstrap_lower95': float(lo), 'episode_bootstrap_upper95': float(hi),
                                  'conditioning': 'fixed two trained seeds; resample physical episode IDs, retaining three density views together'})
    with (folder / 'aggregate.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    result = {'status': 'ALL_FIXED_TIME_POINTS_REPORTED_NO_EVAL_SELECTION', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'aggregate': aggregate, 'physical_episode_intervals': intervals, 'source_summary_sha256': sha(folder / 'summary.json'),
              'source_sha256': sha(pathlib.Path(__file__)), 'unavailable_selections': summary['unavailable_selections'], 'goal_complete': False}
    dump(folder / 'paired_analysis.json', result)
    report = ['# Fixed-time quality comparison', '',
              'Budgets60/120/180 seconds were sealed before primary EVAL. Best observed-VAL checkpoint available within each budget is used. All points are retained.', '',
              '| Process | Budget seconds | Density | Score energy | ELBO energy | Score minus ELBO % |',
              '|---|---:|---:|---:|---:|---:|']
    for row in aggregate:
        report.append(f"| {row['case']} | {row['budget_seconds']} | {row['density']} | {row['rb_energy']:.7f} | {row['elbo_energy']:.7f} | {row['energy_relative_difference_pct']:+.2f} |")
    report += ['', 'Energy is lower-is-better; negative differences favor the observation-averaged score. Rows average two training seeds. All density views of a physical episode remain paired.',
               'This is a measured fit-time frontier under the actual server workload; launch/setup/preflight are excluded. It is not a cross-hardware efficiency guarantee.',
               'The bootstrap describes finite test-episode uncertainty conditional on the two fitted model seeds. It does not certify real-domain utility or isolate the effect of analytic observation averaging from changing the training objective.', '']
    (folder / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'reported_aggregate_points': len(aggregate), 'unavailable_selections': summary['unavailable_selections']}), flush=True)


if __name__ == '__main__':
    main()
