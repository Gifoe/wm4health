"""Paired descriptive/episode-bootstrap analysis after independent verification."""
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
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def csv_save(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    evaluation = R / 'outputs/eval_v1'
    verification = read(evaluation / 'verification.json')
    assert verification['status'] == 'PASS_FRESH_CONDITIONAL_WORLD_SCREEN'
    assert verification['summary_sha256'] == sha(evaluation / 'summary.json')
    summary = read(evaluation / 'summary.json')
    rows = read(evaluation / 'all_rows.json')
    assert summary['all_rows_sha256'] == sha(evaluation / 'all_rows.json')
    assert len(rows) == 12288 and len(summary['groups']) == 48
    out = R / 'outputs/analysis_v1'
    assert not out.exists()
    out.mkdir()
    metrics = ['energy', 'tail12_energy', 'mse', 'tail12_mse', 'observation_score', 'tail12_observation_score']
    lookup = {(r['case'], r['arm'], r['seed'], r['reveal'], r['density'], r['physical_id']): r for r in rows}
    aggregates, comparisons, revealed = [], [], []
    for case in ['rotating', 'bistable']:
        for density in [.25, 1., 4.]:
            for arm in ['rb_score', 'full_clock_elbo']:
                for reveal in [0, 12]:
                    gs = [g for g in summary['groups'] if g['case'] == case and g['density'] == density and g['arm'] == arm and g['reveal'] == reveal]
                    assert len(gs) == 2
                    ent = {'case': case, 'density': density, 'arm': arm, 'reveal': reveal, 'physical_paths': 256, 'training_seeds': 2}
                    for metric in metrics:
                        values = [g[metric] for g in gs]
                        ent.update({metric + '_mean': float(np.mean(values)), metric + '_min_seed': min(values), metric + '_max_seed': max(values)})
                    aggregates.append(ent)
            for seed in [2026491, 2026492]:
                for reveal in [0, 12]:
                    for metric in metrics:
                        left = np.array([lookup[case, 'rb_score', seed, reveal, density, pid][metric] for pid in range(256)])
                        right = np.array([lookup[case, 'full_clock_elbo', seed, reveal, density, pid][metric] for pid in range(256)])
                        difference = left - right
                        comparisons.append({'case': case, 'density': density, 'seed': seed, 'reveal': reveal, 'metric': metric,
                                            'rb_score': float(left.mean()), 'full_clock_elbo': float(right.mean()),
                                            'rb_minus_elbo': float(difference.mean()),
                                            'relative_difference_pct': float(100 * difference.mean() / right.mean()) if 'score' not in metric else None,
                                            'fraction_physical_paths_rb_better': float(np.mean(difference < 0))})
                for arm in ['rb_score', 'full_clock_elbo']:
                    for metric in ['tail12_energy', 'tail12_mse', 'tail12_observation_score']:
                        before = np.array([lookup[case, arm, seed, 0, density, pid][metric] for pid in range(256)])
                        after = np.array([lookup[case, arm, seed, 12, density, pid][metric] for pid in range(256)])
                        revealed.append({'case': case, 'density': density, 'seed': seed, 'arm': arm, 'metric': metric,
                                         'same_target': 'physical future bins37 through48', 'unrevealed': float(before.mean()), 'revealed': float(after.mean()),
                                         'revealed_minus_unrevealed': float((after - before).mean()),
                                         'fraction_physical_paths_reveal_better': float(np.mean(after < before))})
    intervals = []
    rng = np.random.default_rng(20265001)
    # Resample physical episode IDs, preserving all three paired density views.
    # This interval is conditional on the two already-fitted model seeds.
    for case in ['rotating', 'bistable']:
        resampling = rng.integers(0, 256, size=(5000, 256))
        for reveal in [0, 12]:
            for metric in metrics:
                values = []
                for pid in range(256):
                    values.append(np.mean([lookup[case, 'rb_score', seed, reveal, d, pid][metric]
                                           - lookup[case, 'full_clock_elbo', seed, reveal, d, pid][metric]
                                           for seed in [2026491, 2026492] for d in [.25, 1., 4.]]))
                delta = np.array(values)
                boot = delta[resampling].mean(-1)
                lo, hi = np.quantile(boot, [.025, .975])
                intervals.append({'case': case, 'reveal': reveal, 'metric': metric, 'rb_minus_elbo': float(delta.mean()),
                                  'episode_bootstrap_lower95': float(lo), 'episode_bootstrap_upper95': float(hi),
                                  'unit': '256 independent physical episodes; all density views retained together',
                                  'conditioning': 'two fitted seeds fixed, not uncertainty over model training or real domains'})
    fits, costs = [], []
    for fp in sorted((R / 'outputs/fits').glob('*/fit_summary.json')):
        fit = read(fp)
        curve = read(fp.parent / 'validation.json')
        selected = next(v for v in curve if v['step'] == fit['selected_step'])
        costs.append({'model': fit['tag'], 'case': fit['case'], 'arm': fit['arm'], 'seed': fit['seed'],
                      'updates': fit['updates'], 'selected_step': fit['selected_step'],
                      'selected_training_and_selection_seconds': selected['elapsed_seconds'],
                      'terminal_fit_seconds': fit['elapsed_seconds'], 'hard_cap_hit': fit['hard_cap_hit'],
                      'selected_at_final_update': fit['selected_at_final_update']})
        fits.append(fit)
    csv_save(out / 'aggregate.csv', aggregates)
    csv_save(out / 'paired_method_differences.csv', comparisons)
    csv_save(out / 'same_target_reveal.csv', revealed)
    csv_save(out / 'physical_episode_bootstrap.csv', intervals)
    csv_save(out / 'fit_costs.csv', costs)
    result = {'status': 'ANALYSIS_OF_INDEPENDENTLY_VERIFIED_CONDITIONAL_WORLD_RESULTS',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'formal_fits': len(fits),
              'formal_updates': sum(f['updates'] for f in fits), 'aggregate': aggregates, 'comparisons': comparisons,
              'same_target_reveal': revealed, 'physical_episode_bootstrap': intervals, 'fit_costs': costs,
              'sum_terminal_fit_seconds_not_gpu_hours': sum(f['elapsed_seconds'] for f in fits),
              'hard_cap_fits': [f['tag'] for f in fits if f['hard_cap_hit']],
              'selected_at_final_update': [f['tag'] for f in fits if f['selected_at_final_update']],
              'accepted_final_research_directions': 0, 'unique_physical_paths': 512, 'real_domains': 0,
              'source_summary_sha256': sha(evaluation / 'summary.json'), 'verification_sha256': sha(evaluation / 'verification.json'),
              'source_sha256': sha(pathlib.Path(__file__))}
    dump(out / 'analysis.json', result)
    report = ['# C49 completed descriptive analysis', '',
              f"Fits{len(fits)}, formal updates{result['formal_updates']}. Independent all-future/all-score NumPy verification passed.", '',
              'Physical energy uses full independent-pair scoring and RMS coordinate scaling. Neural rows average two training seeds; seed ranges are retained in aggregate.csv.', '',
              '| Process | Sensor density | Arm | Reveal bins | Physical future energy | Mean forecast MSE | Observed score |',
              '|---|---:|---|---:|---:|---:|---:|']
    for a in aggregates:
        report.append(f"| {a['case']} | {a['density']} | {a['arm']} | {a['reveal']} | {a['energy_mean']:.7f} | {a['mse_mean']:.7f} | {a['observation_score_mean']:.7f} |")
    report += ['', '## Evidence boundaries', '',
               'A density change also changes history information. A larger error under sparse sensing is not itself evidence that the dynamics model failed. Compare methods on the same histories and pair physical episodes.',
               'Observed scores omit a distribution-independent constant; negative values are valid. Compare methods within the same context. Relative percentages are not reported for these scores.',
               'Across-reveal comparisons use only identical final12 targets. Full24 and remaining12 horizon scores are different tasks.',
               'Episode-bootstrap intervals keep the three density views together. They condition on the two fitted training seeds and do not certify generalization across training seeds, process families or real domains.',
               'Both arms share predictive capacity/initializations/data schedules; the ELBO control has additional future-recognition parameters. This screen is not a complete matched-time frontier or external baseline study.',
               f"Hard caps: {result['hard_cap_fits']}. Final-update selections: {result['selected_at_final_update']}.",
               'A new framework contribution still requires a useful distinct interaction, current prior-art differentiation, competent controls and independent real-domain evidence.', '']
    (out / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    dump(out / 'MANIFEST.json', {'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()}, 'goal_complete': False})
    print(json.dumps({'status': result['status'], 'formal_updates': result['formal_updates'], 'hard_cap_fits': result['hard_cap_fits']}), flush=True)


if __name__ == '__main__':
    main()
