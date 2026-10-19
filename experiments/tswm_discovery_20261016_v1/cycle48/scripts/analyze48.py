"""Descriptive analysis after locked scoring and independent verification.

No fitting, model selection, test-driven tuning, or inferential iid claims.
"""
import csv
import datetime
import hashlib
import json
import pathlib
import statistics

R = pathlib.Path(__file__).resolve().parents[1]
OUT = R / 'outputs' / 'analysis_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def csv_save(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    evaluation = R / 'outputs' / 'eval_v1'
    verification = read(evaluation / 'verification.json')
    assert verification['status'] == 'PASS_REAL_PROCESS_MEMORY_SCREEN'
    assert verification['summary_sha256'] == sha(evaluation / 'summary.json')
    assert verification['raw_manifest_sha256'] == sha(evaluation / 'raw_manifest.json')
    summary = read(evaluation / 'summary.json')
    rows = read(evaluation / 'all_rows.json')
    assert summary['all_rows_sha256'] == sha(evaluation / 'all_rows.json')
    assert len(rows) == 1694 and len(summary['groups']) == 44
    assert not OUT.exists()
    OUT.mkdir()
    groups = summary['groups']
    lookup = {(r['case'], r['method'], r['seed'], r['reveal'], r['origin']): r for r in rows}
    aggregate = []
    for case in ['ett', 'jena']:
        for method in ['node', 'fixed_dde', 'learned_dde', 'gru', 'ridge', 'persistence', 'seasonal24']:
            for reveal in [0, 24]:
                selected = [g for g in groups if g['case'] == case and g['method'] == method and g['reveal'] == reveal]
                entry = {'case': case, 'method': method, 'reveal': reveal,
                         'horizon': 48 - reveal, 'training_seeds': len(selected),
                         'unique_origins': selected[0]['origins']}
                for metric in ['mse', 'mae', 'tail24_mse', 'aggregate24_mse', 'aggregate_all_mse']:
                    values = [g[metric] for g in selected]
                    entry.update({metric + '_mean': statistics.mean(values),
                                  metric + '_min_seed': min(values), metric + '_max_seed': max(values)})
                aggregate.append(entry)
    csv_save(OUT / 'aggregate.csv', aggregate)
    comparisons = []
    for case in ['ett', 'jena']:
        for left, right in [('learned_dde', 'fixed_dde'), ('learned_dde', 'gru'),
                            ('learned_dde', 'ridge'), ('fixed_dde', 'gru'),
                            ('fixed_dde', 'ridge'), ('gru', 'ridge')]:
            for seed in [2026481, 2026482]:
                for reveal in [0, 24]:
                    for metric in ['mse', 'tail24_mse']:
                        left_rows = [r for r in rows if r['case'] == case and r['method'] == left and r['seed'] == seed and r['reveal'] == reveal]
                        right_seed = None if right == 'ridge' else seed
                        differences, denominators = [], []
                        for lrow in left_rows:
                            rrow = lookup[(case, right, right_seed, reveal, lrow['origin'])]
                            differences.append(lrow[metric] - rrow[metric])
                            denominators.append(rrow[metric])
                        comparisons.append({'case': case, 'left': left, 'right': right,
                                            'seed': seed, 'reveal': reveal, 'metric': metric,
                                            'origins': len(differences),
                                            'mean_left_minus_right': statistics.mean(differences),
                                            'relative_difference_pct': 100 * statistics.mean(differences) / statistics.mean(denominators),
                                            'fraction_origins_left_better': sum(d < 0 for d in differences) / len(differences)})
    csv_save(OUT / 'paired_model_differences.csv', comparisons)
    reveals = []
    for group in [g for g in groups if g['reveal'] == 24]:
        original = next(g for g in groups if g['model'] == group['model'] and g['reveal'] == 0)
        differences = [r['tail24_mse'] - lookup[(r['case'], r['method'], r['seed'], 0, r['origin'])]['tail24_mse']
                       for r in rows if r['model'] == group['model'] and r['reveal'] == 24]
        reveals.append({'case': group['case'], 'method': group['method'], 'seed': group['seed'],
                        'same_target': 'original future hours 25 through 48',
                        'unrevealed_tail24_mse': original['tail24_mse'],
                        'revealed_tail24_mse': group['tail24_mse'],
                        'revealed_minus_unrevealed': statistics.mean(differences),
                        'relative_difference_pct': 100 * statistics.mean(differences) / original['tail24_mse'],
                        'fraction_origins_reveal_better': sum(d < 0 for d in differences) / len(differences)})
    csv_save(OUT / 'same_target_reveal.csv', reveals)
    fits = [read(path) for path in sorted((R / 'outputs' / 'fits').glob('*/fit_summary.json'))]
    assert len(fits) == 16
    diagnostic = read(R / 'outputs' / 'diagnostics_v1' / 'numerical_refinement.json')
    flagged = [d['model'] for d in diagnostic if d.get('numerical_error_exceeds_5pct_forecast_RMSE')]
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'DESCRIPTIVE_ANALYSIS_OF_INDEPENDENTLY_VERIFIED_RESULTS',
              'formal_fits': len(fits), 'formal_updates': sum(f['updates'] for f in fits),
              'sum_fit_wall_seconds_not_elapsed_or_gpu_hours': sum(f['elapsed_seconds'] for f in fits),
              'selected_at_final_update': [f['tag'] for f in fits if f['selected_at_final_update']],
              'hard_cap_hit': [f['tag'] for f in fits if f['hard_cap_hit']],
              'numerical_flags': flagged, 'aggregate': aggregate,
              'paired_model_differences': comparisons, 'same_target_reveal': reveals,
              'source_summary_sha256': sha(evaluation / 'summary.json'),
              'verification_sha256': sha(evaluation / 'verification.json'),
              'source_sha256': sha(pathlib.Path(__file__)),
              'accepted_final_research_directions': 0,
              'scope': 'Two real processes; existing deterministic mean models; dependent chronological origins. No novel method or stochastic-world-law validation.'}
    dump(OUT / 'analysis.json', result)
    report = ['# C48 independently verified descriptive results', '',
              f"Formal fits: {len(fits)}; updates: {result['formal_updates']}; numerical flags: {len(flagged)}.", '',
              'Scores use TRAIN-standardized targets. Seed ranges describe two optimization seeds, not uncertainty across independent worlds.', '',
              '| Process | Arm | Reveal hours | MSE seed mean | Min seed | Max seed | Tail24 MSE mean |',
              '|---|---|---:|---:|---:|---:|---:|']
    for a in aggregate:
        report.append(f"| {a['case']} | {a['method']} | {a['reveal']} | {a['mse_mean']:.7f} | {a['mse_min_seed']:.7f} | {a['mse_max_seed']:.7f} | {a['tail24_mse_mean']:.7f} |")
    report += ['', '## Real-observation continuation on identical target hours', '',
               'Compare only original future hours25-48. Negative differences favor incorporating the true first24-hour prefix.', '',
               '| Process | Arm | Seed | Unrevealed tail MSE | Revealed tail MSE | Change % |',
               '|---|---|---:|---:|---:|---:|']
    for a in reveals:
        report.append(f"| {a['case']} | {a['method']} | {a['seed']} | {a['unrevealed_tail24_mse']:.7f} | {a['revealed_tail24_mse']:.7f} | {a['relative_difference_pct']:+.2f} |")
    report += ['', '## Attribution limits', '',
               f"Hard-cap fits: {result['hard_cap_hit']}. Final-step selections: {result['selected_at_final_update']}.",
               f'VAL numerical-refinement flags: {flagged}. Inspect individual diagnostics before attributing differences to memory.',
               'Paired per-origin differences and win fractions are descriptive; overlapping histories and temporal dependence preclude treating all origins as iid replicates.',
               'The scalar NODE is a weak ablation. Learned delay versus fixed delay shares initialization and schedules; GRU/ridge are stronger history-aware controls.',
               'An existing delay method winning supports its utility under this protocol. It does not establish a novel framework. A weak arm losing does not establish a research gap.',
               'A concrete new learning interaction and independent confirmatory evidence remain required for the full discovery objective.', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    dump(OUT / 'MANIFEST.json', {'utc': result['utc'], 'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
                                'goal_complete': False, 'source_sha256': sha(pathlib.Path(__file__))})
    print(json.dumps({k: result[k] for k in ['status', 'formal_fits', 'formal_updates', 'numerical_flags', 'hard_cap_hit']}), flush=True)


if __name__ == '__main__':
    main()
