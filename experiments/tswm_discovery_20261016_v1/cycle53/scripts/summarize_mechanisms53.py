"""Expose native supervised versus held-out mechanism roles, including coverage."""
from native53 import *
import csv
import numpy as np

EVAL = ART / 'evaluation_v1'; OUT = ART / 'mechanism_summary_v1'
verification = json.loads((EVAL / 'verification.json').read_text())
assert verification['status'] == 'PASS_RETAINED_FORECAST_ARITHMETIC_AND_NATIVE_SCORE_DENOMINATORS'
assert verification['summary_sha256'] == sha(EVAL / 'summary.json')
assert not OUT.exists(); OUT.mkdir()
sys.path.insert(0, str(UP / 'RQ6_intervention/scripts'))
from RQ6_intervention.priors import priors_for
from prior_split import role, train_priors

prs = priors_for('greenhouse', tier='mechanistic')
assert len(prs) == 5
expected_train = {p.key for p in train_priors('greenhouse', 'default')}
assert len(expected_train) == 3
seal = json.loads((R / 'outputs/SELECTION_SEAL_V6.json').read_text())
soft_fits = 0
for selected in seal['selected_checkpoints']:
    if '__ae_gate_soft__' not in selected['tag']: continue
    fit = json.loads(pathlib.Path(selected['summary_path']).read_text())
    assert fit['prior_split'] == 'default' and float(fit['rho']) == 1.
    assert set(fit['constraints']) == expected_train
    soft_fits += 1
assert soft_fits == 10

rows = json.loads((EVAL / 'summary.json').read_text())
metrics = ['sign_agree_all', 'sign_agree_tol', 'sign_agree_active', 'sign_agree_filt',
           'sign_agree_q0.1', 'sign_agree_q-0.1', 'frac_active', 'frac_can_increase',
           'frac_up_is_down', 'delta_mean']
detail = []
for row in rows:
    if row['model'] == 'ordinary_control': continue
    native = json.loads((EVAL / f"{row['tag']}__{row['bank']}__native_mechanisms.json").read_text())
    for prior in prs:
        for name in metrics:
            key = f'{name}::{prior.key}'
            count = native['_native_score_counts'].get(key, 0)
            total = native['_native_score_totals'].get(key)
            if count: assert abs(native[key] - total / count) < 1e-12
            detail.append({'tag': row['tag'], 'model': row['model'], 'arm': row['arm'], 'seed': row['seed'],
                'bank': row['bank'], 'prior': prior.key, 'prior_role': role(prior.key, 'default'),
                'metric': name, 'native_numerator': total if count else None, 'selected_denominator': count,
                'all_windows': row['n'], 'value': total / count if count else None,
                'coverage': count / row['n'], 'status': 'DEFINED' if count else 'UNAVAILABLE'})
assert len(detail) == 4000

def csv_write(name, data):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
csv_write('per_seed_prior_role_and_coverage.csv', detail)
groups = []
for bank in ['source_val', 'new_crop_all']:
    for model in ['CrossLinear', 'PatchTST']:
        for arm in ['obs', 'ae_concat', 'ae_gate', 'ae_gate_soft']:
            for prior in prs:
                for metric in metrics:
                    group = [v for v in detail if v['bank'] == bank and v['model'] == model and
                             v['arm'] == arm and v['prior'] == prior.key and v['metric'] == metric]
                    assert {v['seed'] for v in group} == set(range(5))
                    defined = [v['value'] for v in group if v['value'] is not None]
                    denominators = [v['selected_denominator'] for v in group]
                    groups.append({'bank': bank, 'model': model, 'arm': arm, 'prior': prior.key,
                        'prior_role': role(prior.key, 'default'), 'metric': metric,
                        'defined_seeds': len(defined), 'seed_mean': float(np.mean(defined)) if defined else None,
                        'seed_std': float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
                        'selected_denominator_min': min(denominators), 'selected_denominator_max': max(denominators),
                        'coverage_min': min(v['coverage'] for v in group), 'coverage_max': max(v['coverage'] for v in group)})
assert len(groups) == 800
csv_write('seed_summary_by_prior_role.csv', groups)
dump(OUT / 'summary.json', {'utc': now(), 'status': 'COMPLETE_NATIVE_PRIOR_ROLE_AND_COVERAGE_SUMMARY',
    'native_soft_fits_checked': soft_fits, 'details': len(detail), 'groups': len(groups),
    'training_prior_keys': sorted(expected_train), 'heldout_prior_keys': [p.key for p in prs if role(p.key) == 'heldout_clean'],
    'qualification_scope': 'Training-prior roles are checked against all ten original soft-fit summaries; native totals/counts are retained. No fit or checkpoint selection.',
    'interpretation': 'Training-prior agreement is supervised constraint satisfaction. Held-out-prior agreement is generalization of declared sign constraints, not counterfactual accuracy or causal identification. Native empirical action shifts and MAD tolerances depend on bank. MAE under a changed input against factual targets is not intervention-truth error.',
    'verification_sha256': sha(EVAL / 'verification.json'), 'script_sha256': sha(__file__), 'source_files_rechecked': check_source()})
dump(OUT / 'manifest.json', [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()])
print(json.dumps({'status': 'COMPLETE_NATIVE_PRIOR_ROLE_AND_COVERAGE_SUMMARY', 'details': len(detail), 'groups': len(groups)}), flush=True)
