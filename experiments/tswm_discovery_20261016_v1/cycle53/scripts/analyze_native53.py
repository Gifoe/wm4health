"""Post-verification descriptive analysis; no fit or model selection occurs here."""
from native53 import *
import csv
import collections
import numpy as np
import pandas as pd
import torch

EVAL = ART / 'evaluation_v1'
OUT = ART / 'analysis_v1'
verified = json.loads((EVAL / 'verification.json').read_text())
assert verified['status'] == 'PASS_RETAINED_FORECAST_ARITHMETIC_AND_NATIVE_SCORE_DENOMINATORS'
assert verified['summary_sha256'] == sha(EVAL / 'summary.json')
assert verified['raw_manifest_sha256'] == sha(EVAL / 'raw_manifest.json')
assert not OUT.exists(), 'Preserve previous analysis and version changes'
assert check_source() == 125
OUT.mkdir(parents=True)
dump(OUT / 'OPENING_RECEIPT.json', {'utc': now(), 'pid': os.getpid(),
    'script_sha256': sha(__file__), 'analysis_supplement_sha256': sha(R / 'ANALYSIS_SUPPLEMENT.md'),
    'verification_sha256': sha(EVAL / 'verification.json'),
    'fit_or_selection_performed': False})

def write_csv(name, rows):
    assert rows
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)

rows = json.loads((EVAL / 'summary.json').read_text())
manifest = json.loads((EVAL / 'raw_manifest.json').read_text())
by_id = {(row['tag'], row['bank']): row for row in rows}
assert len(by_id) == len(rows) == len(manifest) == 84
errors = {}; per_window = {}; units = {}; input_banks = {}
for item in manifest:
    key = (item['tag'], item['bank']); path = EVAL / item['path']
    assert sha(path) == item['sha256']
    with np.load(path, allow_pickle=False) as z:
        err = z['prediction'].astype(np.float64) - z['target'].astype(np.float64)
    assert err.shape == (item['n'], 16, 4) and np.isfinite(err).all()
    errors[key] = err
    per_window[key] = {'MAE': np.abs(err).mean((1, 2)), 'MSE': (err ** 2).mean((1, 2))}
    assert abs(per_window[key]['MAE'].mean() - by_id[key]['MAE']) < 1e-12

from dataset_utils.greenhouse.dataset import GreenhouseDataset, TARGET_CHANNELS
target_names = [v[0] for v in TARGET_CHANNELS]
physical_scale = np.array([(v[2] - v[1]) / 2 for v in TARGET_CHANNELS])
identity_rows = []
for bank in ['source_val', 'new_crop_all']:
    inputs = json.loads((EVAL / 'input_manifest.json').read_text())[bank]
    assert sha(EVAL / inputs['path']) == inputs['sha256']
    loaded = torch.load(EVAL / inputs['path'], map_location='cpu', weights_only=False)
    input_banks[bank] = loaded
    root = DATA if bank == 'source_val' else pathlib.Path(STORAGE['data_root']) / 'greenhouse3_original_v1/experiment_2/TimeSeries'
    raw = GreenhouseDataset(root, teams=['reference'] if bank == 'source_val' else None, seq_len=272, stride=80)
    assert len(raw) == loaded['n']
    assert [(str(t), int(s)) for t, s in raw._windows] == list(zip(loaded['unit_ids'], loaded['starts']))
    dates = np.array([pd.Timestamp(raw._team_df[team].iloc[int(start) + 256]['Date']).to_datetime64()
                      for team, start in raw._windows], dtype='datetime64[ns]')
    assert not np.isnat(dates).any()
    first_day = dates.min().astype('datetime64[D]')
    days = (dates.astype('datetime64[D]') - first_day).astype(np.int64)
    blocks = {d: days // d for d in [3, 7]}
    teams = np.array(loaded['unit_ids'])
    units[bank] = {'teams': teams, 'dates': dates, 'blocks': blocks}
    for i, (team, start) in enumerate(raw._windows):
        identity_rows.append({'bank': bank, 'window': i, 'compartment': team, 'start_row': int(start),
            'first_forecast_time': str(dates[i]), 'calendar_block_7days': int(blocks[7][i]),
            'calendar_block_3days': int(blocks[3][i])})
write_csv('window_identities.csv', identity_rows)

fit_rows = []; seal = json.loads((R / 'outputs/SELECTION_SEAL_V6.json').read_text())
assert len(seal['selected_checkpoints']) == 40
for item in seal['selected_checkpoints']:
    assert sha(item['checkpoint']) == item['sha256']
    fit = json.loads(pathlib.Path(item['summary_path']).read_text())
    curve = json.loads((pathlib.Path(item['summary_path']).parent / 'curve.json').read_text())
    assert len(curve) == fit['epochs_run']
    fit_rows.append({k: fit.get(k) for k in ['tag', 'arm_name', 'best_epoch', 'epochs_run', 'hit_epoch_cap',
        'selected_final_epoch', 'formal_updates', 'training_params', 'native_fit_seconds',
        'total_instrumented_seconds', 'peak_cuda_allocated_bytes', 'runtime_storage_mode', 'checkpoint_sha256']})
write_csv('fit_diagnostics.csv', fit_rows)

models = ['CrossLinear', 'PatchTST']; arms = ['obs', 'ae_concat', 'ae_gate', 'ae_gate_soft']
flat = []; team_rows = []; aggregate = []
for row in rows:
    tag, bank = row['tag'], row['bank']; error = errors[(tag, bank)]
    for channel, name in enumerate(target_names):
        for horizon in range(16):
            channel_error = error[:, horizon, channel]
            flat.append({'tag': tag, 'model': row['model'], 'arm': row['arm'], 'seed': row['seed'],
                'bank': bank, 'channel': name, 'horizon_step': horizon + 1, 'nominal_horizon_minutes': 5 * (horizon + 1),
                'n': len(error), 'MAE': float(np.abs(channel_error).mean()),
                'MSE': float((channel_error ** 2).mean()),
                'physical_MAE': float(np.abs(channel_error).mean() * physical_scale[channel]),
                'physical_MSE': float((channel_error ** 2).mean() * physical_scale[channel] ** 2)})
    for team in sorted(set(units[bank]['teams'])):
        mask = units[bank]['teams'] == team
        team_rows.append({'tag': tag, 'model': row['model'], 'arm': row['arm'], 'seed': row['seed'],
            'bank': bank, 'compartment': team, 'n': int(mask.sum()),
            'MAE': float(np.abs(error[mask]).mean()), 'MSE': float((error[mask] ** 2).mean())})
for bank in units:
    for model in models:
        for arm in arms:
            group = [by_id[(f'{model}__{arm}__seed{s}', bank)] for s in range(5)]
            aggregate.append({'bank': bank, 'model': model, 'arm': arm, 'n_seeds': 5, 'n_windows': group[0]['n'],
                **{f'{metric}_{stat}': float(getattr(np, stat)([g[metric] for g in group], **({'ddof': 1} if stat == 'std' else {})))
                   for metric in ['MAE', 'MSE'] for stat in ['mean', 'std']}})
    for control in ['persistence', 'ridge']:
        row = by_id[(control, bank)]
        aggregate.append({'bank': bank, 'model': 'ordinary_control', 'arm': control, 'n_seeds': 1,
            'n_windows': row['n'], 'MAE_mean': row['MAE'], 'MSE_mean': row['MSE'], 'MAE_std': None, 'MSE_std': None})
write_csv('all_channel_horizons.csv', flat); write_csv('all_compartments.csv', team_rows)
write_csv('aggregate_metrics.csv', aggregate)

def paired_summary(baseline, candidate, block_ids, seed=5307):
    # Seeds represent paired training variability; all physical data stay fixed.
    assert baseline.shape == candidate.shape and baseline.ndim == 2 and baseline.shape[0] == 5
    base = baseline.mean(0); delta = (baseline - candidate).mean(0)
    unique = np.unique(block_ids); assert len(unique) >= 2
    numer = np.array([delta[block_ids == b].sum() for b in unique])
    denom = np.array([base[block_ids == b].sum() for b in unique])
    counts = np.array([(block_ids == b).sum() for b in unique])
    rng = np.random.default_rng(seed); draw = rng.integers(0, len(unique), size=(2000, len(unique)))
    absolute = numer[draw].sum(1) / counts[draw].sum(1)
    relative = 100 * numer[draw].sum(1) / denom[draw].sum(1)
    seed_gain = (baseline - candidate).mean(1)
    return {'MAE_reduction': float(delta.mean()), 'MAE_reduction_seed_values': seed_gain.tolist(),
        'MAE_reduction_seed_std': float(seed_gain.std(ddof=1)),
        'percent_MAE_reduction': float(100 * delta.sum() / base.sum()), 'calendar_bins': len(unique),
        'descriptive_MAE_reduction_interval': np.percentile(absolute, [2.5, 97.5]).tolist(),
        'descriptive_percent_reduction_interval': np.percentile(relative, [2.5, 97.5]).tolist(),
        'bootstrap_draws': 2000, 'bootstrap_seed': seed,
        'positive_means': 'candidate has smaller MAE; fixed crop/compartments, not domain-population uncertainty'}

contrasts = []; contrast_csv = []
for bank in units:
    for model in models:
        comparisons = [('obs', 'ae_concat'), ('ae_concat', 'ae_gate'), ('ae_gate', 'ae_gate_soft')]
        comparisons += [(control, arm) for control in ['persistence', 'ridge'] for arm in arms]
        for baseline_arm, candidate_arm in comparisons:
            def series(arm):
                return np.stack([per_window[(arm if arm in ['persistence', 'ridge'] else f'{model}__{arm}__seed{s}', bank)]['MAE'] for s in range(5)])
            base = series(baseline_arm); candidate = series(candidate_arm)
            for d in [7, 3]:
                item = {'bank': bank, 'model': model, 'baseline': baseline_arm, 'candidate': candidate_arm,
                    'calendar_block_days': d, **paired_summary(base, candidate, units[bank]['blocks'][d])}
                contrasts.append(item)
                contrast_csv.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in item.items()})
assert len(contrasts) == 88
dump(OUT / 'paired_contrasts.json', contrasts); write_csv('paired_contrasts.csv', contrast_csv)

# Include unobserved/zero-selected denominator entries, rather than silently dropping them.
sys.path.insert(0, str(UP / 'RQ6_intervention/scripts'))
from RQ6_intervention.priors import priors_for
prior_keys = [p.key for p in priors_for('greenhouse', tier='mechanistic')]
required = [f'{name}::{key}' for key in prior_keys for name in
            ['sign_agree_all', 'sign_agree_tol', 'sign_agree_active', 'sign_agree_filt',
             'sign_agree_q0.1', 'sign_agree_q-0.1', 'frac_active', 'frac_can_increase', 'delta_mean']]
mechanism_rows = []
for row in rows:
    if row['model'] == 'ordinary_control': continue
    native = json.loads((EVAL / f"{row['tag']}__{row['bank']}__native_mechanisms.json").read_text())
    counts, totals = native['_native_score_counts'], native['_native_score_totals']
    for key in sorted(set(counts) | set(required)):
        count = counts.get(key, 0); total = totals.get(key, 0)
        assert count >= 0
        if count: assert abs(native[key] - total / count) < 1e-12
        mechanism_rows.append({'tag': row['tag'], 'bank': row['bank'], 'model': row['model'],
            'arm': row['arm'], 'seed': row['seed'], 'metric': key, 'numerator': total if count else None,
            'denominator': count, 'value': total / count if count else None,
            'status': 'DEFINED' if count else 'UNAVAILABLE_NO_SELECTED_DENOMINATOR',
            'scope': 'native input-shift/sign-prior diagnostic; no intervention truth'})
write_csv('native_mechanism_counts.csv', mechanism_rows)

summary = {'utc': now(), 'status': 'DESCRIPTIVE_ANALYSIS_COMPLETE_NO_RESEARCH_DIRECTION_ACCEPTED',
    'accepted_directions': 0, 'groups': len(rows), 'models': 40, 'channel_horizon_rows': len(flat),
    'compartment_rows': len(team_rows), 'paired_contrasts': len(contrasts), 'fit_diagnostics': len(fit_rows),
    'calendar_bins': {bank: {str(d): int(len(np.unique(v))) for d, v in info['blocks'].items()} for bank, info in units.items()},
    'limitations': ['one crop-pair and repeated compartments', 'factual future schedules supplied',
        'training seeds are not systems', 'native sign-prior scores are not counterfactual truth',
        'PatchTST AE-concat uses qualified microbatch adaptation', 'no new method or independent-domain confirmation'],
    'formal_updates': sum(int(row['formal_updates']) for row in fit_rows),
    'native_fit_seconds': sum(float(row['native_fit_seconds']) for row in fit_rows),
    'hit_epoch_cap': sum(bool(row['hit_epoch_cap']) for row in fit_rows),
    'verification_sha256': sha(EVAL / 'verification.json'), 'selection_seal_sha256': sha(R / 'outputs/SELECTION_SEAL_V6.json'),
    'analysis_supplement_sha256': sha(R / 'ANALYSIS_SUPPLEMENT.md'), 'script_sha256': sha(__file__),
    'aggregate_metrics': aggregate}
dump(OUT / 'summary.json', summary)
files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
dump(OUT / 'manifest.json', files)
print(json.dumps({k: v for k, v in summary.items() if k not in ['aggregate_metrics']}), flush=True)
