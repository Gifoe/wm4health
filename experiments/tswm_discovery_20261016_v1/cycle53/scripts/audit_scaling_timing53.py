"""Describe native clipping and real elapsed times after model selection seals."""
from native53 import *
import csv
import numpy as np
import pandas as pd

seal_path = R / 'outputs/SELECTION_SEAL_V6.json'
seal = json.loads(seal_path.read_text())
assert len(seal['selected_checkpoints']) == 40
for item in seal['selected_checkpoints']: assert sha(item['checkpoint']) == item['sha256']
OUT = ART / 'data_diagnostics_v1'; assert not OUT.exists(); OUT.mkdir()
from dataset_utils.greenhouse.dataset import GreenhouseDataset, TARGET_CHANNELS, ACTION_CHANNELS, EXOG_CHANNELS
teams = sorted(p.name for p in DATA.iterdir() if p.is_dir() and (p / 'GreenhouseClimate.xlsx').exists() and p.name != 'reference')
assert len(teams) == 5
statistics = []; timings = []; windows = []
for bank, root, subset, expected in [
    ('source_train', DATA, teams, 815), ('source_val', DATA, ['reference'], 163),
    ('new_crop_all', pathlib.Path(STORAGE['data_root']) / 'greenhouse3_original_v1/experiment_2/TimeSeries', None, 803)]:
    dataset = GreenhouseDataset(root, teams=subset, seq_len=272, stride=80); assert len(dataset) == expected
    collected = {(part, kind): [] for part in ['history', 'future'] for kind in ['target', 'action', 'exog']}
    elapsed = []; within_steps = []
    for index, (team, start) in enumerate(dataset._windows):
        frame = dataset._team_df[team].iloc[start:start + 272]
        dates = pd.to_datetime(frame.Date).to_numpy(dtype='datetime64[ns]')
        assert len(dates) == 272 and not np.isnat(dates).any()
        lag = (dates[256:] - dates[255]) / np.timedelta64(1, 'm')
        steps = np.diff(dates) / np.timedelta64(1, 'm')
        elapsed.append(lag); within_steps.append(steps)
        windows.append({'bank': bank, 'index': index, 'compartment': team, 'start_row': start,
            'first_forecast_time': str(dates[256]), 'last_forecast_time': str(dates[-1]),
            'horizon16_elapsed_minutes': float(lag[-1]),
            'nonfive_minute_history_and_future_steps': int(np.count_nonzero(steps != 5)),
            'nonpositive_steps': int(np.count_nonzero(steps <= 0))})
        for part, sl in [('history', slice(0, 256)), ('future', slice(256, 272))]:
            for kind, channels in [('target', TARGET_CHANNELS), ('action', ACTION_CHANNELS), ('exog', EXOG_CHANNELS)]:
                collected[(part, kind)].append(frame.iloc[sl][[c[0] for c in channels]].to_numpy(dtype=np.float64))
    elapsed = np.stack(elapsed); within_steps = np.concatenate(within_steps)
    for h in range(16):
        timings.append({'bank': bank, 'horizon_step': h + 1, 'nominal_minutes': (h + 1) * 5,
            'observed_min_minutes': float(elapsed[:, h].min()), 'observed_max_minutes': float(elapsed[:, h].max()),
            'observed_median_minutes': float(np.median(elapsed[:, h])),
            'fraction_exact_nominal_minutes': float(np.mean(elapsed[:, h] == (h + 1) * 5)),
            'windows': len(dataset)})
    for part in ['history', 'future']:
        for kind, channels in [('target', TARGET_CHANNELS), ('action', ACTION_CHANNELS), ('exog', EXOG_CHANNELS)]:
            values = np.concatenate(collected[(part, kind)])
            for col, (name, lo, hi) in enumerate(channels):
                vector = values[:, col]; finite = np.isfinite(vector); valid = vector[finite]
                stats = {'bank': bank, 'part': part, 'role': kind, 'channel': name,
                    'native_lo': lo, 'native_hi': hi, 'window_weighted_values': len(vector),
                    'finite_values': int(finite.sum()), 'nonfinite_fraction': float(np.mean(~finite)),
                    'finite_below_lo_fraction': float(np.mean(valid < lo)) if len(valid) else None,
                    'finite_above_hi_fraction': float(np.mean(valid > hi)) if len(valid) else None,
                    'raw_min': float(valid.min()) if len(valid) else None,
                    'raw_max': float(valid.max()) if len(valid) else None}
                for q in [1, 5, 50, 95, 99]: stats[f'raw_percentile_{q}'] = float(np.percentile(valid, q)) if len(valid) else None
                statistics.append(stats)

def csv_write(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
csv_write('native_scaling_coverage.csv', statistics)
csv_write('nominal_vs_actual_horizons.csv', timings)
csv_write('window_actual_times.csv', windows)
dump(OUT / 'summary.json', {'utc': now(), 'status': 'COMPLETE_DESCRIPTIVE_SCALING_AND_TIMING_AUDIT',
    'selected_models_fixed': 40, 'model_predictions_read': False, 'fits_or_hyperparameter_selection': False,
    'windows': len(windows), 'scaling_rows': len(statistics), 'horizon_rows': len(timings),
    'windows_with_nonfive_minute_steps': sum(w['nonfive_minute_history_and_future_steps'] > 0 for w in windows),
    'windows_with_nonpositive_steps': sum(w['nonpositive_steps'] > 0 for w in windows),
    'scope': 'Window-weighted raw-column values before native interpolation/clipping, and recorded date-field elapsed times. Descriptive input/target-definition diagnostics after the 40-model seal; no method selection.',
    'physical_metric_limit': 'Multiplying normalized forecast errors by native affine scales yields errors against native clipped target values in original column units. It does not recover accuracy against unclipped raw observations.',
    'selection_seal_sha256': sha(seal_path), 'script_sha256': sha(__file__), 'source_files_rechecked': check_source()})
dump(OUT / 'manifest.json', [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()])
print(json.dumps({'status': 'COMPLETE_DESCRIPTIVE_SCALING_AND_TIMING_AUDIT', 'windows': len(windows)}), flush=True)
