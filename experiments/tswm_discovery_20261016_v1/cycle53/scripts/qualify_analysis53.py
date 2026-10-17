"""Qualify paired arithmetic and time-bin identity without reading model forecasts."""
from native53 import *
import ast
import numpy as np
import pandas as pd

OUT = R / 'outputs/analysis_qualification_v1'
assert not OUT.exists(); OUT.mkdir()
source = R / 'scripts/analyze_native53.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
definition = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'paired_summary']
assert len(definition) == 1
namespace = {'np': np}
exec(compile(ast.Module(body=definition, type_ignores=[]), str(source), 'exec'), namespace)
summary = namespace['paired_summary']
base = np.tile(np.linspace(.2, .8, 24), (5, 1)); blocks = np.repeat(np.arange(6), 4)
identical = summary(base, base.copy(), blocks)
assert identical['MAE_reduction'] == identical['percent_MAE_reduction'] == 0.
assert identical['descriptive_MAE_reduction_interval'] == [0., 0.]
proportional = summary(base, base * .8, blocks)
assert abs(proportional['percent_MAE_reduction'] - 20) < 1e-12
assert np.max(np.abs(np.array(proportional['descriptive_percent_reduction_interval']) - 20)) < 1e-12
bad = summary(base, base * 1.1, blocks)
assert bad['MAE_reduction'] < 0 and abs(bad['percent_MAE_reduction'] + 10) < 1e-12
assert summary(base, base * .8, blocks) == proportional

from dataset_utils.greenhouse.dataset import GreenhouseDataset
layouts = {}
for name, root, teams, expected in [
    ('source_val', DATA, ['reference'], 163),
    ('new_crop_all', pathlib.Path(STORAGE['data_root']) / 'greenhouse3_original_v1/experiment_2/TimeSeries', None, 803)]:
    raw = GreenhouseDataset(root, teams=teams, seq_len=272, stride=80)
    assert len(raw) == expected
    dates = np.array([pd.Timestamp(raw._team_df[t].iloc[int(s) + 256]['Date']).to_datetime64()
                      for t, s in raw._windows], dtype='datetime64[ns]')
    assert not np.isnat(dates).any()
    day = dates.astype('datetime64[D]'); relative = (day - day.min()).astype(np.int64)
    layouts[name] = {'n': len(raw), 'compartments': len(set(t for t, s in raw._windows)),
        'calendar_bins': {str(d): int(len(np.unique(relative // d))) for d in [3, 7]},
        'first_forecast_date': str(day.min()), 'last_forecast_date': str(day.max())}
result = {'utc': now(), 'status': 'PASS_PAIRED_RATIO_NULL_DIRECTION_AND_CALENDAR_IDENTITIES',
    'model_predictions_read': False, 'test_scores_read': False, 'training_updates': 0,
    'source_files_rechecked': check_source(), 'analysis_script_sha256': sha(source),
    'analysis_supplement_sha256': sha(R / 'ANALYSIS_SUPPLEMENT.md'), 'calendar_metadata': layouts,
    'tests': ['identical predictions null', '20 percent proportional gain', '10 percent degradation',
              'deterministic resampling', 'native date identities and accepted counts']}
dump(OUT / 'qualification.json', result); print(json.dumps(result), flush=True)
