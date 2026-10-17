"""Execute pinned native preprocessing on a valid constructed forecast boundary."""
import datetime
import hashlib
import importlib.util
import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[1]
ROOT = R.parent
sys.path.insert(0, str(ROOT / 'cycle29/scripts'))
import runtime_profile
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    out = R / 'outputs/source_contract_tswm_v1'
    assert not out.exists()
    out.mkdir()
    source = ROOT / 'upstream/TSWM-Official-Repo'
    loader_path = source / 'dataset_utils/greenhouse/dataset.py'
    window_path = source / 'rq2_latent_vs_ob/common/windowing.py'
    pin = json.loads((source / 'SOURCE_PIN.json').read_text(encoding='utf-8'))
    assert pin['revision'] == '2e5e48540ff0da48e8192163969a6c58df1d5a3e'
    assert sha(loader_path) == 'f3f737d925df34e13da85eb81732e47ce9a2759f1dcd6a00d3d35295ceb9efc6'
    module = load_module('tswm_native_greenhouse_boundary49', loader_path)
    window = load_module('tswm_native_windowing_boundary49', window_path)
    h, f = 256, 16

    def fixture(future_value, missing):
        frame = pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=h + f, freq='5min')})
        target_constants = {'Tair': 15., 'Rhair': 60., 'CO2air': 800., 'PARin': 100.}
        for name, value in target_constants.items():
            frame[name] = value
        for name, low, high in module.ACTION_CHANNELS + module.EXOG_CHANNELS:
            frame[name] = (low + high) / 2
        frame.loc[h:, 'Tair'] = future_value
        if missing:
            frame.loc[h - 1, 'Tair'] = np.nan
        dataset = module.GreenhouseDataset.__new__(module.GreenhouseDataset)
        dataset.seq_len = h + f
        dataset.target_channels = module.TARGET_CHANNELS
        dataset.action_channels = module.ACTION_CHANNELS
        dataset.exog_channels = module.EXOG_CHANNELS
        dataset._tgt_names = [c[0] for c in module.TARGET_CHANNELS]
        dataset._act_names = [c[0] for c in module.ACTION_CHANNELS]
        dataset._exo_names = [c[0] for c in module.EXOG_CHANNELS]
        dataset._windows = [('fixture', 0)]
        dataset._team_df = {'fixture': frame}
        dataset._team_meta = {'fixture': {'team': 'constructed_contract_fixture'}}
        raw = frame[dataset._tgt_names].to_numpy(dtype='float32')
        check = np.concatenate((raw, frame[dataset._exo_names].to_numpy(dtype='float32')), 1)
        valid = module._window_continuous_ok(check, .1, 6)
        assert valid
        sample = dataset[0]
        schema = window.discover_schema(sample)
        result = window.unified_window(sample, h, f, schema)
        history_only = module._interp_nan_columns(raw[:h])
        history_only = module.GreenhouseDataset._norm(history_only, module.TARGET_CHANNELS, -1., 1.)
        return raw, result['target_history'], history_only, valid

    raw_a, a, safe_a, valid_a = fixture(10., True)
    raw_b, b, safe_b, valid_b = fixture(30., True)
    assert np.array_equal(raw_a[:h], raw_b[:h], equal_nan=True)
    raw_c, c, _, _ = fixture(10., False)
    raw_d, d, _, _ = fixture(30., False)
    delta = float(np.max(abs(a - b)))
    control = float(np.max(abs(c - d)))
    corrected = float(np.max(abs(safe_a - safe_b)))
    assert delta > .6 and control == 0 and corrected == 0
    assert sha(loader_path) == 'f3f737d925df34e13da85eb81732e47ce9a2759f1dcd6a00d3d35295ceb9efc6'
    np.savez_compressed(out / 'fixture.npz', raw_a=raw_a, raw_b=raw_b, history_a=a, history_b=b,
                        history_only_a=safe_a, history_only_b=safe_b)
    result = {'status': 'NATIVE_SOURCE_INPUT_CONTRACT_FAILURE_ON_CONSTRUCTED_VALID_WINDOW',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_repo': pin['repo'], 'source_revision': pin['revision'],
              'loader_sha256': sha(loader_path), 'windowing_sha256': sha(window_path),
              'native_executed_functions': ['GreenhouseDataset.__getitem__', '_interp_nan_columns', 'unified_window'],
              'context_length': h, 'forecast_horizon': f, 'both_windows_pass_native_missingness_filter': bool(valid_a and valid_b),
              'raw_history_identical': True, 'only_future_target_values_changed': True,
              'imputed_history_max_change': delta, 'no_missing_control_history_change': control,
              'history_only_imputation_history_change': corrected,
              'real_data_prevalence_measured': False, 'original_paper_headline_reproduced': False,
              'does_not_establish': ['new world-model mechanism', 'real-data effect size', 'benchmark ranking invalidity', 'causal treatment effect'],
              'upstream_sources_modified': False, 'source_sha256': sha(pathlib.Path(__file__)), 'formal_training_updates': 0}
    (out / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'AUDIT.md').write_text('# Pinned TSWM input-boundary contract\n\n'
        f"The unchanged native Greenhouse loader/windowing source at{pin['revision']} was executed on a constructed256+16 window accepted by its missingness filter. "
        'Two inputs had identical raw observed histories, including a single missing last-history Tair value; only future Tair changed. '
        f"The resulting normalized history changed by{delta:.7f}. With no missing boundary value, the difference was0. Imputing the history independently also gave0.\n\n"
        'This proves that the native full-window interpolation can violate the forecast input contract when missingness crosses the history/future boundary. '
        'It does not measure how often this occurs in real data or its effect on any reported result. Neither upstream code nor existing weights were changed. '
        'Before a faithful real-data baseline comparison, measure the prevalence from raw timestamps/masks, retain the original-protocol reproduction separately, '
        'and disclose any causal input-preparation correction. This is a preprocessing issue, not a new research direction.\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
