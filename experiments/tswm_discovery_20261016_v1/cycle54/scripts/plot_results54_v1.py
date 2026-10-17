"""Render all frozen C54 outcomes; qualification uses only explicit dummy values."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse, datetime, hashlib, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
TASKS = ['linear_small', 'linear_medium']
HORIZONS = [0, 1, 5, 10, 25, 50]
METHODS = ['rp', 'rp_emtrue', 'em']
LABELS = ['Native False', 'Existing True', 'Corrected EM']
COLORS = dict(zip(METHODS, ['#0072B2', '#D55E00', '#222222']))
MARKERS = ['o', '^', 's']
STYLES = ['-', '--', ':']

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))

def dump(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def checked_error_scale(ax, values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        ax.set_ylim(0, 1)
        ax.text(.5, .5, 'No defined ratios', ha='center', va='center', transform=ax.transAxes)
        return 'undefined'
    assert np.min(a) >= 0
    scale = 'log' if np.min(a) > 0 else 'linear'
    ax.set_yscale(scale)
    if scale == 'linear':
        ax.set_ylim(bottom=0)
    return scale

def render(primary, extra, risk, target, dummy=False):
    target = pathlib.Path(target)
    assert not target.exists(), 'Preserve existing renders; version any changed procedure'
    target.mkdir(parents=True)
    records = primary['records'] + extra['records']
    assert len(records) == 16 and len({r['run'] for r in records}) == 16
    assert len(risk['records']) == 96
    assert {r['h'] for r in risk['records']} == set(HORIZONS)
    for task in TASKS:
        subset = [r for r in records if r['task'] == task]
        assert sorted((m, sum(r['method'] == m for r in subset)) for m in METHODS) == sorted(zip(METHODS, [3, 2, 3]))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'normal', 'savefig.facecolor': 'white', 'svg.hashsalt': 'c54-results-v1'})
    plotted, outputs, scales, geometry = [], [], [], []

    def finish(fig, stem, note, handles=None):
        if handles:
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .975), ncol=len(handles), frameon=False)
        fig.text(.5, .025, note, ha='center', va='bottom', fontsize=9)
        if dummy:
            fig.suptitle('Dummy fixture: no experiment results', y=.999, color='#666666', fontsize=10)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        for ax in fig.axes:
            box = ax.get_tightbbox(renderer)
            assert box.x0 >= canvas.x0 - 1 and box.y0 >= canvas.y0 - 1 and box.x1 <= canvas.x1 + 1 and box.y1 <= canvas.y1 + 1, (stem, box.bounds, canvas.bounds)
        geometry.append({'figure': stem, 'axes_tight_boxes_within_canvas': True, 'visual_inspection_still_required': True})
        for extension in ['svg', 'png']:
            p = target / (stem + '.' + extension)
            fig.savefig(p, dpi=190)
            outputs.append({'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        plt.close(fig)

    legend = [Line2D([0], [0], color=COLORS[m], marker='o', linestyle='-', label=label) for m, label in zip(METHODS, LABELS)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    fig.subplots_adjust(left=.085, right=.98, bottom=.15, top=.86, wspace=.42, hspace=.48)
    metrics = [('full_smoothing_R2_train_readout', 'Retrospective state R²'), ('transition_eigenvalue_mean_absolute_error', 'Transition eigenvalue error'), ('conditional_mean_discrepancy', 'h10 conditional-mean error')]
    for i, task in enumerate(TASKS):
        for j, (key, label) in enumerate(metrics):
            ax = axes[i, j]; values = []
            for x, method in enumerate(METHODS):
                subset = sorted([r for r in records if r['task'] == task and r['method'] == method], key=lambda r: r['seed'])
                for r in subset:
                    y = r[key] if j < 2 else next(h[key] for h in r['horizons'] if h['h'] == 10)
                    assert np.isfinite(y)
                    ax.scatter(x + (r['seed'] - 1) * .09, y, marker=MARKERS[r['seed']], s=45, color=COLORS[method], edgecolor='white', linewidth=.4, zorder=3)
                    values.append(y)
                    plotted.append({'figure': 'recovery_and_dynamics', 'panel': [i, j], 'run': r['run'], 'field': key, 'h': 10 if j == 2 else None, 'value': float(y)})
            ax.set_xticks(range(3), ['False', 'True', 'EM'])
            ax.set_xlim(-.4, 2.4); ax.set_ylabel(label); ax.set_title(task.replace('_', ' ')); ax.grid(axis='y', alpha=.2)
            if j:
                scales.append({'figure': 'recovery_and_dynamics', 'panel': [i, j], 'scale': checked_error_scale(ax, values)})
    finish(fig, 'recovery_and_dynamics', 'All 16 fits. Circle / triangle / square: initialization 0 / 1 / 2. One data draw per task.\nState maps use training labels only; smoothing uses full sequences, forecasts use the first 50 observations.', legend)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.1))
    fig.subplots_adjust(left=.095, right=.98, bottom=.16, top=.87, wspace=.34, hspace=.44)
    for i, task in enumerate(TASKS):
        for j, field in enumerate(['excess_conditional', 'normalized_excess_conditional']):
            ax = axes[i, j]; values = []; undefined = 0
            for r in [r for r in records if r['task'] == task]:
                rows = sorted([q for q in risk['records'] if q['run'] == r['run']], key=lambda q: q['h'])
                assert [q['h'] for q in rows] == HORIZONS
                ys = [q[field] if q[field] is not None else np.nan for q in rows]
                undefined += sum(q[field] is None for q in rows)
                values.extend(ys)
                ax.plot(range(len(HORIZONS)), ys, color=COLORS[r['method']], marker=MARKERS[r['seed']], linestyle=STYLES[r['seed']], linewidth=1.15, markersize=4, alpha=.8)
                for q in rows:
                    plotted.append({'figure': 'causal_risk', 'panel': [i, j], 'run': r['run'], 'field': field, 'h': q['h'], 'value': q[field]})
            scale = checked_error_scale(ax, values)
            scales.append({'figure': 'causal_risk', 'panel': [i, j], 'scale': scale, 'undefined_values': undefined})
            ax.set_title(task.replace('_', ' ')); ax.set_xlabel('Evaluated horizon (equally spaced)'); ax.set_xticks(range(len(HORIZONS)), HORIZONS)
            ax.set_ylabel('Conditional-mean discrepancy' if j == 0 else 'Discrepancy / predictable energy')
            ax.grid(alpha=.2)
            if undefined:
                ax.text(.02, .02, f'{undefined} undefined ratios (zero energy)', transform=ax.transAxes, fontsize=8)
    finish(fig, 'causal_risk', 'Every frozen fit at all six horizons; no seed selection. Raw errors accompany normalized errors.\nEnergy is oracle conditional-mean deviation from the unconditional prior mean; near-zero energy can amplify ratios.', legend)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(left=.095, right=.98, bottom=.24, top=.81, wspace=.35)
    paired_legend = [Line2D([0], [0], color=COLORS[METHODS[s]], marker=MARKERS[s], linestyle=STYLES[s], label='Initialization ' + str(s)) for s in [0, 1]]
    for i, task in enumerate(TASKS):
        ax = axes[i]
        for seed in [0, 1]:
            c = next(c for c in risk['comparisons'] if c['task'] == task and c['comparison'] == 'existing True minus matched False, seed' + str(seed))
            hs = c['horizons']; assert [h['h'] for h in hs] == HORIZONS
            ys = [h['delta_mean'] for h in hs]; lo = [h['pointwise95_interval'][0] for h in hs]; hi = [h['pointwise95_interval'][1] for h in hs]
            color = COLORS[METHODS[seed]]
            ax.plot(range(len(HORIZONS)), ys, color=color, marker=MARKERS[seed], linestyle=STYLES[seed], markersize=4)
            ax.fill_between(range(len(HORIZONS)), lo, hi, color=color, alpha=.12)
            for h in hs:
                plotted.append({'figure': 'paired_source_option', 'panel': [i], 'task': task, 'seed': seed, 'h': h['h'], 'value': h['delta_mean'], 'pointwise95_interval': h['pointwise95_interval']})
        ax.axhline(0, color='#666666', linewidth=.8); ax.set_title(task.replace('_', ' ')); ax.set_xlabel('Evaluated horizon (equally spaced)'); ax.set_xticks(range(len(HORIZONS)), HORIZONS)
        ax.set_ylabel('Conditional-error delta (True − False)'); ax.grid(alpha=.2)
    finish(fig, 'paired_source_option', 'Negative favors the existing True option. Shading: pointwise 95% bootstrap intervals over heldout sequences.\nFixed fits; these intervals do not cover initialization or domain uncertainty and have no multiplicity correction.', paired_legend)
    dump(target / 'plotted_values.json', {'dummy_only': dummy, 'points': plotted, 'point_count': len(plotted), 'axis_scales': scales})
    dump(target / 'render_geometry.json', geometry)
    return outputs

def dummy_inputs():
    primary, extra, risk = {'records': []}, {'records': []}, {'records': [], 'comparisons': []}
    for ti, task in enumerate(TASKS):
        for mi, method in enumerate(METHODS):
            for seed in range(2 if method == 'rp_emtrue' else 3):
                run = method + '_' + task + '_seed' + str(seed)
                horizons = [{'h': h, 'conditional_mean_discrepancy': (.2 + ti + mi / 3 + seed / 10) / (1 + h)} for h in HORIZONS]
                row = {'run': run, 'method': method, 'task': task, 'seed': seed, 'full_smoothing_R2_train_readout': .6 + mi / 15 + seed / 100, 'transition_eigenvalue_mean_absolute_error': .03 + mi / 30 + seed / 100, 'horizons': horizons}
                (extra if method == 'rp_emtrue' else primary)['records'].append(row)
                for h in horizons:
                    risk['records'].append({'run': run, 'h': h['h'], 'excess_conditional': h['conditional_mean_discrepancy'], 'normalized_excess_conditional': h['conditional_mean_discrepancy'] * (1 + h['h'] / 5)})
        for seed in [0, 1]:
            risk['comparisons'].append({'task': task, 'comparison': 'existing True minus matched False, seed' + str(seed), 'horizons': [{'h': h, 'delta_mean': (-.05 + seed / 50) / (1 + h), 'pointwise95_interval': [(-.08 + seed / 50) / (1 + h), (-.02 + seed / 50) / (1 + h)]} for h in HORIZONS]})
    return primary, extra, risk

def main():
    args = argparse.ArgumentParser(); args.add_argument('--qualify', action='store_true'); qualify = args.parse_args().qualify
    if qualify:
        inputs = {}; data = dummy_inputs(); target = OUT / 'visual_qualification_v1'
    else:
        assert read(OUT / 'postprocess54_v3.json')['status'] == 'COMPLETE_PRIMARY12_AND_FOUR_MATCHED_EXISTING_EM_OPTION_CONTROLS'
        paths = [OUT / 'evaluation54.json', OUT / 'em_option_learning_v1/evaluation/summary.json', OUT / 'causal_risk_interpretation_v1/summary.json']
        inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
        data = tuple(read(p) for p in paths)
        assert data[1]['all_16_models_and_16_readouts_frozen_before_first_heldout_score'] is True
        assert data[2]['status'] == 'COMPLETE_READ_ONLY_CAUSAL_RISK_INTERPRETATION_OF_ALL16_FROZEN_FITS'
        assert data[2]['model_or_hyperparameter_selection'] is False
        target = OUT / 'result_figures_v1'
    outputs = render(*data, target, dummy=qualify)
    for rel, value in inputs.items():
        assert sha(ROOT / rel) == value
    manifest = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'DUMMY_LAYOUT_EXECUTED_NO_FORMAL_RESULTS' if qualify else 'ALL16_FROZEN_FITS_RENDERED_PENDING_ACTUAL_IMAGE_INSPECTION', 'dummy_only': qualify, 'formal_data_reads': 0 if qualify else 'Existing completed scoring JSON only', 'additional_optimizer_updates': 0, 'inputs_unchanged_sha256': inputs, 'script_sha256': sha(__file__), 'contract_sha256': sha(ROOT / 'visual-composer/visual-contract.md'), 'outputs': outputs, 'plotted_values_sha256': sha(target / 'plotted_values.json'), 'render_geometry_sha256': sha(target / 'render_geometry.json'), 'accepted_directions': 0}
    dump(target / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'figures': len(outputs), 'manifest_sha256': sha(target / 'manifest.json')}), flush=True)

if __name__ == '__main__':
    main()
