"""Render complete factual metrics and training diagnostics without ranking models."""
from native53 import *
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = ART / 'analysis_v1'; OUT = A / 'figures'
summary = json.loads((A / 'summary.json').read_text())
assert summary['status'] == 'DESCRIPTIVE_ANALYSIS_COMPLETE_NO_RESEARCH_DIRECTION_ACCEPTED'
assert not OUT.exists(); OUT.mkdir()
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42})
colors = {'obs': '#5470A5', 'ae_concat': '#B77A36', 'ae_gate': '#278578', 'ae_gate_soft': '#915A9F'}
labels = {'obs': 'Observation', 'ae_concat': 'AE concat', 'ae_gate': 'AE Gate', 'ae_gate_soft': 'AE Gate + DS'}
arms = list(colors); models = ['CrossLinear', 'PatchTST']; banks = ['source_val', 'new_crop_all']
bank_labels = {'source_val': 'Source validation (163 windows)', 'new_crop_all': 'New crop (803 windows; repeated compartments)'}

def save(fig, name, caption):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    (OUT / f'{name}.md').write_text(caption + '\n', encoding='utf-8')

metrics = pd.read_csv(A / 'aggregate_metrics.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
for ax, bank in zip(axes, banks):
    frame = metrics[metrics.bank == bank]
    for i, arm in enumerate(arms):
        sub = frame[(frame.model != 'ordinary_control') & (frame.arm == arm)].set_index('model').loc[models]
        ax.bar(np.arange(2) + (i - 1.5) * .17, sub.MAE_mean, .16, yerr=sub.MAE_std,
               color=colors[arm], label=labels[arm], capsize=3)
    for control, style in [('persistence', '--'), ('ridge', ':')]:
        value = float(frame[frame.arm == control].iloc[0].MAE_mean)
        ax.axhline(value, color='#555555', linestyle=style, label=control)
    ax.set_xticks(range(2), models); ax.set_ylabel('Normalized factual MAE')
    ax.set_title(bank_labels[bank]); ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.15)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.08))
fig.tight_layout()
save(fig, 'factual_mae', 'Bars show all five training-seed means; error bars are sample standard deviations across those seeds, not independent-system uncertainty. Dashed/dotted lines are the single predeclared persistence/ridge controls. Future action/exogenous schedules are supplied to models. The crop comparison is a transfer audit on repeated compartments, not independent-domain confirmation. PatchTST AE concat uses the disclosed effective-batch microbatch adaptation.')

details = pd.read_csv(A / 'all_channel_horizons.csv')
channels = ['Tair', 'Rhair', 'CO2air', 'PARin']
for bank in banks:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for r, model in enumerate(models):
        for c, channel in enumerate(channels):
            ax = axes[r, c]
            for arm in arms:
                sub = details[(details.bank == bank) & (details.model == model) & (details.arm == arm) & (details.channel == channel)]
                pivot = sub.pivot(index='horizon_step', columns='seed', values='MAE').sort_index()
                assert pivot.shape == (16, 5)
                x = pivot.index.to_numpy() * 5; mean = pivot.mean(1).to_numpy(); sd = pivot.std(1).to_numpy()
                ax.plot(x, mean, color=colors[arm], label=labels[arm]); ax.fill_between(x, mean - sd, mean + sd, color=colors[arm], alpha=.1)
            for control, style in [('persistence', '--'), ('ridge', ':')]:
                sub = details[(details.bank == bank) & (details.arm == control) & (details.channel == channel)].sort_values('horizon_step')
                ax.plot(sub.horizon_step.to_numpy() * 5, sub.MAE, style, color='#555555', label=control)
            ax.set_title(f'{model}: {channel}'); ax.grid(alpha=.15); ax.set_ylim(bottom=0)
            if c == 0: ax.set_ylabel('Normalized MAE')
            if r == 1: ax.set_xlabel('Nominal horizon (minutes)')
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, -.025))
    fig.suptitle(bank_labels[bank]); fig.tight_layout(rect=(0, .02, 1, .95))
    save(fig, f'channel_horizons_{bank}', 'All four channels and all 16 horizons are shown. Bands represent training-seed sample standard deviation only. Five minutes is the nominal native sampling interval; accepted sessions can contain smaller-than-cutoff gaps. No channel or horizon selected a checkpoint or method.')

contrasts = json.loads((A / 'paired_contrasts.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
comparison_pairs = [('obs', 'ae_concat'), ('ae_concat', 'ae_gate'), ('ae_gate', 'ae_gate_soft')]
for ax, bank in zip(axes, banks):
    ylabels = []
    for i, (model, pair) in enumerate((m, p) for m in models for p in comparison_pairs):
        row = next(r for r in contrasts if r['bank'] == bank and r['model'] == model and (r['baseline'], r['candidate']) == pair and r['calendar_block_days'] == 7)
        center = row['percent_MAE_reduction']; lo, hi = row['descriptive_percent_reduction_interval']
        ax.plot([lo, hi], [i, i], color=colors[pair[1]], linewidth=2)
        ax.scatter([center], [i], color=colors[pair[1]], s=26, zorder=3)
        ylabels.append(f'{model}: {labels[pair[0]]} to {labels[pair[1]]}')
    ax.axvline(0, color='#666666', linewidth=.8); ax.set_yticks(range(6), ylabels); ax.invert_yaxis()
    ax.set_xlabel('MAE reduction (%) — positive favors candidate'); ax.set_title(bank_labels[bank]); ax.grid(axis='x', alpha=.15)
fig.tight_layout()
save(fig, 'paired_calendar_block_contrasts', 'Points are paired five-seed-average factual MAE reductions; lines are descriptive 95% synchronized seven-day block-bootstrap percentile intervals from 2,000 draws. The source/new crop have only seven/six occupied calendar bins. The intervals condition on these compartments and crop periods; they are not population-domain confidence statements. Three-day sensitivity and comparisons against both ordinary controls are supplied in paired_contrasts.json.')

seal = json.loads((R / 'outputs/SELECTION_SEAL_V6.json').read_text())
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for r, model in enumerate(models):
    for c, arm in enumerate(arms):
        ax = axes[r, c]
        for seed in range(5):
            tag = f'{model}__{arm}__seed{seed}'
            selected = next(s for s in seal['selected_checkpoints'] if s['tag'] == tag)
            path = pathlib.Path(selected['summary_path']); fit = json.loads(path.read_text())
            curve = json.loads((path.parent / 'curve.json').read_text())
            val = [x['val']['total'] if isinstance(x['val'], dict) else x['val'] for x in curve]
            x = np.arange(1, len(val) + 1); line, = ax.plot(x, val, label=f'seed {seed}', alpha=.8)
            e = fit['best_epoch']; ax.scatter([e], [val[e - 1]], s=14, color=line.get_color())
        ax.set_title(f'{model}: {labels[arm]}'); ax.set_yscale('log'); ax.grid(alpha=.15)
        if c == 0: ax.set_ylabel('Native validation selection objective')
        if r == 1: ax.set_xlabel('Epoch')
handles, names = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, names, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=(0, .02, 1, 1))
save(fig, 'source_selection_curves', 'Every training seed and observed epoch is retained; dots show selected epochs. Objectives use the native arm-specific validation reduction, including directional-supervision totals when present, so values across arms need not have identical loss semantics. Curves diagnose cap selection or instability; they do not change the completed training budget.')

files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
dump(OUT / 'manifest.json', files)
print(json.dumps({'status': 'RENDERED_COMPLETE_DESCRIPTIVE_FIGURES', 'files': len(files), 'directory': str(OUT)}), flush=True)
