"""Descriptive scientific plot of completed means and paired reveal targets."""
import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parent / 'cycle29/scripts'))
import runtime_profile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a = json.loads((R / 'outputs/analysis_v1/analysis.json').read_text(encoding='utf-8'))
v = json.loads((R / 'outputs/eval_v1/verification.json').read_text(encoding='utf-8'))
assert v['status'] == 'PASS_REAL_PROCESS_MEMORY_SCREEN'
methods = ['node', 'fixed_dde', 'learned_dde', 'gru', 'ridge', 'persistence', 'seasonal24']
labels = ['NODE', 'Fixed DDE', 'Learned DDE', 'GRU', 'Ridge', 'Persistence', 'Seasonal24']
lookup = {(x['case'], x['method'], x['reveal']): x for x in a['aggregate']}
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for row, case in enumerate(['ett', 'jena']):
    x = np.arange(len(methods))
    means = [lookup[case, m, 0]['mse_mean'] for m in methods]
    axes[row, 0].bar(x, means, color='#4c78a8')
    for j, m in enumerate(methods[:4]):
        values = [lookup[case, m, 0]['mse_min_seed'], lookup[case, m, 0]['mse_max_seed']]
        axes[row, 0].scatter([j - .08, j + .08], values, color='black', s=14, zorder=3)
    before = [lookup[case, m, 0]['tail24_mse_mean'] for m in methods]
    after = [lookup[case, m, 24]['tail24_mse_mean'] for m in methods]
    axes[row, 1].bar(x - .19, before, width=.38, label='No new observations', color='#a5b8cf')
    axes[row, 1].bar(x + .19, after, width=.38, label='True first24h observed', color='#f28e2b')
    axes[row, 0].set_title(case.upper() + ': full48h forecast')
    axes[row, 1].set_title(case.upper() + ': identical target hours25-48')
    for ax in axes[row]:
        ax.set_xticks(x, labels, rotation=28, ha='right')
        ax.set_ylabel('TRAIN-standardized MSE')
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
axes[0, 1].legend(frameon=False, fontsize=8)
fig.suptitle('Existing memory mechanisms: two real processes, two neural training seeds\nDescriptive evidence; chronological origins are dependent', fontsize=12)
folder = R / 'outputs/analysis_v1'
fig.savefig(folder / 'memory_comparison.png', dpi=160)
fig.savefig(folder / 'memory_comparison.svg')
plt.close(fig)
print('Completed-score PNG/SVG saved; no new fit or selection')
