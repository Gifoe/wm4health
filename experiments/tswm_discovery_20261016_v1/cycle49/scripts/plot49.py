"""Descriptive physical-law quality, matched reveal targets and actual fit cost."""
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

analysis = json.loads((R / 'outputs/analysis_v1/analysis.json').read_text(encoding='utf-8'))
verification = json.loads((R / 'outputs/eval_v1/verification.json').read_text(encoding='utf-8'))
assert verification['status'] == 'PASS_FRESH_CONDITIONAL_WORLD_SCREEN'
lookup = {(g['case'], g['arm'], g['density'], g['reveal']): g for g in analysis['aggregate']}
densities = [.25, 1., 4.]
arms = ['rb_score', 'full_clock_elbo']
colors = ['#4c78a8', '#f28e2b']
names = ['Observation-averaged score', 'Complete-clock ELBO']
fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), constrained_layout=True)
for row, case in enumerate(['rotating', 'bistable']):
    x = np.arange(3)
    for i, (arm, color, name) in enumerate(zip(arms, colors, names)):
        mean = np.array([lookup[case, arm, d, 0]['energy_mean'] for d in densities])
        low = np.array([lookup[case, arm, d, 0]['energy_min_seed'] for d in densities])
        high = np.array([lookup[case, arm, d, 0]['energy_max_seed'] for d in densities])
        axes[row, 0].bar(x + (i - .5) * .36, mean, width=.36, color=color, label=name,
                         yerr=np.stack((mean - low, high - mean)), capsize=3)
        for reveal, style in [(0, '--'), (12, '-')]:
            values = [lookup[case, arm, d, reveal]['tail12_energy_mean'] for d in densities]
            axes[row, 1].plot(x, values, style, color=color, marker='o' if reveal else 'x',
                              label=name + ('; new observations' if reveal else '; original history'))
    axes[row, 0].set_title(case + ': complete24-bin future')
    axes[row, 1].set_title(case + ': identical final12-bin targets')
    for col in [0, 1]:
        axes[row, col].set_xticks(x, ['0.25', '1', '4'])
        axes[row, col].set_xlabel('Sensor density multiplier')
        axes[row, col].set_ylabel('Physical future energy score (lower is better)')
        axes[row, col].set_ylim(bottom=0)
    costs = [c for c in analysis['fit_costs'] if c['case'] == case]
    costs.sort(key=lambda c: (arms.index(c['arm']), c['seed']))
    xx = np.arange(4)
    axes[row, 2].bar(xx, [c['terminal_fit_seconds'] for c in costs], color=[colors[arms.index(c['arm'])] for c in costs], alpha=.3, label='Whole fit')
    axes[row, 2].bar(xx, [c['selected_training_and_selection_seconds'] for c in costs], color=[colors[arms.index(c['arm'])] for c in costs], label='Until selected step')
    axes[row, 2].set_xticks(xx, ['Score s1', 'Score s2', 'ELBO s1', 'ELBO s2'], rotation=20)
    axes[row, 2].set_ylabel('Measured training + selection seconds')
    axes[row, 2].set_title(case + ': actual fit cost')
    for ax in axes[row]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 1].legend(frameon=False, fontsize=7)
axes[0, 2].legend(frameon=False, fontsize=8)
fig.suptitle('Conditional stochastic-world learning: two controlled process families\nBars/lines average two fitted seeds; whiskers are seed ranges. Density views share physical episodes.', fontsize=12)
folder = R / 'outputs/analysis_v1'
fig.savefig(folder / 'conditional_world_comparison.png', dpi=160)
fig.savefig(folder / 'conditional_world_comparison.svg')
plt.close(fig)
print('Verified-score PNG/SVG saved; no new fit or selection')
