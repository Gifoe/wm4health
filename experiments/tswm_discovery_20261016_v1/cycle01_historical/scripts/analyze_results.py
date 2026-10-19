"""Descriptive aggregation only. No checkpoint/method selection from test rows."""
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'

def main():
    runs=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/result.json'))]
    groups=[]
    for name in sorted({r['model'] for r in runs}):
        rows=[r for r in runs if r['model']==name]
        vals={
          'factual_mse':[r['factual_mse'] for r in rows],
          'effect_nmse':[np.mean([p['effect_nmse'] for p in r['pulses']]) for r in rows],
          'late_tail_gain':[np.mean([p['late_tail_gain'] for p in r['pulses']]) for r in rows],
          'whole_plan_positive_fraction':[r['whole_plan_shift_positive_fraction'] for r in rows]}
        groups.append({'model':name,'seeds':[r['seed'] for r in rows],
             'parameters':rows[0]['parameters'],'train_seconds_total':sum(r['train_seconds'] for r in rows),
             **{k:{'mean':float(np.mean(v)),'sample_std':float(np.std(v,ddof=1)),'raw':v} for k,v in vals.items()}})
    d=json.loads((OUT/'shared_parameter_confirmation.json').read_text());p=json.loads((OUT/'shared_parameter_confirmation_plan.json').read_text())
    n=p['primary_cell']['training_n'];H=p['primary_cell']['horizon']
    a=sorted([r for r in d['records'] if r['training_n']==n and r['horizon']==H and r['method']=='joint_paired_tube'],key=lambda r:r['rep'])
    b=sorted([r for r in d['records'] if r['training_n']==n and r['horizon']==H and r['method']=='joint_shared_parameter_tube'],key=lambda r:r['rep'])
    aw=np.array([r['mean_width'] for r in a]);bw=np.array([r['mean_width'] for r in b])
    ac=np.array([r['sign_certifiable_fraction'] for r in a]);bc=np.array([r['sign_certifiable_fraction'] for r in b])
    rg=np.random.default_rng(8192);idx=rg.integers(0,len(a),(4000,len(a)))
    widths=1-bw[idx].mean(1)/aw[idx].mean(1);signs=(bc-ac)[idx].mean(1)
    width=1-bw.mean()/aw.mean();sign=(bc-ac).mean();cover=np.mean([r['all_episodes_family_covered'] for r in b])
    gates=p['screening_gates'];passes={'width':bool(width>=gates['width_reduction_vs_paired_tube_minimum']),
           'sign':bool(sign>=gates['sign_certification_increase_minimum_absolute']),
           'coverage':bool(cover>=gates['all_episode_family_coverage_minimum'])}
    result={'synthetic_screen':groups,'new_neural_training_jobs':len(runs)+3,
            'shared_parameter_primary_cell':{'training_n':n,'horizon':H,
                'width_reduction':float(width),'width_reduction_paired_bootstrap95':np.quantile(widths,[.025,.975]).tolist(),
                'sign_certification_increase_absolute':float(sign),'sign_increase_paired_bootstrap95':np.quantile(signs,[.025,.975]).tolist(),
                'all_episode_family_coverage':float(cover),'gate_passes':passes,'overall_gate_pass':all(passes.values()),
                'resampling_unit':'independent training-data repetition, NOT individual trajectory or random seed of the same fitted model'},
            'interpretation':'Descriptive confirmation of a limited prototype, not a submission-ready novelty or all-SCM guarantee.'}
    (OUT/'analysis_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
